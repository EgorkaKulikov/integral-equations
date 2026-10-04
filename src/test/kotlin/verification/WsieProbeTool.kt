package verification

import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.pow
import kotlin.test.Test
import numerics.AlgebraicSingularQuadrature
import numerics.Conditioning
import numerics.GaussLegendre
import numerics.NumericsContext
import problems.wsie.WeaklySingularProblem
import problems.wsie.WeaklySingularType
import solvers.core.SolutionFunc
import solvers.wsie.ProductIntegrationOperator
import solvers.wsie.WeaklySingularFredholmOperator
import solvers.wsie.WeaklySingularOperator
import solvers.wsie.WeaklySingularSecondKindSolver
import solvers.wsie.WeaklySingularVolterraOperator
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.Reparametrization
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.functionals.ThetaSampling
import splines.functionals.ThreePointFunctionals

/**
 * PROBE DRIVER for the weakly singular second-kind equations (Gradle task `wsieProbe`).
 *
 * A GENERATOR OF NUMBERS, not a check: it has no PASS/FAIL criteria and is therefore excluded from
 * `test`, `fastTest`, `slowTest` and the Kover tasks (see `generatorTestClasses` in `build.gradle.kts`
 * and `TagConventionTest.GENERATOR_CLASSES`).
 *
 * The matrix problem × scheme × functional family χ × space × grid × n is filtered by the system
 * properties `wsie.problems`, `wsie.schemes`, `wsie.families`, `wsie.spaces`, `wsie.grids`, `wsie.n`
 * (comma lists), `wsie.quad` (nodes per subinterval of [AlgebraicSingularQuadrature], 12 by default),
 * `wsie.refine` (endpoint refinement for the space G and for Fredholm problems, 20 by default),
 * `wsie.refineOther` (endpoint refinement for the remaining rows, i.e. Volterra problems in the space B,
 * 0 by default) and
 * `wsie.cond` (`false` switches off the condition estimate of the base matrix).
 *
 * Schemes: `base`, `kulkarni`, `iteratedKulkarni`, `sloan` (methods of [WeaklySingularSecondKindSolver]),
 * `kulkarni2grid` (two-grid Kulkarni: coarse grid with n cells, nested fine grid with n·p cells of the same type and
 * grading, p = `wsie.p`, 2 by default; the fine solver is built exactly as the coarse one) and `kulkarniDiscrete`
 * (discrete Kulkarni with [ProductIntegrationOperator] on the grid Y_m of the same type and grading with
 * m = n·`wsie.mp` cells, `wsie.mp` = 2 by default, and `wsie.q` nodes per cell, 8 by default).
 * Columns `p`, `mp`, `q` are 0 for the schemes that do not use them; `costMs` equals `ms` (construction + solve).
 * For `kulkarni2grid` the row also carries `EhKulkFine` and `msKulkFine`: E_h and construction + solve time of the
 * classical Kulkarni scheme on the fine grid (solver built afresh, error on the control set of the fine grid, as in
 * a `kulkarni` row with n·p cells); NaN for the other schemes.
 *
 * Every row is computed independently (grid, basis, operator, functionals and solver are rebuilt), an exception
 * turns the row into `FAILED:<message>` and the loop continues. The observed orders
 * `log2(E_n / E_{2n})` are numerical observations, not proven rates.
 *
 * Output (fixed path, relative to the project directory): `build/wsie-probe/probe.tsv`, `probe.json`, `meta.txt`.
 */
class WsieProbeTool {

    private data class Row(
        val problem: String,
        val scheme: String,
        val family: String,
        val space: String,
        val grid: String,
        val r: Double,
        val n: Int,
        val quad: Int,
        val refine: Int,
        var eh: Double = Double.NaN,
        var orderEh: Double = Double.NaN,
        var condInf: Double = Double.NaN,
        var lambda: Double = Double.NaN,
        var ms: Double = Double.NaN,
        var msEval: Double = Double.NaN,
        var converged: Boolean = false,
        var iterations: Int = 0,
        var status: String = "OK",
        var pRef: Int = 0,
        var mp: Int = 0,
        var q: Int = 0,
        var ehKulkFine: Double = Double.NaN,
        var msKulkFine: Double = Double.NaN,
    )

    private val ctx = NumericsContext.default()

    private fun listProp(key: String, default: String): List<String> =
        (System.getProperty(key)?.takeIf { it.isNotBlank() } ?: default)
            .split(',').map { it.trim() }.filter { it.isNotEmpty() }

    private val problemsById = WeaklySingularProblem.ALL.associateBy { it.name }

    // ==================== construction ====================

    private fun repOf(p: WeaklySingularProblem): Reparametrization = Reparametrization.power(1.0 - p.alpha)

    private fun systemOf(space: String, p: WeaklySingularProblem): GeneratingSystem = when (space) {
        "B" -> GeneratingSystem.B
        "G" -> GeneratingSystem.reparametrized(repOf(p))
        else -> error("unknown space $space")
    }

    private fun gridOf(label: String, r: Double, n: Int, p: WeaklySingularProblem): Grid = when {
        label == "uniform" -> Grid.uniform(n)
        p.type == WeaklySingularType.VOLTERRA -> Grid.power(n, r = r)
        else -> Grid.symmetricPower(n, r = r)
    }

    private fun gradingOf(label: String): Double = when {
        label == "uniform" -> 1.0
        label.startsWith("power:") -> label.removePrefix("power:").toDouble()
        else -> error("unknown grid $label (expected uniform or power:r)")
    }

    private fun operatorOf(p: WeaklySingularProblem, grid: Grid, quad: Int, refine: Int): WeaklySingularOperator {
        val q = AlgebraicSingularQuadrature(p.alpha, quad)
        return when (p.type) {
            WeaklySingularType.VOLTERRA -> WeaklySingularVolterraOperator(p.kernel, grid, q, refine)
            WeaklySingularType.FREDHOLM -> WeaklySingularFredholmOperator(p.kernel, grid, q, refine)
        }
    }

    private fun familyOf(family: String, basis: MinimalSplineBasis, p: WeaklySingularProblem): FunctionalFamily =
        when (family) {
            "theta" -> ProjFunctionals(basis, ctx)
            "theta-tau" -> ProjFunctionals(basis, ctx, ThetaSampling.reparametrized(repOf(p)))
            "lambda" -> ThreePointFunctionals(basis, ctx = ctx)
            else -> error("unknown family $family")
        }

    /** Grid, family and solver of one level; the coarse and the fine level are built by the same code. */
    private class Level(val grid: Grid, val funcs: FunctionalFamily, val solver: WeaklySingularSecondKindSolver)

    private fun levelOf(
        p: WeaklySingularProblem, sys: GeneratingSystem, family: String, gLabel: String, r: Double, n: Int,
        quad: Int, refine: Int,
    ): Level {
        val grid = gridOf(gLabel, r, n, p)
        val basis = MinimalSplineBasis(sys, grid, ctx)
        val op = operatorOf(p, grid, quad, refine)
        val funcs = familyOf(family, basis, p)
        return Level(grid, funcs, WeaklySingularSecondKindSolver(basis, funcs, op, p.cL, p.rhs, ctx))
    }

    private fun runScheme(scheme: String, s: WeaklySingularSecondKindSolver): SolutionFunc = when (scheme) {
        "base" -> s.base()
        "kulkarni" -> s.kulkarni()
        "iteratedKulkarni" -> s.iteratedKulkarni()
        "sloan" -> s.sloan()
        else -> error("unknown scheme $scheme")
    }

    // ==================== control set ====================

    /**
     * Knots ∪ cell midpoints ∪ 8 Gauss–Legendre points per cell ∪ 200 points `t = 10^(−12 + 12k/199)`
     * (mirrored `1 − t` added for Fredholm problems), restricted to `[a, b]`.
     */
    private fun controlSet(grid: Grid, p: WeaklySingularProblem): DoubleArray {
        val pts = ArrayList<Double>()
        val bp = grid.breakpoints
        val gl = GaussLegendre(8).refNodesWeights().first
        for (i in 0 until bp.size - 1) {
            val lo = bp[i]
            val hi = bp[i + 1]
            pts.add(lo)
            pts.add(0.5 * (lo + hi))
            for (x in gl) pts.add(0.5 * (lo + hi) + 0.5 * (hi - lo) * x)
        }
        pts.add(bp.last())
        for (k in 0 until 200) {
            val t = 10.0.pow(-12.0 + 12.0 * k / 199.0)
            pts.add(grid.a + t * (grid.b - grid.a))
            if (p.type == WeaklySingularType.FREDHOLM) pts.add(grid.b - t * (grid.b - grid.a))
        }
        return pts.filter { it >= grid.a && it <= grid.b }.distinct().sorted().toDoubleArray()
    }

    private fun maxError(points: DoubleArray, u: (Double) -> Double, uh: (Double) -> Double): Double {
        var m = 0.0
        for (t in points) {
            val e = abs(u(t) - uh(t))
            if (e.isNaN()) return Double.NaN
            if (e > m) m = e
        }
        return m
    }

    // ==================== probe ====================

    @Test
    fun probe() {
        val t0 = System.nanoTime()
        val problems = listProp("wsie.problems", "V-a,V-b,V-c,F-a")
        val schemes = listProp("wsie.schemes", "base,kulkarni,iteratedKulkarni")
        val families = listProp("wsie.families", "theta,theta-tau,lambda")
        val spaces = listProp("wsie.spaces", "B,G")
        val grids = listProp("wsie.grids", "uniform,power:2,power:3,power:4")
        val ns = listProp("wsie.n", "8,16,32,64").map { it.toInt() }
        val quad = System.getProperty("wsie.quad")?.toInt() ?: 12
        val refineSingular = System.getProperty("wsie.refine")?.toInt() ?: 20
        val refineOther = System.getProperty("wsie.refineOther")?.toInt() ?: 0
        val withCond = System.getProperty("wsie.cond")?.toBoolean() ?: true
        val pRef = System.getProperty("wsie.p")?.toInt() ?: 2
        val mpRef = System.getProperty("wsie.mp")?.toInt() ?: 2
        val qNodes = System.getProperty("wsie.q")?.toInt() ?: 8

        val rows = ArrayList<Row>()
        for (pid in problems) {
            val p = problemsById[pid] ?: error("unknown problem $pid; known: ${problemsById.keys}")
            for (space in spaces) {
                if (space == "G" && p.type == WeaklySingularType.FREDHOLM) continue
                val refine = if (space == "G" || p.type == WeaklySingularType.FREDHOLM) refineSingular else refineOther
                for (gLabel in grids) {
                    val r = gradingOf(gLabel)
                    for (n in ns) {
                        for (family in families) {
                            if (family == "theta-tau" && space != "G") continue
                            var cond = Double.NaN
                            val configRows = ArrayList<Row>()
                            for (scheme in schemes) {
                                val row = Row(pid, scheme, family, space, gLabel, r, n, quad, refine)
                                try {
                                    val tc = System.nanoTime()
                                    val sys = systemOf(space, p)
                                    val coarse = levelOf(p, sys, family, gLabel, r, n, quad, refine)
                                    val grid = coarse.grid
                                    val funcs = coarse.funcs
                                    val solver = coarse.solver
                                    val sol = when (scheme) {
                                        "kulkarni2grid" -> {
                                            row.pRef = pRef
                                            val fine = levelOf(p, sys, family, gLabel, r, n * pRef, quad, refine)
                                            solver.twoGridKulkarni(fine.solver, pRef)
                                        }
                                        "kulkarniDiscrete" -> {
                                            row.mp = mpRef
                                            row.q = qNodes
                                            val ym = gridOf(gLabel, r, n * mpRef, p)
                                            val volterra = p.type == WeaklySingularType.VOLTERRA
                                            solver.discreteKulkarni(ProductIntegrationOperator(p.kernel, ym, qNodes, volterra))
                                        }
                                        else -> runScheme(scheme, solver)
                                    }
                                    row.ms = (System.nanoTime() - tc) / 1e6
                                    row.converged = sol.converged
                                    row.iterations = sol.iterations
                                    row.lambda = funcs.cChi()
                                    val te = System.nanoTime()
                                    row.eh = maxError(controlSet(grid, p), p.exact, sol.eval)
                                    row.msEval = (System.nanoTime() - te) / 1e6
                                    if (scheme == "kulkarni2grid") {
                                        // Reference for the two-grid scheme: classical Kulkarni on the fine grid, timed like any row.
                                        try {
                                            val tf = System.nanoTime()
                                            val fineK = levelOf(p, sys, family, gLabel, r, n * pRef, quad, refine)
                                            val solK = fineK.solver.kulkarni()
                                            row.msKulkFine = (System.nanoTime() - tf) / 1e6
                                            row.ehKulkFine = maxError(controlSet(fineK.grid, p), p.exact, solK.eval)
                                        } catch (e: Exception) {
                                            System.err.println("wsieProbe EhKulkFine failed: ${e.javaClass.simpleName}: ${e.message}")
                                        }
                                    }
                                    if (withCond && cond.isNaN()) {
                                        cond = Conditioning.conditionEstimate(solver.baseMatrix(), ctx).valueOrNull()
                                            ?: Double.NaN
                                    }
                                } catch (e: Exception) {
                                    val msg = (e.message ?: e.javaClass.simpleName).replace(Regex("[\\t\\r\\n]+"), " ")
                                    row.status = "FAILED:" + e.javaClass.simpleName + ": " + msg
                                }
                                configRows.add(row)
                                System.err.println(
                                    "wsieProbe ${row.problem} ${row.scheme} ${row.family} ${row.space} ${row.grid} " +
                                        "n=${row.n} Eh=${row.eh} ms=${row.ms} ${row.status}" +
                                        (if (row.scheme == "kulkarni2grid") " EhKulkFine=${row.ehKulkFine} msKulkFine=${row.msKulkFine}" else ""),
                                )
                            }
                            configRows.forEach { it.condInf = cond }
                            rows.addAll(configRows)
                        }
                    }
                }
            }
        }

        // Observed orders log2(E_n / E_{2n}) per (problem, scheme, family, space, grid).
        val byKey = rows.groupBy { listOf(it.problem, it.scheme, it.family, it.space, it.grid) }
        for (group in byKey.values) {
            val byN = group.associateBy { it.n }
            for (row in group) {
                val next = byN[2 * row.n] ?: continue
                if (row.status == "OK" && next.status == "OK" && row.eh > 0.0 && next.eh > 0.0) {
                    row.orderEh = ln(row.eh / next.eh) / ln(2.0)
                }
            }
        }

        val elapsed = (System.nanoTime() - t0) / 1e9
        val ncVersion = System.getProperty("wsie.meta.numericalCoreVersion") ?: "unknown"
        val msVersion = System.getProperty("wsie.meta.minimalSplinesVersion") ?: "unknown"
        val backend = ctx.describe()
        val header = "# wsieProbe numerical-core=$ncVersion minimal-splines=$msVersion $backend " +
            "numerics.backend=${System.getProperty("numerics.backend")} quad=$quad " +
            "refine(G-space,Fredholm)=$refineSingular refine(other)=$refineOther condInf=Conditioning.conditionEstimate(baseMatrix) " +
            "wsie.p=$pRef wsie.mp=$mpRef wsie.q=$qNodes"
        val columns = listOf(
            "problem", "scheme", "family", "space", "grid", "r", "n", "quad", "refine", "Eh", "orderEh",
            "condInf", "Lambda", "ms", "msEval", "converged", "iterations", "status",
            "p", "mp", "q", "costMs", "EhKulkFine", "msKulkFine",
        )
        fun f(x: Double): String = String.format(Locale.ROOT, "%.17g", x)
        fun cells(row: Row): List<String> = listOf(
            row.problem, row.scheme, row.family, row.space, row.grid, f(row.r), row.n.toString(), row.quad.toString(),
            row.refine.toString(), f(row.eh), f(row.orderEh), f(row.condInf), f(row.lambda), f(row.ms), f(row.msEval),
            row.converged.toString(), row.iterations.toString(), row.status,
            row.pRef.toString(), row.mp.toString(), row.q.toString(), f(row.ms), f(row.ehKulkFine), f(row.msKulkFine),
        )

        val dir = File(System.getProperty("user.dir")).resolve("build/wsie-probe")
        dir.mkdirs()
        File(dir, "probe.tsv").writeText(buildString {
            append(header).append('\n')
            append(columns.joinToString("\t")).append('\n')
            rows.forEach { append(cells(it).joinToString("\t")).append('\n') }
        })

        fun jsonString(s: String): String = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        fun jsonNum(x: Double): String = if (x.isNaN() || x.isInfinite()) "null" else f(x)
        File(dir, "probe.json").writeText(buildString {
            append("{\n \"header\": ").append(jsonString(header)).append(",\n \"rows\": [\n")
            append(rows.joinToString(",\n") { row ->
                "  {\"problem\": ${jsonString(row.problem)}, \"scheme\": ${jsonString(row.scheme)}, " +
                    "\"family\": ${jsonString(row.family)}, \"space\": ${jsonString(row.space)}, " +
                    "\"grid\": ${jsonString(row.grid)}, \"r\": ${jsonNum(row.r)}, \"n\": ${row.n}, " +
                    "\"quad\": ${row.quad}, \"refine\": ${row.refine}, \"Eh\": ${jsonNum(row.eh)}, " +
                    "\"orderEh\": ${jsonNum(row.orderEh)}, \"condInf\": ${jsonNum(row.condInf)}, " +
                    "\"Lambda\": ${jsonNum(row.lambda)}, \"ms\": ${jsonNum(row.ms)}, " +
                    "\"msEval\": ${jsonNum(row.msEval)}, \"converged\": ${row.converged}, " +
                    "\"iterations\": ${row.iterations}, \"status\": ${jsonString(row.status)}, " +
                    "\"p\": ${row.pRef}, \"mp\": ${row.mp}, \"q\": ${row.q}, \"costMs\": ${jsonNum(row.ms)}, " +
                    "\"EhKulkFine\": ${jsonNum(row.ehKulkFine)}, \"msKulkFine\": ${jsonNum(row.msKulkFine)}}"
            })
            append("\n ]\n}\n")
        })

        File(dir, "meta.txt").writeText(buildString {
            append(header).append('\n')
            append("problems=").append(problems.joinToString(",")).append('\n')
            append("schemes=").append(schemes.joinToString(",")).append('\n')
            append("families=").append(families.joinToString(",")).append('\n')
            append("spaces=").append(spaces.joinToString(",")).append('\n')
            append("grids=").append(grids.joinToString(",")).append('\n')
            append("n=").append(ns.joinToString(",")).append('\n')
            append("quad=").append(quad).append(" refine=").append(refineSingular).append(" refineOther=").append(refineOther).append(" cond=").append(withCond).append('\n')
            append("p=").append(pRef).append(" mp=").append(mpRef).append(" q=").append(qNodes).append('\n')
            append("java=").append(System.getProperty("java.version")).append('\n')
            append("rows=").append(rows.size).append(" failed=").append(rows.count { it.status != "OK" }).append('\n')
            append("elapsed_s=").append(String.format(Locale.ROOT, "%.1f", elapsed)).append('\n')
        })
        println("written: " + File(dir, "probe.tsv").absolutePath + " (${rows.size} rows, $elapsed s)")
    }
}
