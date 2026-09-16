package demo.uryson

import demo.format.Fmt
import numerics.GaussLegendre
import numerics.orders
import problems.uryson.NoiseNorm
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import solvers.core.FirstKindSolution
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import solvers.uryson.UrysonSecondKindSolver

/**
 * Demonstration printout of convergence tables for the nonlinear Uryson equation.
 *
 * This is not part of the library but an illustration of its use: the code builds solvers on
 * a sequence of refined grids, computes the error `E_h`, the observed
 * order `p_h`, and prints the result to the console and as LaTeX lines in the format
 * of the tables of the new-01 paper (`booktabs` + `siunitx`, columns `S[table-format=1.3e-2]`).
 *
 * Reliability rule (as in new-01): values `E_h < 1e-12` are determined by
 * round-off and are printed as a dash, as are the orders `p_h` for which at least one
 * of the two neighbouring errors is below the threshold or the next grid is missing.
 */
object Tables {
    /** Sequence of grids: each next one is twice as fine as the previous one. */
    private val GRID_SIZES = listOf(8, 16, 32, 64, 128)

    /**
     * Fixed seed of the noise generator for the problems of the first kind.
     *
     * An explicit constant makes the numerical experiment fully reproducible:
     * a repeated run yields the same noisy data and the same tables.
     */
    const val SEED = 20240517L

    /** Quadrature for all demonstrations: the order is deliberately above the approximation order. */
    private val quad = GaussLegendre(8)

    /** Expected convergence order of the base scheme — used to estimate the constant `C_h`. */
    private const val EXPECTED_ORDER = 3.0

    /** Reliability threshold: below it the error is determined by round-off (new-01, sec:protocol). */
    private const val RELIABILITY_FLOOR = 1e-12

    /** Ratio of neighbouring steps of the non-uniform grid (new-01, eq:alt-mesh: mu = 2). */
    private const val GRADED_RATIO = 2.0

    /** Noise normalization for the problems of the first kind: in `C[a,b]`, consistent with condition (VII). */
    private val NOISE_NORM = NoiseNorm.SUP

    // ------------------------------------------------------------------ LaTeX format

    /** Dash in a cell of an `S` column. */
    private const val DASH = "\\multicolumn{1}{c}{---}"

    /** Error in the format `1.014e-4`, or a dash below the threshold. */
    private fun texE(x: Double): String =
        if (x.isNaN() || x < RELIABILITY_FLOOR) DASH else "%.3e".format(x).replace("e-0", "e-").replace("e+0", "e")

    /**
     * Order `p_h` from a pair of neighbouring errors: a dash if the next grid is missing
     * or one of the errors is below the threshold.
     */
    private fun texP(errs: List<Double>, i: Int): String {
        if (i + 1 >= errs.size) return DASH
        if (errs[i] < RELIABILITY_FLOOR || errs[i + 1] < RELIABILITY_FLOOR) return DASH
        return "%.2f".format(Math.log(errs[i] / errs[i + 1]) / Math.log(2.0))
    }

    private fun gridOf(kind: String, n: Int): Grid =
        if (kind == "uniform") Grid.uniform(n) else Grid.graded(n, ratio = GRADED_RATIO)

    /** Builds a second-kind solver for the given problem, basis and grid. */
    private fun makeSolver(problem: UrysonProblem, system: GeneratingSystem, grid: Grid): UrysonSecondKindSolver {
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, space, op)
    }

    // ------------------------------------------------------------------ second kind

    /** Convergence of the base scheme for problem A on the bases B and H, uniform and non-uniform grids. */
    fun tableSecondKindOrder() {
        for (kind in listOf("uniform", "graded")) {
            println("\n--- Problem A (second kind): base scheme, $kind grid ---")
            val errorsB = ArrayList<Double>()
            val errorsH = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val grid = gridOf(kind, n)
                val exact = { t: Double -> UrysonProblem.A.exact(t) }
                errorsB.add(errorEh(exact, makeSolver(UrysonProblem.A, GeneratingSystem.B, grid).base().eval, grid))
                errorsH.add(errorEh(exact, makeSolver(UrysonProblem.A, GeneratingSystem.H, grid).base().eval, grid))
            }
            val ordersB = orders(errorsB)
            val ordersH = orders(errorsH)
            println("   n |   Eh(B)   | ph(B) |   Eh(H)   | ph(H)")
            for (i in GRID_SIZES.indices) {
                println(
                    "%4d | %s | %5s | %s | %5s".format(
                        GRID_SIZES[i], Fmt.e(errorsB[i]), Fmt.p(ordersB[i]), Fmt.e(errorsH[i]), Fmt.p(ordersH[i]),
                    ),
                )
            }
            println("   LaTeX (S columns):")
            for (i in GRID_SIZES.indices) {
                println("    ${GRID_SIZES[i]} & ${texE(errorsB[i])} & ${texP(errorsB, i)} & ${texE(errorsH[i])} & ${texP(errorsH, i)} \\\\")
            }
        }
    }

    /** Comparison of the three generating systems on problem B (base scheme). */
    fun tableGeneratingSystems() {
        println("\n--- Problem B (second kind): three generating systems, uniform grid ---")
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val errors = systems.map { ArrayList<Double>() }
        val steps = ArrayList<Double>()
        for (n in GRID_SIZES) {
            val grid = Grid.uniform(n)
            steps.add(grid.h)
            for ((index, system) in systems.withIndex()) {
                val solver = makeSolver(UrysonProblem.B, system, grid)
                errors[index].add(errorEh({ t -> UrysonProblem.B.exact(t) }, solver.base().eval, grid))
            }
        }
        val observedOrders = systems.indices.map { orders(errors[it]) }
        println("   n |    h    | [B: Eh ph Ch] | [H: Eh ph Ch] | [T: Eh ph Ch]")
        for (i in GRID_SIZES.indices) {
            val columns = systems.indices.joinToString(" | ") { si ->
                "%s %5s %s".format(
                    Fmt.e(errors[si][i]), Fmt.p(observedOrders[si][i]),
                    Fmt.e(errors[si][i] / Math.pow(steps[i], EXPECTED_ORDER)),
                )
            }
            println("%4d | %s | %s".format(GRID_SIZES[i], Fmt.h(steps[i]), columns))
        }
        println("   LaTeX (S columns: n, [Eh ph Ch] x3):")
        for (i in GRID_SIZES.indices) {
            val cells = systems.indices.joinToString(" & ") { si ->
                val e = errors[si][i]
                val ch = if (e < RELIABILITY_FLOOR) DASH else "%.3e".format(e / Math.pow(steps[i], EXPECTED_ORDER)).replace("e-0", "e-")
                "${texE(e)} & ${texP(errors[si], i)} & $ch"
            }
            println("    ${GRID_SIZES[i]} & $cells \\\\")
        }
    }

    /** All second-kind schemes on problem B for the bases B and H. */
    fun tableSchemes() {
        val names = listOf("base", "Sloan", "Kulkarni", "iter. Kulkarni", "Nyström", "comb. Nyström", "iter. Nyström")
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
            println("\n--- Problem B: comparison of schemes, basis ${system.name}, uniform grid ---")
            val errors = names.map { ArrayList<Double>() }
            for (n in GRID_SIZES) {
                val grid = Grid.uniform(n)
                val solver = makeSolver(UrysonProblem.B, system, grid)
                val exact = { t: Double -> UrysonProblem.B.exact(t) }
                errors[0].add(errorEh(exact, solver.base().eval, grid))
                errors[1].add(errorEh(exact, solver.sloan().eval, grid))
                errors[2].add(errorEh(exact, solver.kulkarni().eval, grid))
                errors[3].add(errorEh(exact, solver.iteratedKulkarni().eval, grid))
                errors[4].add(errorEh(exact, solver.nystrom().eval, grid))
                errors[5].add(errorEh(exact, solver.combinedNystrom().eval, grid))
                errors[6].add(errorEh(exact, solver.iteratedCombinedNystrom().eval, grid))
            }
            val observedOrders = names.indices.map { orders(errors[it]) }
            println("   n | " + names.joinToString(" | ") { "$it: Eh ph" })
            for (i in GRID_SIZES.indices) {
                val columns = names.indices.joinToString(" | ") { mi -> "%s %5s".format(Fmt.e(errors[mi][i]), Fmt.p(observedOrders[mi][i])) }
                println("%4d | %s".format(GRID_SIZES[i], columns))
            }
            println("   LaTeX (S columns: n, [Eh ph] x7 in the order: ${names.joinToString(", ")}):")
            for (i in GRID_SIZES.indices) {
                val cells = names.indices.joinToString(" & ") { mi -> "${texE(errors[mi][i])} & ${texP(errors[mi], i)}" }
                println("    ${GRID_SIZES[i]} & $cells \\\\")
            }
        }
    }

    // ------------------------------------------------------------------ first kind

    private fun solveFirstKind(problem: UrysonProblem, system: GeneratingSystem, grid: Grid, delta: Double): FirstKindSolution {
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta = noisyThetaCoefficients(problem, solver, op, grid, quad, delta, SEED, NOISE_NORM)
        return solver.solveMorozov(thetaFDelta, delta)
    }

    /**
     * Regularized solution of problem C of the first kind with consistently decreasing noise
     * level and grid step. The grid is non-uniform with alternating steps (new-01, eq:alt-mesh);
     * the noise is given in `C[a,b]` (`||xi||_infty = delta`).
     */
    fun tableFirstKindNoiseLevels() {
        println("\n--- Problem C (first kind, regularization): basis H, graded(2) grid, SUP noise ---")
        val deltas = listOf(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        println("  delta |  n  |   alpha   |    Eh     |   res     |   Omega")
        val rows = ArrayList<String>()
        for (i in deltas.indices) {
            val delta = deltas[i]
            val n = GRID_SIZES[i]
            val grid = Grid.graded(n, ratio = GRADED_RATIO)
            val solution = solveFirstKind(UrysonProblem.C, GeneratingSystem.H, grid, delta)
            val error = errorEh({ t -> UrysonProblem.C.exact(t) }, solution.eval, grid)
            println(
                "  %s | %3d | %s | %s | %s | %s".format(
                    Fmt.e(delta), n, Fmt.e(solution.alpha), Fmt.e(error), Fmt.e(solution.residual), Fmt.e(solution.omega),
                ),
            )
            rows.add("    ${texE(delta)} & $n & ${texE(solution.alpha)} & ${texE(error)} & ${texE(solution.residual)} & ${"%.3f".format(solution.omega)} \\\\")
        }
        println("   LaTeX (S columns: delta, n, alpha, Eh, res, Omega):")
        rows.forEach { println(it) }
    }

    /** Comparison of the bases B and H on problem D of the first kind at a fixed noise level. */
    fun tableFirstKindBasisComparison() {
        println("\n--- Problem D (first kind): basis B versus H, delta = 1e-3 (SUP), graded(2) grid ---")
        val delta = 1e-3
        println("   n  |   alpha   |  Eh(B)   |  res(B)   |  Eh(H)   |  res(H)")
        val rows = ArrayList<String>()
        for (n in GRID_SIZES) {
            val grid = Grid.graded(n, ratio = GRADED_RATIO)
            val solutionB = solveFirstKind(UrysonProblem.D, GeneratingSystem.B, grid, delta)
            val solutionH = solveFirstKind(UrysonProblem.D, GeneratingSystem.H, grid, delta)
            val exact = { t: Double -> UrysonProblem.D.exact(t) }
            val errorB = errorEh(exact, solutionB.eval, grid)
            val errorH = errorEh(exact, solutionH.eval, grid)
            println(
                "  %3d | %s | %s | %s | %s | %s".format(
                    n, Fmt.e(solutionB.alpha), Fmt.e(errorB), Fmt.e(solutionB.residual), Fmt.e(errorH), Fmt.e(solutionH.residual),
                ),
            )
            rows.add("    $n & ${texE(solutionB.alpha)} & ${texE(errorB)} & ${texE(solutionB.residual)} & ${texE(solutionH.alpha)} & ${texE(errorH)} & ${texE(solutionH.residual)} \\\\")
        }
        println("   LaTeX (S columns: n, alpha(B), Eh(B), res(B), alpha(H), Eh(H), res(H)):")
        rows.forEach { println(it) }
    }
}

/** Entry point of the demonstration: prints convergence tables for the Uryson equation. */
fun main() {
    println("=".repeat(72))
    println("Nonlinear Uryson equation: convergence tables")
    println("Noise generator seed (problems of the first kind): ${Tables.SEED}; noise normalization: SUP")
    println("Quadrature: Gauss--Legendre, 8 nodes per cell; reliability threshold 1e-12")
    println("=".repeat(72))
    Tables.tableSecondKindOrder()
    Tables.tableGeneratingSystems()
    Tables.tableSchemes()
    Tables.tableFirstKindNoiseLevels()
    Tables.tableFirstKindBasisComparison()
    println("\nComputation finished.")
}
