package verification

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import java.io.File

/**
 * THE DUMP OF THE INTERNAL COMPUTATION ARTIFACTS for the external cross-check against SciPy/NumPy.
 *
 * Why exactly the internal artifacts are dumped and not only the resulting `E_h`:
 * a cross-check of "numbers against numbers" at the output would check the coincidence of two implementations
 * as a whole, and on a discrepancy it would be unclear which layer is to blame. Dumping the quadrature
 * nodes, the basis values, the assembled matrices and the operator images makes it possible
 * to check every layer SEPARATELY and to point at the place of the discrepancy.
 *
 * Extracted into a separate object (rather than left inside a test class), since it is
 * used by two consumers:
 *  - [ScipyCrossVerificationTest] — a smoke test, it prepares the data itself so as not to
 *    depend on whether a separate Gradle task was run before it;
 *  - [VerificationArtifactDumpTool] — the task `dumpVerificationArtifacts` for
 *    a manual dump and analysis of discrepancies outside a test run.
 */
object VerificationArtifacts {

    /** The default dump directory; it lies in build/, so it does not get into the repository. */
    val DEFAULT_DIR: File = File("build/verification")

    /** The grid on which the basis and the matrices are dumped: fine enough and fast. */
    private const val DUMP_GRID_SIZE = 8

    /**
     * The bounds of the dump interval.
     *
     * They are set EXPLICITLY and passed into the [Grid] factories rather than left at the default
     * values, because they are now dumped into `dump-meta.tsv` and set
     * the integration limits in the external cross-check script. Previously the script computed
     * the integrals over a hard-wired [0,1], and the agreement with the dumper was an ACCIDENTAL coincidence
     * with the default values.
     */
    private const val DUMP_A = 0.0
    private const val DUMP_B = 1.0

    /** The number of Gauss-Legendre quadrature nodes in the integral operators of the dump. */
    private const val DUMP_QUADRATURE_NODES = 8

    /** The number of basis sample points; coprime with the number of grid nodes. */
    private const val DUMP_SPLINE_SAMPLES = 97

    /** The number of sample intervals for the operator images (one more point than that). */
    private const val DUMP_OPERATOR_SAMPLES = 20

    /** The maximal m of the dumped Gauss-Legendre quadrature. */
    private const val DUMP_GAUSS_MAX_M = 16

    /** Printing a number at full precision: the cross-check goes at the level of 1e-15, rounding is not allowed. */
    private fun Double.full(): String = "%.17g".format(this)

    /** Dumps all the artifacts into the directory [dir]. */
    fun dumpAll(dir: File = DEFAULT_DIR): List<File> {
        dir.mkdirs()
        return listOf(
            dumpMeta(dir),
            dumpGaussLegendreNodes(dir),
            dumpSplineKnots(dir),
            dumpSplineValues(dir),
            dumpAssembledSystem(dir),
            dumpOperatorImages(dir),
            dumpSolutionErrors(dir),
        )
    }

    /**
     * The metadata of the dump: everything without which the external script does not know WHAT to cross-check.
     *
     * Why they are needed. The cross-check script computes the integrals `scipy.integrate.quad` over an INTERVAL,
     * and the bounds of the interval are set by the dumper. While the bounds were hard-wired in the script
     * as the literals `0.0, 1.0`, the agreement rested on the dumper not passing `a`/`b`
     * and using the default values. A change of the interval here would lead to a SILENT
     * cross-check against different integrals — that is, the cross-check would stop being a cross-check, staying
     * green or turning red without an explanation. The format `key<TAB>value` is chosen to be extensible:
     * adding a key does not break the parsing.
     */
    fun dumpMeta(dir: File): File {
        val file = File(dir, "dump-meta.tsv")
        file.printWriter().use { out ->
            out.println("# key\tvalue")
            out.println("a\t${DUMP_A.full()}")
            out.println("b\t${DUMP_B.full()}")
            out.println("dumpGridSize\t$DUMP_GRID_SIZE")
            out.println("quadratureNodes\t$DUMP_QUADRATURE_NODES")
            out.println("splineSampleCount\t$DUMP_SPLINE_SAMPLES")
            out.println("operatorSampleCount\t$DUMP_OPERATOR_SAMPLES")
            out.println("gaussLegendreMaxM\t$DUMP_GAUSS_MAX_M")
        }
        return file
    }

    /**
     * L1. The nodes and weights of the Gauss-Legendre quadrature on `[-1,1]` for m = 1..16.
     *
     * The baseline in SciPy is `numpy.polynomial.legendre.leggauss`: a fundamentally different
     * algorithm (the project uses Newton's method on the zeros of the Legendre polynomial).
     */
    fun dumpGaussLegendreNodes(dir: File): File {
        val file = File(dir, "gauss-legendre.tsv")
        file.printWriter().use { out ->
            out.println("# m\tindex\tnode\tweight")
            for (m in 1..DUMP_GAUSS_MAX_M) {
                val (nodes, weights) = GaussLegendre.gaussLegendreReference(m)
                for (i in nodes.indices) {
                    out.println("$m\t$i\t${nodes[i].full()}\t${weights[i].full()}")
                }
            }
        }
        return file
    }

    /** The grids on which the basis is cross-checked: a uniform one and three substantially non-uniform ones. */
    private fun dumpGrids(): Map<String, Grid> = linkedMapOf(
        "uniform" to Grid.uniform(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B),
        "quasiUniform" to Grid.quasiUniform(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B),
        "geometric" to Grid.geometric(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B),
        "graded" to Grid.graded(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B),
    )

    /**
     * The full knot vector `x_{-2..n+2}`.
     *
     * Knots of multiplicity 3 at the ends define exactly a clamped knot vector
     * of degree 2, so `scipy.interpolate.BSpline` builds the same space
     * on them: the number of basis functions coincides, `(n+5) - 2 - 1 = n + 2`.
     */
    fun dumpSplineKnots(dir: File): File {
        val file = File(dir, "spline-knots.tsv")
        file.printWriter().use { out ->
            out.println("# grid\tindex\tknot")
            for ((name, grid) in dumpGrids()) {
                for (j in -2..grid.n + 2) out.println("$name\t$j\t${grid.x(j).full()}")
            }
        }
        return file
    }

    /**
     * L3. The values of the minimal spline basis and of its two derivatives for
     * the polynomial system `B`.
     *
     * The cross-check is possible exactly for `B`: for the systems `H` and `T` there is no analogue in SciPy
     * (see docs/REFERENCES.md, section 6).
     */
    fun dumpSplineValues(dir: File): File {
        val file = File(dir, "spline-values.tsv")
        // The number of points is coprime with the number of knots: the sample does not fall on the joints.
        val sampleCount = DUMP_SPLINE_SAMPLES
        file.printWriter().use { out ->
            out.println("# grid\tj\tt\tomega\tomegaDeriv\tomegaDeriv2")
            for ((name, grid) in dumpGrids()) {
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                for (i in 0..sampleCount) {
                    val t = grid.a + (grid.b - grid.a) * i / sampleCount
                    for (j in -2..grid.n - 1) {
                        out.println(
                            "$name\t$j\t${t.full()}\t${basis.omega(j, t).full()}\t" +
                                "${basis.omegaDeriv(j, t).full()}\t${basis.omegaDeriv2(j, t).full()}",
                        )
                    }
                }
            }
        }
        return file
    }

    /**
     * L2. The assembled matrices `M`, `M2`, the vectors `g`, `d` and the coefficients of the base scheme.
     *
     * The baseline in SciPy is `scipy.linalg.solve` on the same system `(I - M) c = g`;
     * additionally the conditioning (`numpy.linalg.cond`) is reported, which
     * explains the attainable accuracy.
     */
    fun dumpAssembledSystem(dir: File): File {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(DUMP_QUADRATURE_NODES))
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
        )
        val m = solver.matrixM()
        val m2 = solver.matrixM2()
        val g = solver.vectorG()
        val d = solver.vectorD()
        val coeffs = solver.solveBaseCoeffs()

        val file = File(dir, "assembled-system.tsv")
        file.printWriter().use { out ->
            out.println("# kind\trow\tcol\tvalue")
            for (r in 0 until m.rows) for (c in 0 until m.cols) out.println("M\t$r\t$c\t${m[r, c].full()}")
            for (r in 0 until m2.rows) for (c in 0 until m2.cols) out.println("M2\t$r\t$c\t${m2[r, c].full()}")
            for (i in g.indices) out.println("g\t$i\t0\t${g[i].full()}")
            for (i in d.indices) out.println("d\t$i\t0\t${d[i].full()}")
            for (i in coeffs.indices) out.println("c_base\t$i\t0\t${coeffs[i].full()}")
        }
        return file
    }

    /**
     * L4/L5. The images of the integral operators and the right-hand sides of the model problems.
     *
     * The baseline in SciPy is `scipy.integrate.quad` (the adaptive QUADPACK), independent of
     * the composite quadrature of the project.
     *
     * Four quantities per problem are dumped: the image (`Ku`/`Vu`), the right-hand side `rhs`
     * and its two derivatives `rhsDeriv`, `rhsDeriv2` — and all four are cross-checked. It is exactly
     * the checks `V/<problem>/rhsDeriv` and `V/<problem>/rhsDeriv2` that constitute the only
     * numerical check of the Leibniz formulas for `(Vu)'` and `(Vu)''`, for which there is no separate
     * publication: the baseline is assembled from independently derived `K_t`, `K_tt` and the FULL
     * derivative of the diagonal (see `volterra_image_deriv` in `tools/verify_with_scipy.py`).
     * Without them the derivatives dumped here would be cross-checked against nothing.
     */
    fun dumpOperatorImages(dir: File): File {
        val grid = Grid.uniform(DUMP_GRID_SIZE, a = DUMP_A, b = DUMP_B)
        val quad = GaussLegendre(DUMP_QUADRATURE_NODES)
        val samplePoints = (0..DUMP_OPERATOR_SAMPLES)
            .map { grid.a + (grid.b - grid.a) * it / DUMP_OPERATOR_SAMPLES.toDouble() }

        val file = File(dir, "operator-images.tsv")
        file.printWriter().use { out ->
            out.println("# equation\tproblem\tquantity\tt\tvalue")
            for (problem in listOf(FredholmProblem.F2, FredholmProblem.F2exp)) {
                val op = FredholmOperator(problem.kernel, grid, quad)
                for (t in samplePoints) {
                    val prefix = "F\t${problem.name}"
                    out.println("$prefix\tKu\t${t.full()}\t${op.apply(t) { s -> problem.exact(s) }.full()}")
                    out.println("$prefix\trhs\t${t.full()}\t${problem.rhsExact(t, op).full()}")
                    out.println("$prefix\trhsDeriv\t${t.full()}\t${problem.rhsExactDeriv(t, op).full()}")
                    out.println("$prefix\trhsDeriv2\t${t.full()}\t${problem.rhsExactDeriv2(t, op).full()}")
                }
            }
            for (problem in listOf(
                problems.volterra.VolterraProblem.V2,
                problems.volterra.VolterraProblem.V2exp,
                problems.volterra.VolterraProblem.V2win,
            )) {
                val op = solvers.volterra.VolterraOperator(problem.kernel, grid, quad)
                for (t in samplePoints) {
                    val prefix = "V\t${problem.name}"
                    out.println("$prefix\tVu\t${t.full()}\t${op.apply(t) { s -> problem.exact(s) }.full()}")
                    out.println("$prefix\trhs\t${t.full()}\t${problem.rhsExact(t, op).full()}")
                    out.println("$prefix\trhsDeriv\t${t.full()}\t${problem.rhsExactDeriv(t, op).full()}")
                    out.println("$prefix\trhsDeriv2\t${t.full()}\t${problem.rhsExactDeriv2(t, op).full()}")
                }
            }
        }
        return file
    }

    /**
     * L6. The resulting errors `E_h` of the base scheme and of the Sloan iteration — for a cross-check
     * against a solution built by an independent Nyström method with NumPy.
     */
    fun dumpSolutionErrors(dir: File): File {
        val file = File(dir, "solution-errors.tsv")
        file.printWriter().use { out ->
            out.println("# problem\tsystem\tn\tscheme\tEh")
            for (problem in listOf(FredholmProblem.F2, FredholmProblem.F2exp)) {
                for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                    for (n in listOf(8, 16, 32)) {
                        val grid = Grid.uniform(n, a = DUMP_A, b = DUMP_B)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = ProjFunctionals(basis)
                        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(DUMP_QUADRATURE_NODES))
                        val solver = FredholmSecondKindSolver(
                            basis, funcs, op, 1.0,
                            RhsWithDerivatives(
                                { t -> problem.rhsExact(t, op) },
                                { t -> problem.rhsExactDeriv(t, op) },
                                { t -> problem.rhsExactDeriv2(t, op) },
                            ),
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "${problem.name}\t${system.name}\t$n"
                        out.println("$prefix\tbase\t${errorEh(exact, solver.base().eval, grid).full()}")
                        out.println("$prefix\tsloan\t${errorEh(exact, solver.sloan().eval, grid).full()}")
                    }
                }
            }
        }
        return file
    }
}
