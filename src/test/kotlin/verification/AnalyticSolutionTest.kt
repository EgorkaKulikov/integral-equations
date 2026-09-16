package verification

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.analytic.AnalyticFredholmProblem
import problems.analytic.AnalyticVolterraProblem
import problems.analytic.analyticFredholmSolver
import problems.analytic.analyticVolterraSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * INDEPENDENT VERIFICATION against analytically exact solutions (the assignment, item 2.1).
 *
 * All the other checks of the project are closed on its own implementation: the characterization
 * test records that the result HAS NOT CHANGED, but does not prove that it is CORRECT.
 * Here the baseline is obtained outside the code — by solving finite systems and taking integrals
 * BY HAND (see the derivations in the KDoc of the problems [AnalyticFredholmProblem],
 * [AnalyticVolterraProblem]).
 *
 * The tests are split into two levels, and this split is essential.
 *
 *  1. THE IDENTITY `u* - K u* = f` ([fredholmAnalyticIdentityHolds],
 *     [volterraAnalyticIdentityHolds]). It checks THE DERIVATIONS THEMSELVES and not the solver:
 *     if an error was made in the manual derivation, it will be detected here instead of silently
 *     becoming the "baseline". A high-precision quadrature of [PRECISE_QUADRATURE_ORDER]
 *     nodes per subinterval with a fine partition is used — it does not take part in the work of the schemes and
 *     serves only as an independent cross-check.
 *
 *  2. THE CONVERGENCE of the schemes to the analytic solution. It checks the solver itself against
 *     the baseline, trust in which is ensured by level 1.
 *
 * The key difference from `problems.fredholm.FredholmProblem`: there the right-hand side is built
 * numerically by the same operator that is then checked, because of which the quadrature error
 * enters both sides of the comparison and partly cancels. Here `f` is written out
 * analytically, so there is no such mutual compensation.
 */
@Tag("slow")
class AnalyticSolutionTest {

    private companion object {
        /**
         * The quadrature order for the INDEPENDENT check of the identity. It is twice the
         * working one (8) and is applied on a fine partition: the goal is that the error
         * of the check itself be definitely below the tolerance being checked.
         */
        const val PRECISE_QUADRATURE_ORDER = 16

        /** The number of subintervals of the composite partition when checking the identity. */
        const val PRECISE_SUBDIVISIONS = 32

        /**
         * The tolerance of the check of the identity `u* - K u* = f`. The value reflects the limit
         * of accuracy of a high-order composite quadrature on smooth data.
         */
        const val IDENTITY_TOLERANCE = 1e-12

        /** The number of points at which the identity is checked. */
        const val IDENTITY_SAMPLE_COUNT = 21

        /** The central difference step when cross-checking the analytic derivatives `f'`. */
        const val DERIVATIVE_STEP = 1e-5

        /** The tolerance of the cross-check of `f'` with a central difference (the difference error is ~ h^2). */
        const val FIRST_DERIVATIVE_TOLERANCE = 1e-6

        /** The central difference step when cross-checking `f''` (the second order requires a larger step). */
        const val SECOND_DERIVATIVE_STEP = 1e-4

        /** The tolerance of the cross-check of `f''` with a second central difference. */
        const val SECOND_DERIVATIVE_TOLERANCE = 1e-4

        /**
         * The tolerance for the problem whose solution lies in `span{1, t, t^2}`: the method must
         * reproduce it practically exactly.
         */
        const val SPAN_EXACTNESS_TOLERANCE = 1e-10

        /**
         * The upper bound of the error on the finest grid of the set. The value is chosen
         * with a large margin relative to the observed quantities (of order 1e-6 and below):
         * the test must react to a breakage of the scheme and not to fluctuations of a constant.
         */
        const val MAX_FINE_GRID_ERROR = 1e-3

        /** The grids for checking the convergence of the schemes to the analytic solution. */
        val GRID_SIZES = listOf(8, 16, 32)
    }

    /** The quadrature for the independent check of the identity (NOT used in the schemes). */
    private val preciseQuadrature = GaussLegendre(PRECISE_QUADRATURE_ORDER)

    /** A high-precision integral over `[lo, hi]` by a composite quadrature. */
    private fun preciseIntegral(lo: Double, hi: Double, integrand: (Double) -> Double): Double {
        if (hi <= lo) return 0.0
        val breakpoints = DoubleArray(PRECISE_SUBDIVISIONS + 1) {
            lo + (hi - lo) * it / PRECISE_SUBDIVISIONS
        }
        return preciseQuadrature.integrate(breakpoints, integrand)
    }

    /** A uniform sample of points of the interval `[0, 1]`. */
    private fun samplePoints(): List<Double> =
        (0 until IDENTITY_SAMPLE_COUNT).map { it.toDouble() / (IDENTITY_SAMPLE_COUNT - 1) }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        else -> error("Unknown functional family: '$name'")
    }

    private fun reportIfAny(failures: List<String>) {
        assertTrue(
            failures.isEmpty(),
            "${failures.size} discrepancies with the analytic baseline were found:\n" +
                failures.joinToString("\n").take(6000),
        )
    }

    // ------------------------------------------------------------------------
    // Level 1: checking THE DERIVATIONS THEMSELVES (not the solver)
    // ------------------------------------------------------------------------

    /**
     * The identity `u*(t) - \\int_0^1 K(t,s) u*(s) ds = f(t)` for all the analytic
     * Fredholm problems.
     *
     * This is an insurance against an error in the manual derivation: both `u*` and `f` are written out on paper,
     * and if at least one of the derivations is wrong, the identity will break. The check does not
     * address the basis, the functionals or the solver — only the kernel and
     * an independent high-precision quadrature.
     */
    @Test
    fun fredholmAnalyticIdentityHolds() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticFredholmProblem.ALL) {
            var worstDeviation = 0.0
            var worstPoint = Double.NaN
            for (t in samplePoints()) {
                val image = preciseIntegral(0.0, 1.0) { s -> problem.kernel.k(t, s) * problem.exact(s) }
                val residual = problem.exact(t) - image - problem.rhs(t)
                if (abs(residual) > worstDeviation) {
                    worstDeviation = abs(residual)
                    worstPoint = t
                }
            }
            if (worstDeviation > IDENTITY_TOLERANCE) {
                failures += "${problem.name}: |u* - K u* - f| = $worstDeviation at t=$worstPoint " +
                    "(tolerance $IDENTITY_TOLERANCE). Derivation: ${problem.derivation}"
            }
        }
        reportIfAny(failures)
    }

    /**
     * The identity `u*(t) - \\int_0^t K(t,s) u*(s) ds = f(t)` for all the analytic
     * Volterra problems.
     *
     * For the problems with a convolution kernel this is an independent check of the solution obtained
     * by the Laplace transform: the image is computed by direct integration rather than through
     * operational calculus.
     */
    @Test
    fun volterraAnalyticIdentityHolds() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticVolterraProblem.ALL) {
            var worstDeviation = 0.0
            var worstPoint = Double.NaN
            for (t in samplePoints()) {
                val image = preciseIntegral(0.0, t) { s -> problem.kernel.k(t, s) * problem.exact(s) }
                val residual = problem.exact(t) - image - problem.rhs(t)
                if (abs(residual) > worstDeviation) {
                    worstDeviation = abs(residual)
                    worstPoint = t
                }
            }
            if (worstDeviation > IDENTITY_TOLERANCE) {
                failures += "${problem.name}: |u* - V u* - f| = $worstDeviation at t=$worstPoint " +
                    "(tolerance $IDENTITY_TOLERANCE). Derivation: ${problem.derivation}"
            }
        }
        reportIfAny(failures)
    }

    /**
     * The consistency of the analytic derivatives of the right-hand side `f'` and `f''` with `f` itself.
     *
     * The derivatives are written out by hand and are used by the de Boor–Fix functional
     * families. An error in them would not show up in the families `theta`/`mu`
     * (they do not read derivatives) and would stay unnoticed. The cross-check is performed
     * by finite differences — independently of any part of the project.
     */
    @Test
    fun analyticRightHandSideDerivativesAreConsistent() {
        val failures = mutableListOf<String>()

        fun check(
            name: String,
            rhs: (Double) -> Double,
            rhsDeriv: (Double) -> Double,
            rhsDeriv2: (Double) -> Double,
        ) {
            for (i in 1 until IDENTITY_SAMPLE_COUNT - 1) {
                val t = i.toDouble() / (IDENTITY_SAMPLE_COUNT - 1)
                val numericFirst =
                    (rhs(t + DERIVATIVE_STEP) - rhs(t - DERIVATIVE_STEP)) / (2 * DERIVATIVE_STEP)
                if (abs(rhsDeriv(t) - numericFirst) > FIRST_DERIVATIVE_TOLERANCE) {
                    failures += "$name: f'($t)=${rhsDeriv(t)} disagrees with the difference value $numericFirst"
                }
                val h = SECOND_DERIVATIVE_STEP
                val numericSecond = (rhs(t + h) - 2 * rhs(t) + rhs(t - h)) / (h * h)
                if (abs(rhsDeriv2(t) - numericSecond) > SECOND_DERIVATIVE_TOLERANCE) {
                    failures += "$name: f''($t)=${rhsDeriv2(t)} disagrees with the difference value $numericSecond"
                }
            }
        }

        for (problem in AnalyticFredholmProblem.ALL) {
            check(problem.name, problem.rhs, problem.rhsDeriv, problem.rhsDeriv2)
        }
        for (problem in AnalyticVolterraProblem.ALL) {
            check(problem.name, problem.rhs, problem.rhsDeriv, problem.rhsDeriv2)
        }
        reportIfAny(failures)
    }

    // ------------------------------------------------------------------------
    // Level 2: the convergence of the schemes to the analytic solution
    // ------------------------------------------------------------------------

    /**
     * The problem from the statement of the assignment (`K = t*s`, `f = t`, the exact solution `u* = (3/2)t`)
     * is solved EXACTLY: a linear function lies in `span{1, t, t^2}` of the polynomial
     * generating system.
     *
     * The check covers the whole chain — the basis, the functionals, the quadrature, the assembly
     * of the matrix and the linear solve — against a result obtained by hand on paper.
     */
    @Test
    fun separableRank1ExampleIsReproducedExactly() {
        val problem = AnalyticFredholmProblem.SEPARABLE_RANK1_LINEAR
        val failures = mutableListOf<String>()
        for (familyName in listOf("theta", "mu")) {
            for (n in listOf(8, 16)) {
                val grid = Grid.uniform(n)
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                val funcs = family(familyName, basis)
                val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                val solver = analyticFredholmSolver(problem, basis, funcs, op)
                val error = errorEh(problem.exact, solver.base().eval, grid)
                if (!(error < SPAN_EXACTNESS_TOLERANCE)) {
                    failures += "${problem.name}/$familyName/n=$n: E_h=$error must be " +
                        "below $SPAN_EXACTNESS_TOLERANCE (u*=(3/2)t lies in the span of the generating system B)"
                }
            }
        }
        reportIfAny(failures)
    }

    /**
     * The schemes `base`, `sloan`, `kulkarni` converge to the ANALYTIC solution of the Fredholm
     * problems with a degenerate kernel and of the MMS problem.
     *
     * Two substantial conditions are checked: the error decreases under refinement of the
     * grid and on the finest grid reaches a reasonable absolute magnitude.
     * Measuring the convergence order is the subject of a separate test,
     * [convergence.ConvergenceOrderTest] (an explicit table of expected orders on the problems
     * `F2`/`V2`); here the convergence to the ANALYTIC solution is checked, not its rate.
     *
     * IMPORTANT about the family `mu`. For quasi-interpolants the Kulkarni scheme is implemented
     * by a SIMPLE ITERATION (`kulkarniQuasi`), which requires `rho(K) < 1`. On the problems
     * [AnalyticFredholmProblem.SEPARABLE_RANK2] (`rho ~ 1.27`) and
     * [AnalyticFredholmProblem.SEPARABLE_RANK3] (`rho ~ 1.41`) it diverges. This is a
     * limitation OF THE METHOD, not a defect of the implementation and not a property of the problems: the problems themselves
     * are uniquely solvable, and the direct schemes solve them regularly. Therefore the Kulkarni scheme
     * for `mu` is checked only on the problems with `rho < 1`, and the divergence on the rest is
     * recorded by a separate test, [quasiKulkarniDivergesWhenSpectralRadiusExceedsOne]
     * — this is a deliberate limitation and not a fitting of the tolerance.
     */
    @Test
    fun fredholmSchemesConvergeToAnalyticSolution() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticFredholmProblem.CONVERGENT) {
            for (familyName in listOf("theta", "mu")) {
                // The projector theta uses the Kulkarni reduction with a DIRECT linear solve
                // and is always applicable; the quasi-interpolant mu uses a simple iteration.
                val iterativeKulkarni = familyName == "mu"
                val includeKulkarni = !iterativeKulkarni || problem.supportsFixedPointSchemes
                val errors = linkedMapOf<String, MutableList<Double>>(
                    "base" to mutableListOf(),
                    "sloan" to mutableListOf(),
                )
                if (includeKulkarni) errors["kulkarni"] = mutableListOf()
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = analyticFredholmSolver(problem, basis, funcs, op)
                    errors.getValue("base") += errorEh(problem.exact, solver.base().eval, grid)
                    errors.getValue("sloan") += errorEh(problem.exact, solver.sloan().eval, grid)
                    if (includeKulkarni) {
                        errors.getValue("kulkarni") += errorEh(problem.exact, solver.kulkarni().eval, grid)
                    }
                }
                collectConvergenceFailures("F.${problem.name}.$familyName", errors, failures)
            }
        }
        reportIfAny(failures)
    }

    /**
     * A DOCUMENTED LIMITATION: the Kulkarni scheme for quasi-interpolants
     * diverges at `rho(K) > 1`, whereas the direct schemes solve the same problems regularly.
     *
     * Found while developing this very test: on [AnalyticFredholmProblem.SEPARABLE_RANK2]
     * and [AnalyticFredholmProblem.SEPARABLE_RANK3] the iteration gives quantities of order 1e21
     * and 1e29 respectively (independent of `n` — a sign of a divergence of the iteration
     * and not of an approximation error, which would decrease).
     *
     * The test fixes three facts: (a) the direct schemes work — hence the problem
     * is well posed and the baseline is right; (b) the iterative one by default SIGNALS
     * the error with an exception (the single convergence contract); (c) in the mode
     * `throwOnDivergence = false` the same scheme returns a result marked
     * `converged = false`, and its error is indeed catastrophic.
     *
     * HISTORY: originally the test checked that `kulkarni()` SILENTLY returns
     * a diverging result with `E_h ~ 1e21`. After the introduction of the single contract
     * such behaviour became inadmissible, and the test was updated deliberately: it is exactly
     * such a failure that task 4 required — a silent return is no longer possible.
     */
    @Test
    fun quasiKulkarniDivergesWhenSpectralRadiusExceedsOne() {
        val failures = mutableListOf<String>()
        val divergent = AnalyticFredholmProblem.ALL.filterNot { it.supportsFixedPointSchemes }
        assertTrue(
            divergent.isNotEmpty(),
            "The set of problems must contain at least one with rho > 1 for this check",
        )
        for (problem in divergent) {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))

            // (a) The direct scheme on the projector theta — the problem is solved regularly.
            val projectorSolver = analyticFredholmSolver(problem, basis, ProjFunctionals(basis), op)
            val directError = errorEh(problem.exact, projectorSolver.base().eval, grid)
            if (!(directError < MAX_FINE_GRID_ERROR)) {
                failures += "${problem.name}: the direct scheme (theta/base) must solve the problem " +
                    "with rho=${problem.spectralRadius}, but E_h=$directError"
            }

            // (b) By default a divergence must be an explicit error.
            val strictSolver = analyticFredholmSolver(problem, basis, AveragingFunctionals(basis), op)
            assertFailsWith<IllegalStateException>(
                "${problem.name}: at rho=${problem.spectralRadius} > 1 the scheme must " +
                    "report the divergence with an exception instead of returning a result",
            ) { strictSolver.kulkarni() }

            // (c) In the explicitly permitted mode the result is available but marked as non-converged.
            val lenientSolver = analyticFredholmSolver(
                problem, basis, AveragingFunctionals(basis), op, throwOnDivergence = false,
            )
            val solution = lenientSolver.kulkarni()
            if (solution.converged) {
                failures += "${problem.name}: the result at rho=${problem.spectralRadius} > 1 " +
                    "must be marked converged = false"
            }
            val iterativeError = errorEh(problem.exact, solution.eval, grid)
            if (iterativeError < 1.0) {
                failures += "${problem.name}: a divergence of kulkarniQuasi was expected at " +
                    "rho=${problem.spectralRadius} > 1, but E_h=$iterativeError was obtained. " +
                    "If the scheme was improved — update the expectation deliberately"
            }
        }
        reportIfAny(failures)
    }

    /**
     * The schemes `base`, `sloan`, `kulkarni` converge to the ANALYTIC solution of the Volterra
     * problems: three with a convolution kernel (solved by the Laplace transform) and one MMS.
     */
    @Test
    fun volterraSchemesConvergeToAnalyticSolution() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticVolterraProblem.ALL) {
            for (familyName in listOf("theta", "mu")) {
                val errors = linkedMapOf<String, MutableList<Double>>(
                    "base" to mutableListOf(),
                    "sloan" to mutableListOf(),
                    "kulkarni" to mutableListOf(),
                )
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = analyticVolterraSolver(problem, basis, funcs, op)
                    errors.getValue("base") += errorEh(problem.exact, solver.base().eval, grid)
                    errors.getValue("sloan") += errorEh(problem.exact, solver.sloan().eval, grid)
                    errors.getValue("kulkarni") += errorEh(problem.exact, solver.kulkarni().eval, grid)
                }
                collectConvergenceFailures("V.${problem.name}.$familyName", errors, failures)
            }
        }
        reportIfAny(failures)
    }

    /**
     * A common check of a set of errors: a monotone decrease and reaching a reasonable
     * absolute accuracy on the finest grid.
     *
     * Values below [SPAN_EXACTNESS_TOLERANCE] are exempt from the decrease check:
     * there the rounding noise dominates and requiring monotonicity is pointless.
     */
    private fun collectConvergenceFailures(
        tag: String,
        errors: Map<String, List<Double>>,
        failures: MutableList<String>,
    ) {
        for ((scheme, values) in errors) {
            val formatted = values.joinToString(", ") { "%.3e".format(it) }
            val detail = "$tag.$scheme: E_h(n=${GRID_SIZES.joinToString(",")}) = [$formatted]"
            if (values.any { !it.isFinite() }) {
                failures += "$detail — a non-numeric value is present"
                continue
            }
            val fine = values.last()
            if (fine > MAX_FINE_GRID_ERROR) {
                failures += "$detail — the error on the fine grid exceeds $MAX_FINE_GRID_ERROR"
            }
            // We check the decrease only until the rounding level is reached.
            for (i in 0 until values.size - 1) {
                if (values[i] < SPAN_EXACTNESS_TOLERANCE) break
                if (values[i + 1] >= values[i]) {
                    failures += "$detail — the error does not decrease at the step ${GRID_SIZES[i]}->" +
                        "${GRID_SIZES[i + 1]}"
                    break
                }
            }
        }
    }
}
