package regression

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraSecondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * REGRESSION TESTS for detected and fixed defects.
 *
 * Every test is named after the defect and documented: what the error was, why it
 * did not show up in the existing tests and what exactly is checked now. All these
 * tests FAIL on the code before the fix.
 *
 * Defects 4 and 5 belong to the linear algebra layer and are checked in the
 * `numerical-core` library (`LinearAlgebraRegressionTest`); the numbering is kept here.
 */
@Tag("fast")
class DefectRegressionTest {

    /**
     * DEFECT 1. The model problems F2span/V2span/F1 declared the kernel derivatives
     * (K_s, K_tt) and the second derivative of the solution to be zero, although the true values
     * are non-zero. The functional family xi^<0> (de Boor--Fix, r=0) reads second
     * derivatives — and silently got a wrong right-hand side.
     *
     * Manifestation: for the problem F2span the exact solution u*(t)=t^2 belongs to
     * span{1,t,t^2} = the generating system B, so the method MUST reproduce
     * it to machine accuracy. Before the fix the error was ~1e-2, that is
     * twelve orders of magnitude larger than it should be.
     *
     * The defect was not caught before: the tests for xi^<0> checked only finiteness
     * of the result and its biorthogonality, but not the accuracy on the span problem.
     */
    @Test
    fun defect1_xi0OnSpanProblemIsExactForFredholm() {
        val problem = problems.fredholm.FredholmProblem.F2span
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = DeBoorFixFunctionals(basis, 0)
            val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
            val solver = FredholmSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives(
                    { t -> problem.rhsExact(t, op) },
                    { t -> problem.rhsExactDeriv(t, op) },
                    { t -> problem.rhsExactDeriv2(t, op) },
                ),
            )
            val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
            assertTrue(
                error < 1e-10,
                "F2span (u*=t^2 in the span of the generating system B) must be solved exactly " +
                    "by the family xi^<0>, but E_h(n=$n)=$error (before the fix it was ~1e-2)",
            )
        }
    }

    /** The same defect for the Volterra solver (problem V2span). */
    @Test
    fun defect1_xi0OnSpanProblemIsExactForVolterra() {
        val problem = problems.volterra.VolterraProblem.V2span
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = DeBoorFixFunctionals(basis, 0)
            val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
            val solver = VolterraSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives(
                    { t -> problem.rhsExact(t, op) },
                    { t -> problem.rhsExactDeriv(t, op) },
                    { t -> problem.rhsExactDeriv2(t, op) },
                ),
            )
            val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
            assertTrue(
                error < 1e-10,
                "V2span (u*=t^2 in the span of the generating system B) must be solved exactly " +
                    "by the family xi^<0>, but E_h(n=$n)=$error",
            )
        }
    }

    /**
     * DEFECT 1 (checking the kernel coefficients themselves). A direct check that the
     * kernel derivatives are defined consistently with the kernel itself: comparison of the
     * analytic K_s and K_tt with a central finite difference.
     *
     * For the Fredholm kernels only `K_tt` is checked: the derivative in `s` is not used in the
     * Fredholm schemes (the integration limits are constant) and was removed from
     * [solvers.fredholm.KernelF]. In the Volterra schemes `K_s` is needed for the boundary term
     * of the Leibniz formula, so both derivatives are checked there.
     */
    @Test
    fun defect1_kernelDerivativesAreConsistentWithKernel() {
        val step = 1e-5
        val samplePoints = listOf(0.1 to 0.2, 0.5 to 0.3, 0.8 to 0.75)

        val fredholmKernels = listOf(
            "F2span" to problems.fredholm.FredholmProblem.F2span.kernel,
            "F1" to problems.fredholm.FredholmProblem.F1.kernel,
            "F2" to problems.fredholm.FredholmProblem.F2.kernel,
            "F2exp" to problems.fredholm.FredholmProblem.F2exp.kernel,
        )
        for ((name, kernel) in fredholmKernels) {
            for ((t, s) in samplePoints) {
                val numericKtt = (kernel.kT(t + step, s) - kernel.kT(t - step, s)) / (2 * step)
                assertTrue(
                    abs(kernel.kTT(t, s) - numericKtt) < 1e-6,
                    "$name: K_tt($t,$s)=${kernel.kTT(t, s)} disagrees with the difference value ${numericKtt}",
                )
            }
        }

        val volterraKernels = listOf(
            "V2span" to problems.volterra.VolterraProblem.V2span.kernel,
            "V2" to problems.volterra.VolterraProblem.V2.kernel,
            "V2exp" to problems.volterra.VolterraProblem.V2exp.kernel,
        )
        for ((name, kernel) in volterraKernels) {
            for ((t, s) in samplePoints) {
                val numericKs = (kernel.k(t, s + step) - kernel.k(t, s - step)) / (2 * step)
                assertTrue(
                    abs(kernel.kS(t, s) - numericKs) < 1e-6,
                    "$name: K_s($t,$s)=${kernel.kS(t, s)} disagrees with the difference value ${numericKs}",
                )
                val numericKtt = (kernel.kT(t + step, s) - kernel.kT(t - step, s)) / (2 * step)
                assertTrue(
                    abs(kernel.kTT(t, s) - numericKtt) < 1e-6,
                    "$name: K_tt($t,$s)=${kernel.kTT(t, s)} disagrees with the difference value ${numericKtt}",
                )
            }
        }
    }

    /**
     * DEFECT 2. `FredholmFirstKindSolver` (Fredholm) did not pass the second derivative
     * of the right-hand side to the inner second-kind solver, because of which the family xi^<0>
     * silently got f'' = 0.
     *
     * It is checked that the first-kind solver with xi^<0> now gives a finite and meaningful
     * result, consistent with the solution through the family theta (which uses no second
     * derivatives and was therefore not affected by the defect).
     */
    @Test
    fun defect2_firstKindSolverPassesSecondDerivative() {
        val problem = problems.fredholm.FredholmProblem.F1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))

        val viaXi0 = problems.fredholm.firstKindSolver(problem, basis, DeBoorFixFunctionals(basis, 0), op)
        val errorXi0 = errorEh({ t -> problem.exact(t) }, viaXi0.base().eval, grid)
        assertTrue(errorXi0.isFinite(), "The first-kind solution via xi^<0> must be finite, got $errorXi0")

        val viaTheta = problems.fredholm.firstKindSolver(problem, basis, ProjFunctionals(basis), op)
        val errorTheta = errorEh({ t -> problem.exact(t) }, viaTheta.base().eval, grid)
        assertTrue(
            errorXi0 < 100.0 * maxOf(errorTheta, 1e-12),
            "The xi^<0> error ($errorXi0) must not catastrophically exceed theta ($errorTheta)",
        )
    }

    /**
     * DEFECT 2 (Volterra). After the I->II kind reduction the second derivative of the
     * reduced kernel is analytically unavailable, so the family xi^<0>
     * must be EXPLICITLY rejected on this path instead of silently giving a wrong result.
     */
    @Test
    fun defect2_volterraFirstKindRejectsSecondDerivativeFamilies() {
        val problem = problems.volterra.VolterraProblem.V1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
        assertFailsWith<IllegalArgumentException>(
            "The first-kind Volterra solver must reject families that require a second derivative",
        ) {
            problems.volterra.firstKindSolver(problem, basis, DeBoorFixFunctionals(basis, 0), op)
        }
    }

    /**
     * DEFECT 3. `FredholmFirstKindSolver` (Fredholm) did not check positivity of the
     * regularization parameter: at alpha = 0 the factor c_L = -1/alpha became infinite,
     * at alpha < 0 the meaning of the regularization changed — in both cases without diagnostics.
     */
    @Test
    fun defect3_firstKindSolverRejectsNonPositiveAlpha() {
        val problem = problems.fredholm.FredholmProblem.F1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        for (badAlpha in listOf(0.0, -1e-10)) {
            assertFailsWith<IllegalArgumentException>("alpha=$badAlpha must be rejected") {
                problems.fredholm.firstKindSolver(problem, basis, funcs, op, badAlpha)
            }
        }
    }



    /**
     * DEFECT 6. The Nyström scheme for the Uryson equation started from the EXACT solution
     * of the problem (`problem.exact`), unavailable in real use: the method was
     * irreproducible, and a possible divergence from a neutral approximation was
     * masked.
     *
     * It is checked that the method converges from a neutral initial approximation
     * (the projection of a constant function) and attains an accuracy comparable to the base
     * scheme. The test does not depend on `problem.exact` as a starting point — only as
     * a baseline for measuring the error.
     */
    @Test
    fun defect6_urysonNystromConvergesFromNeutralStart() {
        for (problem in listOf(problems.uryson.UrysonProblem.A, problems.uryson.UrysonProblem.B)) {
            for (n in listOf(8, 16)) {
                val grid = Grid.uniform(n)
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                val funcs = ProjFunctionals(basis)
                val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                val solver = problems.uryson.secondKindSolver(problem, basis, funcs, space, op)

                val nystromSolution = solver.nystrom()
                val exact = { t: Double -> problem.exact(t) }
                val nystromError = errorEh(exact, nystromSolution.eval, grid)
                val baseError = errorEh(exact, solver.base().eval, grid)

                assertTrue(
                    nystromError.isFinite() && nystromError < 1.0,
                    "Problem ${problem.name}, n=$n: Nyström must converge from a neutral " +
                        "initial approximation, got E_h=$nystromError",
                )
                assertTrue(
                    nystromError < 100.0 * maxOf(baseError, 1e-14),
                    "Problem ${problem.name}, n=$n: the accuracy of Nyström ($nystromError) must not " +
                        "fall catastrophically behind the base scheme ($baseError)",
                )
                assertTrue(
                    nystromSolution.iterations >= 1,
                    "Problem ${problem.name}, n=$n: the iteration counter must be meaningful",
                )
            }
        }
    }

    /**
     * DEFECT 7. The first-kind Volterra solver checked `K(t,t) != 0` only at the grid
     * breakpoints and interval midpoints, although the division by the diagonal is performed on a far
     * wider set of points: at all Gauss nodes of the composite quadrature,
     * at the finite-difference stencil points `t ± k*h` and at ANY point requested
     * from the ready solution.
     *
     * Manifestation BEFORE the fix (recorded by an actual run): for a kernel with the
     * diagonal `K(t,t) = (t - t0)^2`, where `t0 = 0.37625` is a point between a breakpoint and a midpoint,
     * the constructor did NOT reject the kernel (the minimum of `|K|` over the checked points was
     * `1.56e-6`, which is above the threshold `1e-12`), `base()` returned a plausible
     * `E_h = 1.2533e-5`, while `sloan()` SILENTLY returned `E_h = NaN`. The user got `NaN`
     * without any diagnostics.
     *
     * Now both lines of defence must fire: either the preliminary check
     * at construction, or the guard at the division point itself — but a silent `NaN` must not occur.
     */
    @Test
    fun defect7_volterraFirstKindRejectsZeroDiagonalBetweenGridPoints() {
        val n = 8
        val grid = Grid.uniform(n)
        // t0 lies BETWEEN a grid breakpoint and an interval midpoint: the breakpoints are multiples of 0.125,
        // the midpoints of 0.0625; 0.37625 coincides with none of these points.
        val t0 = 0.37625
        val kernel = solvers.volterra.KernelV(
            k = { t, _ -> (t - t0) * (t - t0) },
            kT = { t, _ -> 2.0 * (t - t0) },
            kS = { _, _ -> 0.0 },
            kTT = { _, _ -> 2.0 },
        )

        // Precondition of the test: at the breakpoints and midpoints the diagonal is DEFINITELY above the
        // threshold, that is, the old check let this kernel through.
        var worstAtCheckedPoints = Double.MAX_VALUE
        for (i in 0..n) {
            worstAtCheckedPoints = minOf(worstAtCheckedPoints, abs(kernel.k(grid.x(i), grid.x(i))))
            if (i < n) {
                val middle = 0.5 * (grid.x(i) + grid.x(i + 1))
                worstAtCheckedPoints = minOf(worstAtCheckedPoints, abs(kernel.k(middle, middle)))
            }
        }
        assertTrue(
            worstAtCheckedPoints > 1e-12,
            "Precondition of the test: at the breakpoints and midpoints the diagonal must be above the threshold, " +
                "otherwise the old check would have caught the defect too; got $worstAtCheckedPoints",
        )

        val problem = problems.volterra.VolterraProblem(
            name = "V1zeroDiagonal",
            kernel = kernel,
            exact = { t -> Math.cos(t) },
            exactDeriv = { t -> -Math.sin(t) },
            secondKind = false,
            exactDeriv2 = { t -> -Math.cos(t) },
        )
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = solvers.volterra.VolterraOperator(kernel, grid, GaussLegendre(8))

        // The degeneracy must be detected: either at solver construction, or at the division
        // point itself during the run. A silent NaN/Inf is unacceptable.
        val failure = kotlin.runCatching {
            val solver = problems.volterra.firstKindSolver(problem, basis, funcs, op)
            val baseError = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
            val sloanError = errorEh({ t -> problem.exact(t) }, solver.sloan().eval, grid)
            baseError to sloanError
        }.exceptionOrNull()

        assertTrue(
            failure != null,
            "A kernel with a zero of the diagonal at t=$t0 (between a breakpoint and a midpoint) must " +
                "raise an exception instead of silently returning NaN (before the fix sloan() gave NaN)",
        )
        assertTrue(
            failure is IllegalStateException || failure is IllegalArgumentException,
            "Expected an exception about the degenerate diagonal, got ${failure!!::class.simpleName}: " +
                failure.message,
        )
        assertTrue(
            failure.message?.contains("K(t,t)") == true,
            "The error message must name the point and the value of K(t,t), got: " +
                failure.message,
        )
    }

    /**
     * DEFECT 7 (second part). The finite-difference step is ABSOLUTE (1e-3), so on a
     * short interval the stencil `t ± 4h` leaves `[a,b]`, where the kernel and the operator
     * are extended by zero — the derivative would be silently distorted.
     *
     * Such intervals are now EXPLICITLY forbidden with a clear message. The alternative — making
     * the step relative — was rejected: it would change the numerical results on ALL
     * existing problems and would require re-shooting the baseline.
     */
    @Test
    fun defect7_volterraFirstKindRejectsTooShortInterval() {
        // b - a = 1e-3 = one difference step: the stencil t ± 4h definitely does not fit.
        val grid = Grid.uniform(8, a = 0.0, b = 1e-3)
        val problem = problems.volterra.VolterraProblem.V1
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))

        val failure = assertFailsWith<IllegalArgumentException>(
            "On an interval of length 1e-3 the finite-difference stencil does not fit, " +
                "the solver must reject it",
        ) {
            problems.volterra.firstKindSolver(problem, basis, funcs, op)
        }
        assertTrue(
            failure.message?.contains("inapplicable on a too short interval") == true,
            "The message must explain the cause (a short interval), got: ${failure.message}",
        )

        // Control: on the standard interval [0,1] the solver is still constructed.
        val normalGrid = Grid.uniform(8)
        val normalBasis = MinimalSplineBasis(GeneratingSystem.B, normalGrid)
        val normalOp = solvers.volterra.VolterraOperator(problem.kernel, normalGrid, GaussLegendre(8))
        problems.volterra.firstKindSolver(problem, normalBasis, ProjFunctionals(normalBasis), normalOp)
    }
}
