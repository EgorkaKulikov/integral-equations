package healthchecks

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import solvers.uryson.CollocationCore
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Health checks SPECIFIC to the solver of the nonlinear Uryson equation.
 *
 * The common checks of the numerical core (splines, functionals, quadrature) are moved
 * into [SplineCoreHealthCheckTest] and are not duplicated here.
 *
 * Previously these checks were arranged as an object `HealthChecks` in the production
 * code and returned a numerical "measured quantity" compared with a threshold. Some
 * of them were in fact disguised BINARY checks: when the condition failed,
 * the quantity was artificially set to one in order to exceed the threshold.
 * Here such checks are written as explicit assertions.
 */
@Tag("fast")
class UrysonHealthCheckTest {

    private companion object {
        /** The threshold for identities that hold exactly (up to rounding). */
        const val EXACT_IDENTITY_TOLERANCE = 1e-10

        /** The threshold for the sum of the weights: the quantity is computed without integration, the error is minimal. */
        const val WEIGHTS_SUM_TOLERANCE = 1e-12

        /** The upper bound of the error of the base scheme on a coarse grid (a protection against divergence). */
        const val MAX_COARSE_GRID_ERROR = 1e-2

        /** The admissible excess of the residual under grid refinement (the residual is not obliged to fall strictly monotonically). */
        const val RESIDUAL_GROWTH_TOLERANCE = 1.1

        /** The regularization parameter for the checks where the very fact of positive definiteness matters. */
        const val PROBE_ALPHA = 1e-3
    }

    private val quad = GaussLegendre(8)

    /**
     * The stabilizer matrix `R_h` is symmetric, positive definite and banded.
     *
     * These three properties ensure the solvability of the Gauss-Newton system: without
     * positive definiteness the regularized problem stops being convex.
     * The bandedness (`|i-j| <= 2`) follows from the supports of distant splines not
     * intersecting, and its violation would mean an error in computing the common support.
     */
    @Test
    fun gramMatrixIsSymmetricPositiveDefiniteAndBanded() {
        for (grid in listOf(Grid.uniform(8), Grid.quasiUniform(8))) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(system, grid)
                val space = SplineSpace(basis, quad)
                val gram = space.gramR

                val asymmetry = LinearAlgebra.maxAsymmetry(gram)
                assertTrue(
                    asymmetry < EXACT_IDENTITY_TOLERANCE,
                    "Basis ${system.name}: the matrix R_h must be symmetric, " +
                        "max|R - R^T| = $asymmetry",
                )
                assertNotNull(
                    LinearAlgebra.cholesky(gram),
                    "Basis ${system.name}: the matrix R_h must be positive definite " +
                        "(the Cholesky decomposition does not exist)",
                )
                for (i in 0 until space.dim) {
                    for (j in 0 until space.dim) {
                        if (abs(i - j) > 2) {
                            assertTrue(
                                abs(gram[i, j]) < EXACT_IDENTITY_TOLERANCE,
                                "Basis ${system.name}: the element R_h[$i][$j] outside the band must be " +
                                    "zero, got ${gram[i, j]}",
                            )
                        }
                    }
                }
            }
        }
    }

    /**
     * The sum of the weights of the discrete norm equals the length of the interval.
     *
     * The weights `w_j = (x_{j+3} - x_j)/3` sum to `b - a`, since every interval
     * enters the supports of exactly three splines. A violation would mean that the discrete
     * norm is inconsistent with `L^2`, and the residual estimate in Morozov's principle would be
     * systematically biased.
     */
    @Test
    fun weightsSumEqualsIntervalLength() {
        for (grid in listOf(Grid.uniform(8), Grid.quasiUniform(8))) {
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val space = SplineSpace(basis, quad)
            val deviation = abs(space.weightsSum() - (grid.b - grid.a))
            assertTrue(
                deviation < WEIGHTS_SUM_TOLERANCE,
                "The sum of the weights must equal the length of the interval ${grid.b - grid.a}, " +
                    "the deviation is $deviation",
            )
        }
    }

    /**
     * The matrix of the Gauss-Newton system `B^T W_h B + alpha R_h` is symmetric and positive
     * definite for any `alpha > 0`.
     *
     * The term `B^T W_h B` is non-negative definite, but may be singular
     * (a first-kind problem is ill posed); it is exactly the addition of `alpha R_h` that makes the system
     * solvable. The check confirms that the regularization really plays
     * its role.
     */
    @Test
    fun gaussNewtonMatrixIsPositiveDefinite() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.C.exact(t) })
        val b = core.bMatrix(coefficients)
        val normalMatrix = LinearAlgebra.atWa(b, space.weights)
        val regularized = LinearAlgebra.addScaled(normalMatrix, space.gramR, PROBE_ALPHA)

        val asymmetry = LinearAlgebra.maxAsymmetry(regularized)
        assertTrue(
            asymmetry < EXACT_IDENTITY_TOLERANCE,
            "The matrix B^T W B + alpha R must be symmetric, max|A - A^T| = $asymmetry",
        )
        assertNotNull(
            LinearAlgebra.cholesky(regularized),
            "The matrix B^T W B + alpha R at alpha = $PROBE_ALPHA must be positive definite",
        )
    }

    /**
     * The consistency of the right-hand side with the operator: substituting the exact solution into
     * the discrete residual gives a small quantity that decreases under grid refinement.
     *
     * The check catches an inconsistency between the way the right-hand side
     * `f = U x*` is built and the way it is discretized by the functionals `theta_j`.
     */
    @Test
    fun rightHandSideIsConsistentWithOperator() {
        fun residualOn(n: Int): Double {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
            val core = CollocationCore(basis, funcs, op)
            val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.C.exact(t) })
            val xi = core.xiVector(coefficients)
            val thetaF = firstKindSolver(basis, funcs, space, op)
                .thetaOf { t -> UrysonProblem.C.rhsExact(t, op) }
            // The getter returns a COPY of the array — we take it once BEFORE the loop, not per iteration.
            val weights = space.weights
            var sum = 0.0
            for (j in 0 until grid.n + 2) {
                val d = xi[j] - thetaF[j]
                sum += weights[j] * d * d
            }
            return Math.sqrt(sum)
        }

        val coarse = residualOn(8)
        val fine = residualOn(16)
        assertTrue(
            fine <= coarse * RESIDUAL_GROWTH_TOLERANCE,
            "The residual at the exact solution must decrease under grid refinement: " +
                "res(n=8) = $coarse, res(n=16) = $fine",
        )
        assertTrue(
            fine.isFinite() && fine < MAX_COARSE_GRID_ERROR,
            "The residual at the exact solution must be small, got $fine",
        )
    }

    /**
     * The base second-kind scheme converges on the model problem A.
     *
     * Previously this check returned a "measured quantity" equal to the error when the
     * condition held and to one when it did not — that is, it was binary under
     * the guise of a numerical one. Here the condition is written out directly.
     */
    @Test
    fun secondKindSchemeConverges() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val solver = secondKindSolver(UrysonProblem.A, basis, funcs, space, op)
        val error = errorEh({ t -> UrysonProblem.A.exact(t) }, solver.base().eval, grid)
        assertTrue(
            error.isFinite() && error < MAX_COARSE_GRID_ERROR,
            "The base scheme on problem A must give a small error, got E_h = $error",
        )
    }

    /**
     * The Gauss-Newton iteration decreases the regularized Tikhonov functional.
     *
     * This is a basic property of the correctness of the descent: if the functional grows, then
     * the system for the step is assembled with a wrong sign or the matrix does not correspond
     * to the gradient.
     */
    @Test
    fun gaussNewtonStepDecreasesTikhonovFunctional() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta =
            noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 1e-2, 999L)
        val core = CollocationCore(basis, funcs, op)

        // The getter returns a COPY of the array — we take it once, not on every iteration of the loop.
        val weights = space.weights

        fun tikhonovFunctional(c: DoubleArray): Double {
            val xi = core.xiVector(c)
            var sum = 0.0
            for (j in 0 until grid.n + 2) {
                val d = xi[j] - thetaFDelta[j]
                sum += weights[j] * d * d
            }
            return sum + PROBE_ALPHA * space.omegaReg(c)
        }

        val start = DoubleArray(grid.n + 2)
        val valueBefore = tikhonovFunctional(start)
        val afterStep = solver.solveFixedAlpha(thetaFDelta, PROBE_ALPHA, start)
        val valueAfter = tikhonovFunctional(afterStep)
        assertTrue(
            valueAfter <= valueBefore,
            "The Gauss-Newton iterations must decrease the Tikhonov functional: " +
                "it was $valueBefore, it became $valueAfter",
        )
    }
}
