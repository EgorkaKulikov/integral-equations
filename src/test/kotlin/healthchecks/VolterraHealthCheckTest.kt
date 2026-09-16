package healthchecks

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.volterra.VolterraProblem
import problems.volterra.secondKindSolver
import solvers.volterra.KernelV
import solvers.volterra.VolterraOperator
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Health checks SPECIFIC to the Volterra equation solver.
 *
 * The common checks of the numerical core (splines, functionals, quadrature) are moved
 * into [SplineCoreHealthCheckTest] and are not duplicated here: previously the sets of checks
 * of Fredholm and Volterra coincided by about 90 %, although nine of the twelve
 * checks did not depend on the type of the integral operator at all.
 *
 * Only the properties concerning the Volterra operator itself, with a variable upper limit,
 * and the schemes built on it are left here.
 */
@Tag("fast")
class VolterraHealthCheckTest {

    private companion object {
        /** The threshold for the schemes exact on the generating space. */
        const val EXACT_ON_SPAN_TOLERANCE = 1e-8

        /**
         * The minimal admissible factor of the decrease of the error when the number of nodes
         * is doubled. The value 4 corresponds to an observed order not below the second
         * (`2^2 = 4`) — certainly weaker than the theoretical order 3, so that the check
         * reacts to a breakage of the scheme and not to fluctuations of a constant.
         */
        const val MIN_ERROR_REDUCTION_FACTOR = 4.0

        /** The upper bound of the absolute error on a coarse grid (a protection against divergence). */
        const val MAX_COARSE_GRID_ERROR = 1e-1
    }

    private val quad = GaussLegendre(8)

    /**
     * The consistency of the right-hand side with the operator: on a problem whose solution lies
     * in the generating space (`u* = t^2` with a polynomial basis), the base
     * scheme must give machine accuracy.
     *
     * The check catches an inconsistency between the way the right-hand side
     * `f = u* - V u*` is built and the way it is discretized in the solver. For the Volterra operator
     * this is especially significant: the integral is taken over the variable interval `[a,t]`, and any
     * error in the composite partition immediately breaks the exact reproduction.
     */
    @Test
    fun rightHandSideIsConsistentWithOperator() {
        val problem = VolterraProblem.V2span
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(problem.kernel, grid, quad)
        val solver = secondKindSolver(problem, basis, funcs, op)
        val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
        assertTrue(
            error < EXACT_ON_SPAN_TOLERANCE,
            "On the problem V2span (u* = t^2 lies in the span of the generating system B) the base scheme " +
                "must be exact, got E_h = $error",
        )
    }

    /**
     * The convergence of the base scheme: when the number of nodes is doubled the error must decrease
     * by at least a factor of [MIN_ERROR_REDUCTION_FACTOR].
     *
     * Previously this check returned the inverted quantity `ratioMin / ratio`
     * and a "penalty" value of 1e9 on a failure, which hid the meaning of what was measured.
     * Here three substantial conditions are checked directly: the error is finite and
     * small, it decreases, and it decreases fast enough.
     */
    @Test
    fun baseSchemeConvergesUnderRefinement() {
        val problem = VolterraProblem.V2

        fun errorAt(n: Int): Double {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val op = VolterraOperator(problem.kernel, grid, quad)
            val solver = secondKindSolver(problem, basis, funcs, op)
            return errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
        }

        val coarseError = errorAt(8)
        val fineError = errorAt(16)

        assertTrue(
            coarseError.isFinite() && coarseError < MAX_COARSE_GRID_ERROR,
            "The error on the coarse grid must be finite and small, got E_8 = $coarseError",
        )
        assertTrue(
            fineError < coarseError,
            "The error must decrease under grid refinement: E_8 = $coarseError, E_16 = $fineError",
        )

        val reductionFactor = coarseError / fineError
        val observedOrder = ln(reductionFactor) / ln(2.0)
        assertTrue(
            reductionFactor >= MIN_ERROR_REDUCTION_FACTOR,
            "The error must decrease by at least a factor of $MIN_ERROR_REDUCTION_FACTOR " +
                "(an observed order >= 2), got: E_8 = $coarseError, E_16 = $fineError, " +
                "ratio = $reductionFactor, observed order = $observedOrder",
        )
    }

    /**
     * The accuracy of the Nyström scheme on a matched problem.
     *
     * If the kernel depends only on `t` (here `K = 1 + t`), then the integrand
     * `g_t(s) = K(t) * u*(s)` at `u* = s^2` lies entirely in `span{1, s, s^2}`,
     * which coincides with the polynomial generating system. Hence the spline quadrature
     * with the weights `W_j(t) = \\int_a^t omega_j` reproduces the integral over the variable interval
     * exactly, and the approximation must coincide with the exact solution up to machine accuracy.
     */
    @Test
    fun nystromIsExactWhenIntegrandLiesInSpan() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val kernel = KernelV({ t, _ -> 1.0 + t })
        val problem = VolterraProblem(
            name = "V2nyst",
            kernel = kernel,
            exact = { s -> s * s },
            exactDeriv = { s -> 2.0 * s },
            secondKind = true,
            exactDeriv2 = { 2.0 },
        )
        val op = VolterraOperator(kernel, grid, quad)
        val solver = secondKindSolver(problem, basis, funcs, op)
        val error = errorEh({ t -> problem.exact(t) }, solver.nystrom().eval, grid)
        assertTrue(
            error < EXACT_ON_SPAN_TOLERANCE,
            "The Nyström scheme must be exact when the integrand lies in the span " +
                "of the generating system, got E_h = $error",
        )
    }
}
