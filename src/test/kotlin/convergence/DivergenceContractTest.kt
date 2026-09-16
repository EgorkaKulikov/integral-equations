package convergence

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * SINGLE DIAGNOSTIC CONTRACT FOR DIVERGENCE (task 4).
 *
 * Before the contract, the iterative schemes behaved differently: the combined
 * Nyström threw an exception, `kulkarniQuasi` in both linear solvers SILENTLY
 * returned the last iterate, and the Uryson schemes only wrote a warning to the log.
 * A user of the library could not tell a converged result from a diverging one:
 * in programmatic use a log record goes unnoticed.
 *
 * Here it is checked that the contract holds: by default an exception, and in the
 * `throwOnDivergence = false` mode a result with `converged = false` and a meaningful
 * residual.
 *
 * The problem for the divergence check is chosen NOT arbitrarily: the kernel `K = 4` on `[0,1]`
 * gives the operator norm `||L|| = 4 > 1`, so the simple iteration underlying the Kulkarni
 * scheme for quasi-interpolants and the combined Nyström diverges for any number of
 * steps. This is a property of the problem itself, not a consequence of a small
 * iteration limit.
 */
@Tag("fast")
class DivergenceContractTest {

    private companion object {
        /** Kernel with operator norm 4 > 1: the simple iteration must diverge. */
        val DIVERGENT_KERNEL = KernelF(k = { _, _ -> 4.0 })
    }

    /** Builds a solver of a deliberately diverging problem with the given policy. */
    private fun divergentSolver(
        throwOnDivergence: Boolean,
        useQuasiInterpolant: Boolean,
    ): FredholmSecondKindSolver {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        // mu is a quasi-interpolant: kulkarni() goes into the iterative branch kulkarniQuasi.
        // theta is a projector: kulkarni() solves a linear system, no iterations.
        val funcs = if (useQuasiInterpolant) AveragingFunctionals(basis) else ProjFunctionals(basis)
        val op = FredholmOperator(DIVERGENT_KERNEL, grid, GaussLegendre(8))
        return FredholmSecondKindSolver(
            basis, funcs, op, cL = 1.0,
            rhs = RhsWithDerivatives(
                value = { t -> t },
                deriv = { 1.0 },
                deriv2 = { 0.0 },
            ),
            throwOnDivergence = throwOnDivergence,
        )
    }

    /**
     * By default the Kulkarni scheme for a quasi-interpolant must THROW an exception
     * instead of returning the last iterate, as it did before.
     */
    @Test
    fun quasiKulkarniThrowsByDefaultOnDivergence() {
        val solver = divergentSolver(throwOnDivergence = true, useQuasiInterpolant = true)
        val failure = assertFailsWith<IllegalStateException> { solver.kulkarni() }
        val message = failure.message ?: ""
        // The message must be diagnostic: without the iteration count and the attained
        // residual the user cannot understand what happened.
        assertTrue(
            message.contains("convergence not reached", ignoreCase = true),
            "The message must name the cause explicitly, got: $message",
        )
        assertTrue(
            message.contains("iteration") && message.contains("required"),
            "The message must report the iteration count and the required accuracy, got: $message",
        )
    }

    /**
     * In the `throwOnDivergence = false` mode the same call must return a result
     * EXPLICITLY marked as non-converged, with a meaningful residual. This is the
     * deliberate access to a non-converged result for research scenarios.
     */
    @Test
    fun quasiKulkarniReportsNotConvergedWhenAllowed() {
        val solver = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = true)
        val solution = solver.kulkarni()
        assertFalse(solution.converged, "A diverging result must be marked converged = false")
        assertTrue(
            solution.iterations > 0,
            "The number of performed iterations must be meaningful, got ${solution.iterations}",
        )
        assertTrue(
            solution.residual > 0.0 && !solution.residual.isNaN(),
            "The attained residual must be meaningful, got ${solution.residual}",
        )
    }

    /** The same contract for the combined Nyström operator. */
    @Test
    fun combinedNystromFollowsSameContract() {
        assertFailsWith<IllegalStateException>(
            "By default the combined Nyström must throw an exception on divergence",
        ) {
            divergentSolver(throwOnDivergence = true, useQuasiInterpolant = false).combinedNystrom()
        }

        val solution = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = false)
            .combinedNystrom()
        assertFalse(solution.converged, "A diverging result must be marked converged = false")
        assertTrue(solution.residual > 0.0, "The residual must be meaningful")
    }

    /**
     * The convergence flag is INHERITED by derived schemes. The Sloan iteration on top of
     * a diverging approximation performs no iterations itself, but its result is meaningful
     * only when the underlying approximation is meaningful — otherwise `converged = true`
     * of the iterated scheme would hide the divergence of the base one.
     */
    @Test
    fun iteratedSchemesInheritConvergenceFlag() {
        val solver = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = true)
        val iterated = solver.iteratedKulkarni()
        assertFalse(
            iterated.converged,
            "The iterated scheme must inherit the divergence flag of the base one",
        )
    }

    /**
     * DIRECT schemes (a linear system without iterations) report trivial convergence
     * on a regular problem: `converged = true`, `iterations = 0`. The check guards against
     * the opposite mistake: marking a correct result as non-converged.
     */
    @Test
    fun directSchemesReportTrivialConvergence() {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
        )
        for ((name, solution) in listOf(
            "base" to solver.base(),
            "sloan" to solver.sloan(),
            "kulkarni (projector)" to solver.kulkarni(),
            "nystrom" to solver.nystrom(),
        )) {
            assertTrue(solution.converged, "Direct scheme $name must report converged = true")
            assertTrue(
                solution.iterations == 0,
                "Direct scheme $name performs no iterations, got ${solution.iterations}",
            )
        }
    }
}
