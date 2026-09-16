package solvers.fredholm

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import solvers.core.SolutionFunc
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue
import problems.fredholm.FredholmProblem

/**
 * Tests of the COMBINED Nyström operator L_n = P_chi L + (I - P_chi) L^N_h.
 *
 * It is exactly this operator that the published superconvergence estimates
 * O(h^7) / O(h^8) refer to (see docs/REFERENCES.md), whereas the "bare" quadrature
 * [FredholmSecondKindSolver.nystrom] does not give such an order. The tests check not the
 * theoretical constants themselves (attaining them requires very smooth data and large
 * n, where the conditioning interferes), but a PRACTICALLY CHECKABLE consequence:
 * the combined operator is substantially more accurate than the classical one on the same grid.
 */
@Tag("fast")
class CombinedNystromTest {

    private fun solverFor(problem: FredholmProblem, n: Int): Pair<FredholmSecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
        )
        return solver to grid
    }

    /**
     * The combined operator must be NOTICEABLY more accurate than the classical quadrature:
     * in the difference L - L_n the remainder of the projector enters twice.
     */
    @Test
    fun combinedIsMoreAccurateThanClassicalNystrom() {
        val problem = FredholmProblem.F2
        for (n in listOf(8, 16, 32)) {
            val (solver, grid) = solverFor(problem, n)
            val exact = { t: Double -> problem.exact(t) }
            val classicalError = errorEh(exact, solver.nystrom().eval, grid)
            val combinedError = errorEh(exact, solver.combinedNystrom().eval, grid)
            assertTrue(
                combinedError < classicalError,
                "n=$n: the combined operator ($combinedError) must be more accurate than " +
                    "the classical quadrature ($classicalError)",
            )
        }
    }

    /**
     * The convergence order of the combined operator must exceed the order of the
     * classical quadrature. The empirical order p = log2(E_n / E_2n) is checked.
     */
    @Test
    fun combinedHasHigherConvergenceOrderThanClassical() {
        val problem = FredholmProblem.F2
        val exact = { t: Double -> problem.exact(t) }

        fun errorsFor(scheme: (FredholmSecondKindSolver) -> SolutionFunc): List<Double> =
            listOf(8, 16, 32).map { n ->
                val (solver, grid) = solverFor(problem, n)
                errorEh(exact, scheme(solver).eval, grid)
            }

        val classicalErrors = errorsFor { it.nystrom() }
        val combinedErrors = errorsFor { it.combinedNystrom() }

        val classicalOrder = ln(classicalErrors[0] / classicalErrors[2]) / ln(4.0)
        val combinedOrder = ln(combinedErrors[0] / combinedErrors[2]) / ln(4.0)

        assertTrue(
            combinedOrder > classicalOrder + 0.5,
            "The order of the combined operator ($combinedOrder) must noticeably exceed " +
                "the order of the classical quadrature ($classicalOrder). " +
                "The errors: classical=$classicalErrors, combined=$combinedErrors",
        )
    }

    /** The iterated combined Nyström refines the combined one (an analogue of the Sloan iteration). */
    @Test
    fun iteratedCombinedRefinesCombined() {
        val problem = FredholmProblem.F2
        for (n in listOf(8, 16)) {
            val (solver, grid) = solverFor(problem, n)
            val exact = { t: Double -> problem.exact(t) }
            val combinedError = errorEh(exact, solver.combinedNystrom().eval, grid)
            val iteratedError = errorEh(exact, solver.iteratedCombinedNystrom().eval, grid)
            assertTrue(
                iteratedError <= combinedError,
                "n=$n: the iterated variant ($iteratedError) must not be worse than " +
                    "the original one ($combinedError)",
            )
        }
    }

    /**
     * On a problem whose exact solution lies in the span of the generating system, the combined
     * operator must be many orders more accurate than the classical quadrature.
     *
     * A note: exact (machine) accuracy is NOT required here, unlike for the base
     * scheme: the approximation u^N_h = f + L_n u^N_h lies outside the spline space, while
     * the quadrature part (I - P_chi)L^N_h does not reproduce the span exactly. In fact
     * ~3e-10 is observed at n=8 against ~5e-5 for the classical quadrature.
     */
    @Test
    fun combinedIsFarMoreAccurateThanClassicalOnSpanProblem() {
        val problem = FredholmProblem.F2span
        for (n in listOf(8, 16)) {
            val (solver, grid) = solverFor(problem, n)
            val exact = { t: Double -> problem.exact(t) }
            val classicalError = errorEh(exact, solver.nystrom().eval, grid)
            val combinedError = errorEh(exact, solver.combinedNystrom().eval, grid)
            assertTrue(
                combinedError < 1e-4 * classicalError,
                "F2span, n=$n: the combined Nyström ($combinedError) must be many " +
                    "orders more accurate than the classical one ($classicalError)",
            )
        }
    }

    /** The combined operator, like the classical one, does not support the family xi (derivatives are needed). */
    @Test
    fun combinedRejectsDerivativeFamilies() {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = splines.functionals.DeBoorFixFunctionals(basis, 1)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
            ),
        )
        kotlin.test.assertFailsWith<IllegalArgumentException> { solver.combinedNystrom() }
    }
}
