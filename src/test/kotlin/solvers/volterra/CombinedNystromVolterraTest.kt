package solvers.volterra

import org.junit.jupiter.api.Tag
import problems.volterra.VolterraProblem
import problems.volterra.firstKindSolver
import problems.volterra.secondKindSolver
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Tests of the COMBINED Nyström operator for the Volterra equation.
 *
 * ATTENTION: unlike the Fredholm equation, for Volterra there are NO theoretical
 * superconvergence estimates in the known literature (the variable upper limit gives
 * t-dependent weights and a truncation of the last cell).
 *
 * THE RECORDED NUMERICAL OBSERVATION (measured on the model problems, basis B,
 * family theta, grids n = 8, 16, 32):
 *
 *  - for the Fredholm equation the combined operator raises the observed order
 *    from about 4.2 to 7.0, which agrees with the published estimate O(h^7);
 *  - for the Volterra equation there is NO such effect. On the problem V2 the combined
 *    operator is even worse than the classical quadrature (1.3e-5 against 1.0e-5 at n=8),
 *    on V2exp and V2span it wins slightly (by about 1.5-1.7 times), while the observed
 *    order of both variants stays around 3.8.
 *
 * Therefore the tests do NOT claim a superiority of the combined operator for Volterra:
 * they check only the correctness of the implementation (finiteness, convergence under refinement,
 * the consistency of the iterated variant). Claiming superconvergence here would be
 * fitting the conclusions to a theory obtained for another class of equations.
 */
@Tag("fast")
class CombinedNystromVolterraTest {

    private fun solverFor(problem: VolterraProblem, n: Int): Pair<VolterraSecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = VolterraSecondKindSolver(
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
     * The combined operator gives a result OF THE SAME ORDER OF ACCURACY as the classical
     * quadrature (within one decimal order in both directions).
     *
     * What is checked is exactly comparability and not superiority: as noted in the description of the
     * class, for the Volterra equation no gain from the combined operator is
     * observed, and the test records this state of affairs honestly.
     */
    @Test
    fun combinedIsComparableToClassical() {
        for (problem in listOf(VolterraProblem.V2, VolterraProblem.V2exp)) {
            for (n in listOf(8, 16)) {
                val (solver, grid) = solverFor(problem, n)
                val exact = { t: Double -> problem.exact(t) }
                val classicalError = errorEh(exact, solver.nystrom().eval, grid)
                val combinedError = errorEh(exact, solver.combinedNystrom().eval, grid)
                assertTrue(
                    combinedError < 10.0 * classicalError && classicalError < 10.0 * combinedError,
                    "${problem.name}, n=$n: the combined ($combinedError) and the classical " +
                        "($classicalError) must be comparable in accuracy",
                )
            }
        }
    }

    /** The results are finite and converge under grid refinement. */
    @Test
    fun combinedConvergesUnderRefinement() {
        val problem = VolterraProblem.V2
        val exact = { t: Double -> problem.exact(t) }
        val errors = listOf(8, 16, 32).map { n ->
            val (solver, grid) = solverFor(problem, n)
            errorEh(exact, solver.combinedNystrom().eval, grid)
        }
        assertTrue(errors.all { it.isFinite() }, "All the values must be finite: $errors")
        assertTrue(errors[1] < errors[0] && errors[2] < errors[1], "The error must decrease: $errors")
    }

    /** The iterated variant is no worse than the original one. */
    @Test
    fun iteratedCombinedIsNotWorse() {
        val problem = VolterraProblem.V2
        for (n in listOf(8, 16)) {
            val (solver, grid) = solverFor(problem, n)
            val exact = { t: Double -> problem.exact(t) }
            val combinedError = errorEh(exact, solver.combinedNystrom().eval, grid)
            val iteratedError = errorEh(exact, solver.iteratedCombinedNystrom().eval, grid)
            assertTrue(
                iteratedError <= combinedError * 1.001,
                "n=$n: the iterated one ($iteratedError) must not be worse than the original one ($combinedError)",
            )
        }
    }
}
