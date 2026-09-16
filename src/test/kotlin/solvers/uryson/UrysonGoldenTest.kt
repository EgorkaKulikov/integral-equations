package solvers.uryson

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import problems.uryson.secondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * A golden test of the base scheme for the problem A (of the second kind) on the polynomial basis.
 *
 * The baseline values are taken from a published table of convergence orders and are not
 * recorded from the current implementation: a match confirms that the code reproduces
 * the numerical experiment of the source (see `docs/REFERENCES.md`).
 */
@Tag("fast")
class UrysonGoldenTest {

    private companion object {
        /** The admissible relative deviation from the published value. */
        const val RELATIVE_TOLERANCE = 0.02

        /** The published value of E_h at n = 8. */
        const val PUBLISHED_ERROR_N8 = 1.006e-4

        /** The published value of E_h at n = 16. */
        const val PUBLISHED_ERROR_N16 = 1.243e-5
    }

    private fun baseError(n: Int): Double {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, GaussLegendre(8))
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, GaussLegendre(8))
        val solver = secondKindSolver(UrysonProblem.A, basis, funcs, space, op)
        return errorEh({ t -> UrysonProblem.A.exact(t) }, solver.base().eval, grid)
    }

    @Test
    fun problemAMatchesPublishedErrors() {
        val errorN8 = baseError(8)
        val errorN16 = baseError(16)
        assertTrue(
            matches(errorN8, PUBLISHED_ERROR_N8),
            "E_h(n=8) = $errorN8, the published value is $PUBLISHED_ERROR_N8",
        )
        assertTrue(
            matches(errorN16, PUBLISHED_ERROR_N16),
            "E_h(n=16) = $errorN16, the published value is $PUBLISHED_ERROR_N16",
        )
        assertTrue(errorN16 < errorN8, "The error must decrease under grid refinement")
    }

    private fun matches(value: Double, reference: Double) =
        abs(value - reference) <= RELATIVE_TOLERANCE * reference
}
