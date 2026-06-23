package solvers.uryson

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertTrue

/** Light golden test (n=8,16) for Uryson problem A, basis B, base scheme. Numbers from baseline tab:num-ury2A-order. */
class UrysonGoldenTest {
    private fun ehBase(n: Int): Double {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, GaussLegendre(8))
        val op = UrysohnOperator(ModelProblem.A.kernel, grid, GaussLegendre(8))
        val solver = SecondKindSolver(ModelProblem.A, basis, funcs, space, op)
        return errorEhEval({ t -> ModelProblem.A.exact(t) }, solver.base().eval, grid)
    }

    @Test fun ury2A_basis_b_n8_n16() {
        val e8 = ehBase(8)
        val e16 = ehBase(16)
        assertTrue(approx(e8, 1.006e-4), "E_h(n=8)=$e8 expected ~1.006e-4")
        assertTrue(approx(e16, 1.243e-5), "E_h(n=16)=$e16 expected ~1.243e-5")
        assertTrue(e16 < e8)
    }

    private fun approx(v: Double, ref: Double) = kotlin.math.abs(v - ref) <= 0.02 * ref
}
