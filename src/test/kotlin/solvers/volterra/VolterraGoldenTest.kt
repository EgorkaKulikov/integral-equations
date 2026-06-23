package solvers.volterra

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/** Light golden test (n=8,16) for Volterra V2, basis B, base scheme. Numbers from baseline T1[V2]. */
class VolterraGoldenTest {
    private fun ehBase(p: ModelProblem, n: Int): Double {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(p.kernel, grid, numerics.GaussLegendre(8))
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) })
        return errorEh({ t -> p.exact(t) }, solver.base().eval, grid)
    }

    @Test fun v2_basis_b_n8_n16() {
        val e8 = ehBase(ModelProblem.V2, 8)
        val e16 = ehBase(ModelProblem.V2, 16)
        assertTrue(approx(e8, 9.340e-5), "E_h(n=8)=$e8 expected ~9.340e-5")
        assertTrue(approx(e16, 1.194e-5), "E_h(n=16)=$e16 expected ~1.194e-5")
        assertTrue(e16 < e8)
    }

    private fun approx(v: Double, ref: Double) = kotlin.math.abs(v - ref) <= 0.02 * ref
}
