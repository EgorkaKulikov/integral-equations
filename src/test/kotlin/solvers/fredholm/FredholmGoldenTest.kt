package solvers.fredholm

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Light golden/regression test (n=8,16): rebuild a minimal solver and check key E_h
 * against numbers from a real baseline run (FredholmSolver main(), table T1[F2], basis B).
 * Does NOT compare full stdout.
 */
class FredholmGoldenTest {
    private fun ehBase(p: ModelProblem, n: Int): Double {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(p.kernel, grid, numerics.GaussLegendre(8))
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) })
        return errorEh({ t -> p.exact(t) }, solver.base().eval, grid)
    }

    @Test fun f2_basis_b_n8_n16() {
        val e8 = ehBase(ModelProblem.F2, 8)
        val e16 = ehBase(ModelProblem.F2, 16)
        assertTrue(approx(e8, 1.014e-4), "E_h(n=8)=$e8 expected ~1.014e-4")
        assertTrue(approx(e16, 1.246e-5), "E_h(n=16)=$e16 expected ~1.246e-5")
        assertTrue(e16 < e8, "should converge")
    }

    private fun approx(v: Double, ref: Double) = kotlin.math.abs(v - ref) <= 0.02 * ref
}
