package solvers.fredholm

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Golden/regression-тест сплайн-метода Nyström (базовый + итерированный) для F2.
 * Числа зафиксированы из реального прогона main() (таблица T3[F2], базис B, theta).
 * Проверяется также, что xi (де Бура--Фикса) кидает понятное исключение.
 */
class FredholmNystromGoldenTest {
    private fun solver(p: ModelProblem, n: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(p.kernel, grid, numerics.GaussLegendre(8))
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    private fun approx(v: Double, ref: Double) = kotlin.math.abs(v - ref) <= 0.03 * ref

    @Test fun nystrom_f2_basis_b() {
        val (s8, g8) = solver(ModelProblem.F2, 8)
        val (s16, g16) = solver(ModelProblem.F2, 16)
        val e8 = errorEh({ t -> ModelProblem.F2.exact(t) }, s8.nystrom().eval, g8)
        val e16 = errorEh({ t -> ModelProblem.F2.exact(t) }, s16.nystrom().eval, g16)
        assertTrue(approx(e8, 2.228e-5), "E_h(n=8)=$e8 expected ~2.228e-5")
        assertTrue(approx(e16, 1.199e-6), "E_h(n=16)=$e16 expected ~1.199e-6")
        assertTrue(e16 < e8, "Nyström should converge")
    }

    @Test fun iterated_nystrom_f2_basis_b() {
        val (s8, g8) = solver(ModelProblem.F2, 8)
        val (s16, g16) = solver(ModelProblem.F2, 16)
        val e8 = errorEh({ t -> ModelProblem.F2.exact(t) }, s8.iteratedNystrom().eval, g8)
        val e16 = errorEh({ t -> ModelProblem.F2.exact(t) }, s16.iteratedNystrom().eval, g16)
        assertTrue(approx(e8, 9.488e-6), "E_h(n=8)=$e8 expected ~9.488e-6")
        assertTrue(approx(e16, 5.063e-7), "E_h(n=16)=$e16 expected ~5.063e-7")
        assertTrue(e16 < e8, "iterated Nyström should converge")
    }

    @Test fun nystrom_xi_unsupported() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = numerics.functionals.DeBoorFixFunctionals(basis)
        val op = FredholmOperator(ModelProblem.F2.kernel, grid, numerics.GaussLegendre(8))
        val s = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.F2.rhsExact(t, op) }, { t -> ModelProblem.F2.rhsExactDeriv(t, op) })
        assertFailsWith<IllegalArgumentException> { s.nystrom() }
    }
}
