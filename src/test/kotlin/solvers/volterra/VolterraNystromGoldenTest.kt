package solvers.volterra

import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Golden/regression-тест сплайн-метода Nyström (базовый + итерированный) для V2.
 * Числа зафиксированы из реального прогона main() (таблица T3[V2], базис B, theta).
 * Проверяется также, что xi (де Бура--Фикса) кидает понятное исключение.
 */
class VolterraNystromGoldenTest {
    private fun solver(p: ModelProblem, n: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(p.kernel, grid, numerics.GaussLegendre(8))
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    private fun approx(v: Double, ref: Double) = kotlin.math.abs(v - ref) <= 0.03 * ref

    @Test fun nystrom_v2_basis_b() {
        val (s8, g8) = solver(ModelProblem.V2, 8)
        val (s16, g16) = solver(ModelProblem.V2, 16)
        val e8 = errorEh({ t -> ModelProblem.V2.exact(t) }, s8.nystrom().eval, g8)
        val e16 = errorEh({ t -> ModelProblem.V2.exact(t) }, s16.nystrom().eval, g16)
        assertTrue(approx(e8, 9.969e-6), "E_h(n=8)=$e8 expected ~9.969e-6")
        assertTrue(approx(e16, 7.602e-7), "E_h(n=16)=$e16 expected ~7.602e-7")
        assertTrue(e16 < e8, "Nyström should converge")
    }

    @Test fun iterated_nystrom_v2_basis_b() {
        val (s8, g8) = solver(ModelProblem.V2, 8)
        val (s16, g16) = solver(ModelProblem.V2, 16)
        val e8 = errorEh({ t -> ModelProblem.V2.exact(t) }, s8.iteratedNystrom().eval, g8)
        val e16 = errorEh({ t -> ModelProblem.V2.exact(t) }, s16.iteratedNystrom().eval, g16)
        assertTrue(approx(e8, 9.279e-7), "E_h(n=8)=$e8 expected ~9.279e-7")
        assertTrue(approx(e16, 4.816e-8), "E_h(n=16)=$e16 expected ~4.816e-8")
        assertTrue(e16 < e8, "iterated Nyström should converge")
    }

    /** RAW xi (с производной) ПО-ПРЕЖНЕМУ не поддерживается в Nyström (usesDerivative=true). */
    @Test fun nystrom_xi_unsupported() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = numerics.functionals.DeBoorFixFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V2.kernel, grid, numerics.GaussLegendre(8))
        val s = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.V2.rhsExact(t, op) }, { t -> ModelProblem.V2.rhsExactDeriv(t, op) })
        assertFailsWith<IllegalArgumentException> { s.nystrom() }
    }

    private fun xiTildeSolver(p: ModelProblem, n: Int, r: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = numerics.functionals.DiscreteDeBoorFixFunctionals(basis, r)
        val op = VolterraOperator(p.kernel, grid, numerics.GaussLegendre(8))
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    /** Дискретизованный xitilde^{<1>},xitilde^{<2>} ПРИНИМАЕТСЯ Nyström и сходится на V2. */
    @Test fun nystrom_xitilde_v2_converges() {
        for (r in 1..2) {
            val (s8, g8) = xiTildeSolver(ModelProblem.V2, 8, r)
            val (s16, g16) = xiTildeSolver(ModelProblem.V2, 16, r)
            val e8 = errorEh({ t -> ModelProblem.V2.exact(t) }, s8.nystrom().eval, g8)
            val e16 = errorEh({ t -> ModelProblem.V2.exact(t) }, s16.nystrom().eval, g16)
            assertTrue(e16 < e8, "xitilde<$r> Nyström should converge: e8=$e8 e16=$e16")
            assertTrue(e8 < 1e-1, "xitilde<$r> Nyström e8=$e8 unexpectedly large")
        }
    }

    /** Health-check: xitilde value-only — аргументы-производные игнорируются. */
    @Test fun xitilde_is_value_only() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        for (r in 1..2) {
            val funcs = numerics.functionals.DiscreteDeBoorFixFunctionals(basis, r)
            assertTrue(!funcs.usesDerivative, "xitilde<$r> must be value-only")
            val f = { t: Double -> Math.sin(t) + t * t }
            for (j in -2..grid.n - 1) {
                val withDeriv = funcs.chi(j).apply(f, { 42.0 }, { 99.0 })
                val withoutDeriv = funcs.chi(j).apply(f, { 0.0 }, { 0.0 })
                assertTrue(kotlin.math.abs(withDeriv - withoutDeriv) < 1e-15,
                    "xitilde<$r> chi($j) used derivative argument")
            }
        }
    }
}
