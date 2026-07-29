package solvers.fredholm

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.SolutionFunc
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue
import problems.fredholm.FredholmProblem

/**
 * Тесты КОМБИНИРОВАННОГО оператора Nyström L_n = P_chi L + (I - P_chi) L^N_h.
 *
 * Именно к этому оператору относятся опубликованные оценки суперсходимости
 * O(h^7) / O(h^8) (см. docs/REFERENCES.md), тогда как «голая» квадратура
 * [SecondKindSolver.nystrom] такого порядка не даёт. Тесты проверяют не сами
 * теоретические константы (для их достижения нужны очень гладкие данные и большие
 * n, где вмешивается обусловленность), а ПРАКТИЧЕСКИ ПРОВЕРЯЕМОЕ следствие:
 * комбинированный оператор существенно точнее классического на одной и той же сетке.
 */
class CombinedNystromTest {

    private fun solverFor(problem: FredholmProblem, n: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = SecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        return solver to grid
    }

    /**
     * Комбинированный оператор обязан быть ЗАМЕТНО точнее классической квадратуры:
     * в разности L - L_n остаток проектора входит дважды.
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
                "n=$n: комбинированный оператор ($combinedError) должен быть точнее " +
                    "классической квадратуры ($classicalError)",
            )
        }
    }

    /**
     * Порядок сходимости комбинированного оператора должен превосходить порядок
     * классической квадратуры. Проверяется эмпирический порядок p = log2(E_n / E_2n).
     */
    @Test
    fun combinedHasHigherConvergenceOrderThanClassical() {
        val problem = FredholmProblem.F2
        val exact = { t: Double -> problem.exact(t) }

        fun errorsFor(scheme: (SecondKindSolver) -> SolutionFunc): List<Double> =
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
            "Порядок комбинированного оператора ($combinedOrder) должен заметно превосходить " +
                "порядок классической квадратуры ($classicalOrder). " +
                "Ошибки: классические=$classicalErrors, комбинированные=$combinedErrors",
        )
    }

    /** Итерированный комбинированный Nyström уточняет комбинированный (аналог итерации Слоана). */
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
                "n=$n: итерированный вариант ($iteratedError) не должен быть хуже " +
                    "исходного ($combinedError)",
            )
        }
    }

    /**
     * На задаче, точное решение которой лежит в span порождающей системы, комбинированный
     * оператор должен быть на много порядков точнее классической квадратуры.
     *
     * Примечание: точной (машинной) точности здесь НЕ требуется, в отличие от базовой
     * схемы: приближение u^N_h = f + L_n u^N_h лежит вне сплайнового пространства, а
     * квадратурная часть (I - P_chi)L^N_h не воспроизводит span точно. Фактически
     * наблюдается ~3e-10 при n=8 против ~5e-5 у классической квадратуры.
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
                "F2span, n=$n: комбинированный Nyström ($combinedError) должен быть на много " +
                    "порядков точнее классического ($classicalError)",
            )
        }
    }

    /** Комбинированный оператор, как и классический, не поддерживает семейство xi (нужны производные). */
    @Test
    fun combinedRejectsDerivativeFamilies() {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = numerics.functionals.DeBoorFixFunctionals(basis, 1)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = SecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
        )
        kotlin.test.assertFailsWith<IllegalArgumentException> { solver.combinedNystrom() }
    }
}
