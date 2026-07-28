package solvers.volterra

import problems.volterra.VolterraProblem
import problems.volterra.firstKindSolver
import problems.volterra.secondKindSolver
import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Тесты КОМБИНИРОВАННОГО оператора Nyström для уравнения Вольтерры.
 *
 * ВНИМАНИЕ: в отличие от уравнения Фредгольма, для Вольтерры теоретических оценок
 * суперсходимости в известной литературе НЕТ (переменный верхний предел даёт
 * t-зависимые веса и усечение последней ячейки).
 *
 * ЗАФИКСИРОВАННОЕ ЧИСЛЕННОЕ НАБЛЮДЕНИЕ (измерено на модельных задачах, базис B,
 * семейство theta, сетки n = 8, 16, 32):
 *
 *  - для уравнения Фредгольма комбинированный оператор поднимает наблюдаемый порядок
 *    примерно с 4.2 до 7.0, что согласуется с опубликованной оценкой O(h^7);
 *  - для уравнения Вольтерры такого эффекта НЕТ. На задаче V2 комбинированный
 *    оператор даже уступает классической квадратуре (1.3e-5 против 1.0e-5 при n=8),
 *    на V2exp и V2span немного выигрывает (примерно в 1.5–1.7 раза), а наблюдаемый
 *    порядок обоих вариантов остаётся около 3.8.
 *
 * Поэтому тесты НЕ утверждают превосходства комбинированного оператора для Вольтерры:
 * они проверяют лишь корректность реализации (конечность, сходимость при измельчении,
 * согласованность итерированного варианта). Утверждать здесь суперсходимость было бы
 * подгонкой выводов под теорию, полученную для другого класса уравнений.
 */
class CombinedNystromVolterraTest {

    private fun solverFor(problem: VolterraProblem, n: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = SecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        return solver to grid
    }

    /**
     * Комбинированный оператор даёт результат ТОГО ЖЕ ПОРЯДКА ТОЧНОСТИ, что и классическая
     * квадратура (в пределах одного десятичного порядка в обе стороны).
     *
     * Проверяется именно сопоставимость, а не превосходство: как отмечено в описании
     * класса, для уравнения Вольтерры выигрыша от комбинированного оператора не
     * наблюдается, и тест фиксирует это положение дел честно.
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
                    "${problem.name}, n=$n: комбинированный ($combinedError) и классический " +
                        "($classicalError) должны быть сопоставимы по точности",
                )
            }
        }
    }

    /** Результаты конечны и сходятся при измельчении сетки. */
    @Test
    fun combinedConvergesUnderRefinement() {
        val problem = VolterraProblem.V2
        val exact = { t: Double -> problem.exact(t) }
        val errors = listOf(8, 16, 32).map { n ->
            val (solver, grid) = solverFor(problem, n)
            errorEh(exact, solver.combinedNystrom().eval, grid)
        }
        assertTrue(errors.all { it.isFinite() }, "Все значения должны быть конечны: $errors")
        assertTrue(errors[1] < errors[0] && errors[2] < errors[1], "Ошибка должна убывать: $errors")
    }

    /** Итерированный вариант не хуже исходного. */
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
                "n=$n: итерированный ($iteratedError) не должен быть хуже исходного ($combinedError)",
            )
        }
    }
}
