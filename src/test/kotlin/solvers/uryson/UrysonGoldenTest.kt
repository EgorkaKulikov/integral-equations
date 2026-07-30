package solvers.uryson

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import problems.uryson.secondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Golden-тест базовой схемы для задачи A (второго рода) на полиномиальном базисе.
 *
 * Эталонные значения взяты из опубликованной таблицы порядков сходимости, а не
 * зафиксированы с текущей реализации: совпадение подтверждает, что код воспроизводит
 * численный эксперимент источника (см. `docs/REFERENCES.md`).
 */
@Tag("fast")
class UrysonGoldenTest {

    private companion object {
        /** Допустимое относительное отклонение от опубликованного значения. */
        const val RELATIVE_TOLERANCE = 0.02

        /** Опубликованное значение E_h при n = 8. */
        const val PUBLISHED_ERROR_N8 = 1.006e-4

        /** Опубликованное значение E_h при n = 16. */
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
            "E_h(n=8) = $errorN8, опубликованное значение $PUBLISHED_ERROR_N8",
        )
        assertTrue(
            matches(errorN16, PUBLISHED_ERROR_N16),
            "E_h(n=16) = $errorN16, опубликованное значение $PUBLISHED_ERROR_N16",
        )
        assertTrue(errorN16 < errorN8, "Погрешность должна убывать при сгущении сетки")
    }

    private fun matches(value: Double, reference: Double) =
        abs(value - reference) <= RELATIVE_TOLERANCE * reference
}
