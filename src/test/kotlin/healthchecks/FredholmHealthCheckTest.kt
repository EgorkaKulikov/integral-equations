package healthchecks

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.fredholm.secondKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.KernelF
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Health-checks, СПЕЦИФИЧНЫЕ для решателя уравнения Фредгольма.
 *
 * Общие проверки вычислительного ядра (сплайны, функционалы, квадратура) вынесены
 * в [SplineCoreHealthCheckTest] и здесь не дублируются.
 *
 * Здесь остаются только те свойства, которые касаются самого интегрального
 * оператора и построенных на нём схем: согласованность правой части, сходимость
 * базовой схемы и точность квадратурной схемы Nyström.
 */
@Tag("fast")
class FredholmHealthCheckTest {

    private companion object {
        /** Порог для схем, точных на порождающем пространстве. */
        const val EXACT_ON_SPAN_TOLERANCE = 1e-8

        /**
         * Минимальный допустимый множитель убывания погрешности при удвоении числа
         * узлов. Значение 4 соответствует наблюдаемому порядку не ниже второго
         * (`2^2 = 4`) — заведомо слабее теоретического порядка 3, чтобы проверка
         * реагировала на поломку схемы, а не на колебания константы.
         */
        const val MIN_ERROR_REDUCTION_FACTOR = 4.0

        /** Верхняя граница абсолютной погрешности на грубой сетке (защита от расходимости). */
        const val MAX_COARSE_GRID_ERROR = 1e-1
    }

    private val quad = GaussLegendre(8)

    /**
     * Согласованность правой части с оператором: на задаче, решение которой лежит
     * в порождающем пространстве (`u* = t^2` при полиномиальном базисе), базовая
     * схема обязана давать машинную точность.
     *
     * Проверка ловит рассогласование между способом построения правой части
     * `f = u* - K u*` и способом её дискретизации в решателе: любая несогласованность
     * немедленно нарушает точное воспроизведение.
     */
    @Test
    fun rightHandSideIsConsistentWithOperator() {
        val problem = FredholmProblem.F2span
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, quad)
        val solver = secondKindSolver(problem, basis, funcs, op)
        val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
        assertTrue(
            error < EXACT_ON_SPAN_TOLERANCE,
            "На задаче F2span (u* = t^2 лежит в span порождающей системы B) базовая схема " +
                "должна быть точна, получено E_h = $error",
        )
    }

    /**
     * Сходимость базовой схемы: при удвоении числа узлов погрешность должна убывать
     * не менее чем в [MIN_ERROR_REDUCTION_FACTOR] раз.
     *
     * Ранее эта проверка возвращала инвертированную величину `ratioMin / ratio`
     * и «штрафное» значение 1e9 при провале, что скрывало смысл измеряемого.
     * Здесь проверяются напрямую три содержательных условия: погрешность конечна и
     * мала, она убывает, и убывает достаточно быстро.
     */
    @Test
    fun baseSchemeConvergesUnderRefinement() {
        val problem = FredholmProblem.F2

        fun errorAt(n: Int): Double {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val op = FredholmOperator(problem.kernel, grid, quad)
            val solver = secondKindSolver(problem, basis, funcs, op)
            return errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
        }

        val coarseError = errorAt(8)
        val fineError = errorAt(16)

        assertTrue(
            coarseError.isFinite() && coarseError < MAX_COARSE_GRID_ERROR,
            "Погрешность на грубой сетке должна быть конечной и малой, получено E_8 = $coarseError",
        )
        assertTrue(
            fineError < coarseError,
            "Погрешность обязана убывать при измельчении сетки: E_8 = $coarseError, E_16 = $fineError",
        )

        val reductionFactor = coarseError / fineError
        val observedOrder = ln(reductionFactor) / ln(2.0)
        assertTrue(
            reductionFactor >= MIN_ERROR_REDUCTION_FACTOR,
            "Погрешность должна убывать не менее чем в $MIN_ERROR_REDUCTION_FACTOR раза " +
                "(наблюдаемый порядок >= 2), получено: E_8 = $coarseError, E_16 = $fineError, " +
                "отношение = $reductionFactor, наблюдаемый порядок = $observedOrder",
        )
    }

    /**
     * Точность схемы Nyström на согласованной задаче.
     *
     * Если ядро зависит только от `t` (здесь `K = 1 + t`), то подынтегральная функция
     * `g_t(s) = K(t) * u*(s)` при `u* = s^2` целиком лежит в `span{1, s, s^2}`,
     * совпадающем с полиномиальной порождающей системой. Значит, сплайновая квадратура
     * Nyström воспроизводит интеграл точно, и приближение обязано совпасть с точным
     * решением до машинной точности.
     */
    @Test
    fun nystromIsExactWhenIntegrandLiesInSpan() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val kernel = KernelF({ t, _ -> 1.0 + t })
        val problem = FredholmProblem(
            name = "F2nyst",
            kernel = kernel,
            exact = { s -> s * s },
            exactDeriv = { s -> 2.0 * s },
            secondKind = true,
            exactDeriv2 = { 2.0 },
        )
        val op = FredholmOperator(kernel, grid, quad)
        val solver = secondKindSolver(problem, basis, funcs, op)
        val error = errorEh({ t -> problem.exact(t) }, solver.nystrom().eval, grid)
        assertTrue(
            error < EXACT_ON_SPAN_TOLERANCE,
            "Схема Nyström должна быть точна, когда подынтегральная функция лежит в span " +
                "порождающей системы, получено E_h = $error",
        )
    }
}
