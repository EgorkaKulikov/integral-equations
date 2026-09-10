package solvers.core

import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.functionals.ValueFunctional
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals


/**
 * Тесты индексации опорных точек [SupportPoints] (пункт 8.2 плана).
 *
 * Отвечают на два независимых вопроса.
 *
 * 1. УСТРАНЕНА ЛИ ХРУПКОСТЬ. Раньше индекс точки искался ПО ЗНАЧЕНИЮ в
 *    `HashMap<Double, Int>`, то есть требовал ПОБИТОВОГО совпадения `Double`. Точка,
 *    пересчитанная другим — математически эквивалентным — путём, отличается в младшем
 *    бите, и прежняя схема на ней ломалась: [realFamiliesDisagreeBitwiseOnTheSameMidpoint].
 *    Новая индексация по паре «функционал, узел» плюс слияние по явному допуску такую
 *    точку обрабатывает: [samePointViaTwoPathsGetsSingleIndex].
 *
 * 2. РАБОТАЕТ ЛИ ДОПУСК. Слияние точек управляется явным `mergeEps`: при нулевом допуске
 *    побитово различные точки остаются разными ([zeroToleranceKeepsBitwiseDistinctPointsApart]),
 *    цепочка близких точек не схлопывается дальше допуска
 *    ([chainOfNearPointsDoesNotCollapseBeyondTolerance]), отрицательный допуск отвергается.
 */
@Tag("fast")
class SupportPointsTest {

    /**
     * Пара «одна и та же точка, посчитанная двумя путями» — не выдуманная, а взятая
     * с реальной сетки: середина интервала `[x_1, x_2]` сетки `Grid.graded(8)`.
     *
     * `0.5*(p+q)` (так считает [ProjFunctionals.buildTheta]) и `p + 0.5*(q-p)` (так считают
     * [AveragingFunctionals] и [ThreePointFunctionals] при theta = 1/2) дают РАЗНЫЕ числа,
     * отличающиеся ровно на 1 ulp. Обе формулы законны; расхождение — свойство IEEE-754.
     */
    private class UlpPair {
        val grid: Grid = Grid.graded(8)
        val p: Double = grid.x(1)
        val q: Double = grid.x(2)
        val viaAverage: Double = 0.5 * (p + q)
        val viaOffset: Double = p + 0.5 * (q - p)
    }

    @Test
    fun twoEquivalentMidpointFormulasDifferByOneUlp() {
        val pair = UlpPair()
        assertNotEquals(pair.viaAverage, pair.viaOffset, "исходная посылка теста: пути различаются побитово")
        assertEquals(
            1.0,
            abs(pair.viaAverage - pair.viaOffset) / Math.ulp(pair.viaAverage),
            0.0,
            "расхождение путей должно быть ровно 1 ulp",
        )
    }

    /**
     * Оба пути вычисления середины ИСПОЛЬЗУЮТСЯ В ПРОЕКТЕ ОДНОВРЕМЕННО, и на реальной
     * сетке они дают РАЗНЫЕ числа для одного и того же интервала.
     *
     * [ProjFunctionals] считает середину как `0.5*(p+q)`, а [ThreePointFunctionals]
     * и [AveragingFunctionals] — как `p + theta*(q-p)`. Сейчас это безопасно только потому,
     * что схемы работают с ОДНИМ семейством за раз: смешение двух источников точек
     * ломало бы поиск по значению. Тест фиксирует, что проблема не выдумана.
     */
    @Test
    fun realFamiliesDisagreeBitwiseOnTheSameMidpoint() {
        val grid = Grid.graded(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        // theta_0: (x_0, mid(x_0,x_1), mid(x_1,x_2), mid(x_2,x_3), x_3) — середина [x_1,x_2] под № 2.
        val theta0 = ProjFunctionals(basis).chi(0) as ValueFunctional
        // lambda_0: (x_1, x_1 + 1/2 (x_2 - x_1), x_2) — та же середина под № 1.
        val lambda0 = ThreePointFunctionals(basis).chi(0) as ValueFunctional
        val fromTheta = theta0.nodes[2]
        val fromLambda = lambda0.nodes[1]
        assertEquals(1.0, abs(fromTheta - fromLambda) / Math.ulp(fromTheta), 0.0, "расхождение ровно 1 ulp")
        assertFailsWith<NoSuchElementException>("поиск по значению не находит ту же точку") {
            HashMap<Double, Int>().apply { put(fromTheta, 0) }.getValue(fromLambda)
        }
        // Новая индексация с теми же двумя функционалами отрабатывает корректно.
        val mixed = SupportPoints.byFirstOccurrence(arrayOf(theta0, lambda0), grid.breakpointInclusionEps)
        assertEquals(mixed.indexOf(0, 2), mixed.indexOf(1, 1), "середина [x_1,x_2] — одна точка")
    }

    /**
     * ПОСЛЕ ПРАВКИ: та же пара «пересчитанных» точек сливается в одну, и оба функционала
     * получают ОДИН индекс — ни исключения, ни дубликата в наборе.
     */
    @Test
    fun samePointViaTwoPathsGetsSingleIndex() {
        val pair = UlpPair()
        val eps = pair.grid.breakpointInclusionEps
        val byAverage = ValueFunctional(doubleArrayOf(pair.p, pair.viaAverage, pair.q), doubleArrayOf(1.0, 1.0, 1.0))
        val byOffset = ValueFunctional(doubleArrayOf(pair.viaOffset, pair.q), doubleArrayOf(1.0, 1.0))
        val functionals = arrayOf(byAverage, byOffset)

        for (support in listOf(
            SupportPoints.byAscendingValue(functionals, eps),
            SupportPoints.byFirstOccurrence(functionals, eps),
        )) {
            assertEquals(3, support.size, "точки, различающиеся на 1 ulp, обязаны слиться в одну")
            assertEquals(
                support.indexOf(0, 1),
                support.indexOf(1, 0),
                "один и тот же математический узел обязан получить один индекс",
            )
            assertEquals(support.indexOf(0, 2), support.indexOf(1, 1), "узел x_2 общий у обоих функционалов")
        }
    }

    /** Без допуска (`mergeEps = 0`) слияния нет: допуск действительно работает, а не декоративен. */
    @Test
    fun zeroToleranceKeepsBitwiseDistinctPointsApart() {
        val pair = UlpPair()
        val functionals = arrayOf(
            ValueFunctional(doubleArrayOf(pair.viaAverage), doubleArrayOf(1.0)),
            ValueFunctional(doubleArrayOf(pair.viaOffset), doubleArrayOf(1.0)),
        )
        assertEquals(2, SupportPoints.byAscendingValue(functionals, 0.0).size)
        assertEquals(1, SupportPoints.byAscendingValue(functionals, pair.grid.breakpointInclusionEps).size)
    }

    /** Отрицательный допуск — ошибка вызывающего, а не тихо игнорируемое значение. */
    @Test
    fun negativeToleranceRejected() {
        val functionals = arrayOf(ValueFunctional(doubleArrayOf(0.0), doubleArrayOf(1.0)))
        assertFailsWith<IllegalArgumentException> { SupportPoints.byAscendingValue(functionals, -1e-16) }
        assertFailsWith<IllegalArgumentException> { SupportPoints.byFirstOccurrence(functionals, Double.NaN) }
    }

    /**
     * Цепочка попарно близких точек НЕ схлопывается в одну группу неограниченной ширины:
     * сравнение идёт с ПРЕДСТАВИТЕЛЕМ группы, а не с предыдущей точкой.
     */
    @Test
    fun chainOfNearPointsDoesNotCollapseBeyondTolerance() {
        val eps = 1e-3
        val functionals = arrayOf(
            ValueFunctional(doubleArrayOf(0.0, 0.0009, 0.0018, 0.0027), DoubleArray(4) { 1.0 }),
        )
        val support = SupportPoints.byAscendingValue(functionals, eps)
        assertEquals(2, support.size, "группы обязаны быть шириной не более допуска")
        assertEquals(0, support.indexOf(0, 0))
        assertEquals(0, support.indexOf(0, 1))
        assertEquals(1, support.indexOf(0, 2))
        assertEquals(1, support.indexOf(0, 3))
    }
}
