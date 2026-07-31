package numerics

import org.junit.jupiter.api.Tag
import solvers.uryson.SplineSpace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Инкапсуляция ХОЛОДНЫХ массивов: геттер обязан отдавать КОПИЮ, чтобы мутация у
 * вызывающего не портила источник.
 *
 * Раньше `GaussLegendre.refNodesWeights()` и поля `SplineSpace.weights/wInt/gramR`
 * отдавали наружу сами внутренние массивы. Любой вызывающий мог одной записью тихо
 * изменить квадратуру или веса СРАЗУ ДЛЯ ВСЕХ пользователей объекта, причём без
 * исключения — просто получались другие числа.
 *
 * ГРАНИЦА ПОЛИТИКИ (важно для чтения теста): копируются только холодные поля, которые
 * читаются однократно. Горячие массивы (`Grid.breakpoints`, `gNode`/`gW` операторов,
 * `ValueFunctional.nodes/coeffs`) читаются в горячих циклах десятки тысяч раз за запуск
 * и остаются read-only ПО СОГЛАШЕНИЮ — копирование на каждом обращении откатило бы
 * оптимизации горячего пути. Поэтому здесь их нет намеренно.
 */
@Tag("fast")
class DefensiveCopyTest {

    private fun space(): SplineSpace {
        val grid = Grid.uniform(8, 0.0, 1.0)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        return SplineSpace(basis, GaussLegendre(8))
    }

    /** Мутация узлов/весов, полученных из квадратуры, не меняет саму квадратуру. */
    @Test fun refNodesWeightsReturnsCopy() {
        val quad = GaussLegendre(8)
        val (nodes, weights) = quad.refNodesWeights()
        val nodesBefore = nodes.copyOf()
        val weightsBefore = weights.copyOf()

        nodes[0] = 12345.0
        weights[0] = -777.0

        val (nodesAgain, weightsAgain) = quad.refNodesWeights()
        assertTrue(nodesBefore.contentEquals(nodesAgain), "узлы квадратуры изменились после мутации копии")
        assertTrue(weightsBefore.contentEquals(weightsAgain), "веса квадратуры изменились после мутации копии")
    }

    /**
     * Мутация не должна доходить и до РЕЗУЛЬТАТА квадратуры: проверка не на равенство
     * массивов, а на само вычисляемое число (порча узлов проявилась бы именно здесь).
     */
    @Test fun mutatingRefNodesDoesNotAffectIntegral() {
        val quad = GaussLegendre(8)
        val bp = doubleArrayOf(0.0, 0.5, 1.0)
        val exact = 1.0 / 3.0
        val before = quad.integrate(bp) { t -> t * t }

        val (nodes, weights) = quad.refNodesWeights()
        for (i in nodes.indices) { nodes[i] = 0.0; weights[i] = 0.0 }

        val after = quad.integrate(bp) { t -> t * t }
        assertEquals(exact, before, 1e-14)
        assertEquals(before, after, 0.0, "интеграл изменился после мутации возвращённых узлов/весов")
    }

    /** `SplineSpace.weights` — копия: запись в неё не меняет ни поле, ни `weightsSum()`. */
    @Test fun spaceWeightsReturnsCopy() {
        val space = space()
        val sumBefore = space.weightsSum()
        val w = space.weights
        val snapshot = w.copyOf()

        w[0] = 1e9

        assertTrue(snapshot.contentEquals(space.weights), "space.weights изменились после мутации копии")
        assertEquals(sumBefore, space.weightsSum(), 0.0, "weightsSum() изменилась после мутации копии")
    }

    /** `SplineSpace.wInt` — копия. */
    @Test fun spaceWIntReturnsCopy() {
        val space = space()
        val w = space.wInt
        val snapshot = w.copyOf()

        w[0] = 1e9

        assertTrue(snapshot.contentEquals(space.wInt), "space.wInt изменились после мутации копии")
    }

    /**
     * `SplineSpace.gramR` — ГЛУБОКАЯ копия.
     *
     * Проверяется именно запись ВНУТРЬ строки (`g[0][0]`), а не подмена строки целиком:
     * поверхностный `copyOf()` защитил бы только от второго и молча пропустил первое.
     */
    @Test fun spaceGramRReturnsDeepCopy() {
        val space = space()
        val g = space.gramR
        val original = g[0][0]
        val regBefore = space.omegaReg(DoubleArray(space.dim) { 1.0 })

        g[0][0] = original + 1e9

        assertEquals(original, space.gramR[0][0], 0.0, "gramR изменилась: копия оказалась поверхностной")
        assertEquals(
            regBefore, space.omegaReg(DoubleArray(space.dim) { 1.0 }), 0.0,
            "omegaReg изменилась после мутации копии gramR",
        )
    }

    /** Разные обращения к геттеру возвращают РАЗНЫЕ объекты (иначе копии нет). */
    @Test fun coldGettersReturnDistinctInstances() {
        val space = space()
        assertTrue(space.weights !== space.weights, "weights: возвращается один и тот же массив")
        assertTrue(space.wInt !== space.wInt, "wInt: возвращается один и тот же массив")
        assertTrue(space.gramR !== space.gramR, "gramR: возвращается один и тот же массив")
        assertTrue(space.gramR[0] !== space.gramR[0], "gramR: строки не копируются (поверхностная копия)")

        val quad = GaussLegendre(8)
        assertTrue(quad.refNodesWeights().first !== quad.refNodesWeights().first, "refNodes: массив не копируется")
        assertTrue(quad.refNodesWeights().second !== quad.refNodesWeights().second, "refWeights: массив не копируется")
    }

    /** Значения копий совпадают с источником — копирование не исказило числа. */
    @Test fun copiesCarryIdenticalValues() {
        val space = space()
        assertEquals(space.dim, space.weights.size)
        assertEquals(space.dim, space.wInt.size)
        assertEquals(space.dim, space.gramR.size)
        // Сумма весов = длина отрезка: содержательный инвариант, а не только размер.
        assertEquals(1.0, space.weights.sum(), 1e-12)
        assertEquals(space.weightsSum(), space.weights.sum(), 0.0)
        // Симметрия матрицы Грама сохраняется в копии. Копия берётся один раз ДО циклов:
        // геттер глубоко копирует, и вызов внутри цикла давал бы dim^2 копий матрицы.
        val gram = space.gramR
        for (i in 0 until space.dim) for (j in 0 until space.dim) {
            assertEquals(gram[i][j], gram[j][i], 1e-12)
        }
    }
}
