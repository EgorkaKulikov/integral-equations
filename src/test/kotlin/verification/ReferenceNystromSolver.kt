package verification

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos

/**
 * ЭТАЛОННЫЙ метод Нюстрёма для СВЕРКИ. НЕ ЧАСТЬ БИБЛИОТЕКИ.
 *
 * Назначение и почему он существует отдельно
 * ------------------------------------------
 * Все штатные проверки проекта написаны на его же коде и потому подтверждают лишь
 * внутреннюю согласованность. Этот класс — независимая реализация классического
 * метода Нюстрёма (учебная схема, [Atkinson 1997], гл. 4) для уравнения Фредгольма
 * второго рода
 *
 *     u(t) - lambda * ∫_a^b K(t,s) u(s) ds = f(t),
 *
 * которая служит ВНЕШНИМ эталоном внутри тестового прогона: с ней сверяются решения
 * схем проекта, не выходя в Python (сверка со SciPy — отдельный контур,
 * `tools/verify_with_scipy.py`, слои L6a/L6b).
 *
 * НЕЗАВИСИМОСТЬ — ГЛАВНОЕ ТРЕБОВАНИЕ, а не пожелание
 * ---------------------------------------------------
 * Эталон, использующий проверяемый код, замыкается сам на себя: ошибка, лежащая в
 * общем для обоих звене, сократится и станет невидимой. Поэтому здесь СОЗНАТЕЛЬНО
 * НЕ используется ничего из `src/main`:
 *
 *  - НЕТ сплайнового базиса (`MinimalSplineBasis`) и аппроксимационных функционалов
 *    (`ProjFunctionals`, `AveragingFunctionals`) — метод Нюстрёма их не требует
 *    в принципе: неизвестными служат значения решения в узлах квадратуры;
 *  - НЕТ `numerics.GaussLegendre` — узлы и веса вычисляются здесь ([legendreNodes]);
 *  - НЕТ `LinearAlgebra`/`DenseOps`/бэкендов проекта — СЛАУ решается здесь
 *    ([solveDense]) методом Гаусса с выбором главного элемента по столбцу;
 *  - НЕТ `Grid`: сетка проекта к схеме Нюстрёма отношения не имеет.
 *
 * Единственный вход извне — сами `kernel`, `rhs`, `lambda`, `a`, `b`: это ПОСТАНОВКА
 * задачи, а не средство её решения. Формально независимость подтверждается тем, что
 * список импортов файла состоит только из `kotlin.math`.
 *
 * Достижимая точность
 * -------------------
 * На гладких ядрах квадратура Гаусса–Лежандра сходится экспоненциально, поэтому уже
 * при [nodeCount] = 64 решение воспроизводится на уровне 1e-14…1e-15 — на порядки
 * точнее схем проекта на сетках n = 8..32. Это и делает его пригодным эталоном:
 * измеряемая величина (погрешность схемы проекта) заведомо больше погрешности
 * самого эталона.
 *
 * @param kernel ядро `K(t,s)`.
 * @param rhs правая часть `f(t)`.
 * @param lambda параметр при интегральном операторе.
 * @param a левая граница отрезка.
 * @param b правая граница отрезка.
 * @param nodeCount число узлов квадратуры (>= 2).
 */
class ReferenceNystromSolver(
    private val kernel: (Double, Double) -> Double,
    private val rhs: (Double) -> Double,
    private val lambda: Double = 1.0,
    private val a: Double = 0.0,
    private val b: Double = 1.0,
    private val nodeCount: Int = 64,
) {
    init {
        require(nodeCount >= 2) { "ReferenceNystromSolver: требуется nodeCount >= 2, получено $nodeCount" }
        require(b > a) { "ReferenceNystromSolver: требуется b > a, получено a=$a, b=$b" }
    }

    /** Узлы и веса квадратуры, отображённые с `[-1,1]` на `[a,b]`. */
    private val nodes: DoubleArray
    private val weights: DoubleArray

    /** Значения решения в узлах квадратуры — неизвестные системы Нюстрёма. */
    private val nodeValues: DoubleArray

    init {
        val (raw, rawWeights) = legendreNodes(nodeCount)
        val half = 0.5 * (b - a)
        val mid = 0.5 * (a + b)
        nodes = DoubleArray(nodeCount) { mid + half * raw[it] }
        weights = DoubleArray(nodeCount) { half * rawWeights[it] }
        // Система Нюстрёма: (I - lambda * w_j * K(t_i, t_j)) u = f.
        val matrix = Array(nodeCount) { i ->
            DoubleArray(nodeCount) { j ->
                (if (i == j) 1.0 else 0.0) - lambda * weights[j] * kernel(nodes[i], nodes[j])
            }
        }
        nodeValues = solveDense(matrix, DoubleArray(nodeCount) { rhs(nodes[it]) })
    }

    /**
     * Значение эталонного решения в произвольной точке [t].
     *
     * Используется формула Нюстрёма (интерполяция самим уравнением, а не сплайном):
     * `u(t) = f(t) + lambda * sum_j w_j K(t, t_j) u_j`. Она сохраняет точность
     * квадратуры и вне узлов, поэтому отдельная интерполяция не нужна.
     */
    fun eval(t: Double): Double {
        var sum = 0.0
        for (j in 0 until nodeCount) sum += weights[j] * kernel(t, nodes[j]) * nodeValues[j]
        return rhs(t) + lambda * sum
    }

    companion object {

        /**
         * Интеграл `∫_a^b f(s) ds` независимой квадратурой Гаусса–Лежандра.
         *
         * Вынесено в публичный метод, потому что правая часть модельной задачи
         * `f = u - Ku` тоже содержит интеграл, и взять его оператором проекта значило бы
         * втянуть проверяемый код в эталон через чёрный ход.
         *
         * На гладких подынтегральных функциях 96 узлов дают машинную точность.
         */
        fun integrate(
            a: Double,
            b: Double,
            nodeCount: Int = 96,
            f: (Double) -> Double,
        ): Double {
            require(nodeCount >= 2) { "integrate: требуется nodeCount >= 2, получено $nodeCount" }
            require(b > a) { "integrate: требуется b > a, получено a=$a, b=$b" }
            val (raw, rawWeights) = legendreNodes(nodeCount)
            val half = 0.5 * (b - a)
            val mid = 0.5 * (a + b)
            var sum = 0.0
            for (i in 0 until nodeCount) sum += half * rawWeights[i] * f(mid + half * raw[i])
            return sum
        }

        /**
         * Узлы и веса квадратуры Гаусса–Лежандра на `[-1,1]`.
         *
         * Алгоритм: метод Ньютона по нулям многочлена Лежандра `P_m`, значения и
         * производная которого берутся из трёхчленной рекуррентности
         * `(k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}`, вес `w = 2 / ((1-x^2) P'_m(x)^2)`.
         * Начальное приближение — асимптотика Трикоми `cos(pi (i - 1/4)/(m + 1/2))`.
         *
         * Реализовано ЗДЕСЬ, а не взято из `numerics.GaussLegendre`, именно затем,
         * чтобы эталон не зависел от проверяемого кода: узлы квадратуры входят и в
         * схемы проекта, поэтому общая ошибка в них сократилась бы при сравнении.
         */
        fun legendreNodes(m: Int): Pair<DoubleArray, DoubleArray> {
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until m) {
                var x = cos(PI * (i + 0.75) / (m + 0.5))
                var derivative = 0.0
                // Ньютон сходится квадратично; 100 итераций — заведомый запас,
                // фактически хватает 4-5. Ограничение обязательно: без него
                // ошибка в формуле дала бы вечный цикл вместо внятного отказа.
                var converged = false
                for (iteration in 0 until 100) {
                    var previous = 1.0
                    var current = x
                    for (k in 1 until m) {
                        val next = ((2 * k + 1) * x * current - k * previous) / (k + 1)
                        previous = current
                        current = next
                    }
                    // current = P_m(x), previous = P_{m-1}(x).
                    derivative = m * (x * current - previous) / (x * x - 1.0)
                    val step = current / derivative
                    x -= step
                    if (abs(step) < 1e-15) {
                        converged = true
                        break
                    }
                }
                check(converged) { "legendreNodes: метод Ньютона не сошёлся для m=$m, узел $i" }
                nodes[i] = x
                weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative)
            }
            return nodes to weights
        }

        /**
         * Решение плотной системы `A x = rhs` методом Гаусса с выбором главного
         * элемента по столбцу.
         *
         * Реализовано ЗДЕСЬ по той же причине, что и квадратура: обращение к
         * `LinearAlgebra` проекта сделало бы эталон зависимым от проверяемого кода.
         * Матрица Нюстрёма на гладком ядре хорошо обусловлена (порядка единиц),
         * поэтому простого частичного выбора достаточно — уточнения не требуются.
         */
        fun solveDense(matrix: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
            val size = rhs.size
            val a = Array(size) { matrix[it].copyOf() }
            val x = rhs.copyOf()
            for (column in 0 until size) {
                var pivot = column
                for (row in column + 1 until size) {
                    if (abs(a[row][column]) > abs(a[pivot][column])) pivot = row
                }
                check(abs(a[pivot][column]) > 0.0) {
                    "solveDense: матрица вырождена, нулевой столбец $column"
                }
                if (pivot != column) {
                    val rowTmp = a[pivot]; a[pivot] = a[column]; a[column] = rowTmp
                    val valueTmp = x[pivot]; x[pivot] = x[column]; x[column] = valueTmp
                }
                for (row in column + 1 until size) {
                    val factor = a[row][column] / a[column][column]
                    if (factor == 0.0) continue
                    for (k in column until size) a[row][k] -= factor * a[column][k]
                    x[row] -= factor * x[column]
                }
            }
            for (row in size - 1 downTo 0) {
                var sum = x[row]
                for (k in row + 1 until size) sum -= a[row][k] * x[k]
                x[row] = sum / a[row][row]
            }
            return x
        }
    }
}
