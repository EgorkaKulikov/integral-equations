package solvers.core

import splines.functionals.ValueFunctional

/**
 * Набор РАЗЛИЧНЫХ опорных точек семейства функционалов-значений вместе с явной
 * индексацией по паре «номер функционала, номер узла внутри функционала».
 *
 * ЗАЧЕМ. Схемы Nyström и сборка коллокационных матриц устроены так: точки всех
 * функционалов `chi_j` объединяются в один набор `{eta_r}`, а затем для каждого узла
 * `s_{j,q}` нужен его номер `r` в этом наборе. Раньше соответствие искалось ПО ЗНАЧЕНИЮ
 * через `HashMap<Double, Int>` — то есть требовало ПОБИТОВОГО совпадения `Double`.
 * Работало это лишь потому, что и заполнение карты, и чтение брали значения из ОДНОГО
 * И ТОГО ЖЕ массива `ValueFunctional.nodes` кэшированного функционала: совпадение
 * обеспечивалось идентичностью объекта, а не арифметикой. Любой пересчёт той же
 * математической точки другим (эквивалентным) путём даёт расхождение в младшем бите,
 * и поиск падал бы с `NoSuchElementException` — при том что задача была бы корректной.
 *
 * ЧТО СДЕЛАНО. Соответствие `(j, q) -> r` вычисляется ОДИН РАЗ при сборке набора и
 * дальше читается по индексам; поиска по значению во время использования нет вовсе.
 * Совпадение точек разных функционалов определяется по ЯВНОМУ допуску [mergeEps]
 * (см. фабрики), а не по точному равенству.
 *
 * ПОРЯДОК ТОЧЕК — часть контракта, а не деталь реализации: у Фредгольма и Вольтерры от
 * него зависит порядок суммирования агрегированных весов, у Урысона — нумерация
 * неизвестных, порядок строк матрицы Ньютона и вектор начального приближения. Поэтому
 * фабрик две, и каждая воспроизводит ровно тот порядок, который был у заменённого кода:
 * [byAscendingValue] (было `sortedSetOf`) и [byFirstOccurrence] (было `LinkedHashSet`).
 *
 * @property points различные опорные точки. READ-ONLY ПО СОГЛАШЕНИЮ: содержимое нельзя
 *   изменять. Копия не возвращается сознательно — это ГОРЯЧИЕ данные: массив читается
 *   во внутренних циклах сборки матриц Nyström (порядка `P^2` обращений на систему).
 */
class SupportPoints private constructor(
    val points: DoubleArray,
    private val indexByNode: Array<IntArray>,
) {
    /** Число различных опорных точек. */
    val size: Int get() = points.size

    /**
     * Номер точки `nodes[node]` функционала [functional] в наборе [points].
     *
     * @param functional порядковый номер функционала в массиве, переданном фабрике
     *   (для семейств с математическим индексом `j = -2..n-1` это `j + 2`).
     * @param node номер узла внутри `ValueFunctional.nodes`.
     */
    fun indexOf(functional: Int, node: Int): Int = indexByNode[functional][node]

    companion object {
        /**
         * Набор точек, упорядоченный ПО ВОЗРАСТАНИЮ ЗНАЧЕНИЯ (схемы Nyström Фредгольма
         * и Вольтерры).
         *
         * Воспроизводит прежнее `sortedSetOf<Double>()`: тот же набор и тот же порядок,
         * с единственным отличием — слияние идёт по допуску [mergeEps], а не по точному
         * равенству. Представителем группы слитых точек служит НАИМЕНЬШАЯ из них
         * (при точном совпадении, как сейчас, выбор безразличен и результат побитово
         * тот же, что у `TreeSet`).
         *
         * @param mergeEps неотрицательный допуск слияния: точки `u <= v` считаются одной,
         *   если `v - u <= mergeEps`. Осмысленное значение — `Grid.breakpointInclusionEps`.
         */
        fun byAscendingValue(functionals: Array<ValueFunctional>, mergeEps: Double): SupportPoints {
            val slots = Slots(functionals)
            val clusters = slots.clusterByValue(mergeEps)
            val index = slots.emptyIndex()
            for (slot in slots.order) index[slots.funcOf[slot]][slots.nodeOf[slot]] = clusters.clusterOf[slot]
            return SupportPoints(clusters.representatives, index)
        }

        /**
         * Набор точек в ПОРЯДКЕ ПЕРВОГО ВСТРЕЧЕННОГО ВХОЖДЕНИЯ при обходе «функционалы по
         * возрастанию номера, внутри — узлы по возрастанию номера» (схемы Урысона).
         *
         * Воспроизводит прежнее `LinkedHashSet<Double>()`: тот же набор и тот же порядок,
         * с тем же единственным отличием — слияние по допуску [mergeEps]. Представителем
         * группы слитых точек служит ПЕРВАЯ ВСТРЕЧЕННАЯ (так же ведёт себя `LinkedHashSet`,
         * сохраняющий ранее добавленный элемент).
         *
         * Порядок вставки НЕ заменяется сортировкой намеренно: у Урысона он задаёт
         * нумерацию неизвестных и порядок строк матрицы Ньютона.
         *
         * @param mergeEps неотрицательный допуск слияния (см. [byAscendingValue]).
         */
        fun byFirstOccurrence(functionals: Array<ValueFunctional>, mergeEps: Double): SupportPoints {
            val slots = Slots(functionals)
            val clusters = slots.clusterByValue(mergeEps)
            val index = slots.emptyIndex()
            // Кластеры пронумерованы по возрастанию значения; перенумеровываем их в порядке
            // первого вхождения при ЛИНЕЙНОМ обходе слотов (он и есть порядок вставки).
            val renumbered = IntArray(clusters.representatives.size) { UNASSIGNED }
            val points = DoubleArray(clusters.representatives.size)
            var assigned = 0
            for (slot in 0 until slots.total) {
                val cluster = clusters.clusterOf[slot]
                if (renumbered[cluster] == UNASSIGNED) {
                    renumbered[cluster] = assigned
                    points[assigned] = slots.value[slot]
                    assigned++
                }
                index[slots.funcOf[slot]][slots.nodeOf[slot]] = renumbered[cluster]
            }
            return SupportPoints(points, index)
        }

        /** Маркер «кластеру ещё не присвоен номер» в перенумерации [byFirstOccurrence]. */
        private const val UNASSIGNED = -1

        /**
         * Плоский разбор всех узлов всех функционалов: слот — это пара «функционал, узел».
         *
         * Слоты пронумерованы в порядке обхода (функционалы по возрастанию, внутри — узлы
         * по возрастанию), поэтому линейный обход слотов и есть порядок вставки.
         */
        private class Slots(functionals: Array<ValueFunctional>) {
            val sizes: IntArray = IntArray(functionals.size) { functionals[it].nodes.size }
            val total: Int = sizes.sum()
            val value = DoubleArray(total)
            val funcOf = IntArray(total)
            val nodeOf = IntArray(total)

            init {
                var slot = 0
                for (j in functionals.indices) {
                    val nodes = functionals[j].nodes
                    for (q in nodes.indices) {
                        value[slot] = nodes[q]
                        funcOf[slot] = j
                        nodeOf[slot] = q
                        slot++
                    }
                }
            }

            /**
             * Слоты, упорядоченные по возрастанию значения.
             *
             * Сортировка УСТОЙЧИВА (`sortedBy` использует устойчивый сорт объектов), поэтому
             * при точном совпадении значений порядок слотов остаётся порядком вставки —
             * это и делает представителя группы тем же элементом, что выбирал `TreeSet`.
             */
            val order: List<Int> by lazy { (0 until total).sortedBy { value[it] } }

            fun emptyIndex(): Array<IntArray> = Array(sizes.size) { IntArray(sizes[it]) }

            /**
             * Разбивает слоты на группы «одинаковых» точек одним проходом по возрастанию.
             *
             * Сравнение ведётся с ПРЕДСТАВИТЕЛЕМ группы, а не с предыдущей точкой: иначе
             * цепочка попарно близких точек могла бы слиться в группу шириной много больше
             * [mergeEps].
             */
            fun clusterByValue(mergeEps: Double): Clusters {
                require(mergeEps >= 0.0 && !mergeEps.isNaN()) {
                    "SupportPoints: допуск слияния должен быть неотрицательным, получено $mergeEps"
                }
                val clusterOf = IntArray(total)
                val representatives = DoubleArray(total)
                var count = 0
                for (slot in order) {
                    val v = value[slot]
                    if (count == 0 || v - representatives[count - 1] > mergeEps) {
                        representatives[count] = v
                        count++
                    }
                    clusterOf[slot] = count - 1
                }
                return Clusters(clusterOf, representatives.copyOf(count))
            }
        }

        /**
         * Результат группировки: номер группы для каждого слота и представитель каждой
         * группы (группы пронумерованы по возрастанию значения).
         */
        private class Clusters(val clusterOf: IntArray, val representatives: DoubleArray)
    }
}
