package solvers.volterra

import java.util.concurrent.atomic.AtomicReferenceArray
import numerics.*
import splines.*
import splines.functionals.*
import splines.metrics.*

/**
 * Ядро `K(t,s)` линейного уравнения Вольтерры вместе с аналитическими частными
 * производными.
 *
 * @param k само ядро `K(t,s)`.
 * @param kT производная `K_t(t,s)`; требуется семействам функционалов
 *        де Бура–Фикса `xi^<1>`, `xi^<2>`, а также редукции уравнения I рода.
 * @param kS производная `K_s(t,s)`; входит в диагональный член второй производной
 *        образа `(V u)''` и нужна семейству `xi^<0>`. В отличие от уравнения
 *        Фредгольма здесь она действительно используется: из-за переменного верхнего
 *        предела дифференцирование затрагивает диагональ `K(t,t)`.
 * @param kTT вторая производная `K_tt(t,s)`; требуется семейству `xi^<0>`.
 *
 * Значения по умолчанию равны нулю и допустимы ТОЛЬКО тогда, когда соответствующая
 * производная действительно тождественно нулевая, либо когда выбранное семейство
 * функционалов её не использует: иначе система будет построена неверно без какой-либо
 * диагностики.
 */
public class KernelV(
    public val k: (Double, Double) -> Double,
    public val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kS: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
)

/**
 * Оператор Вольтерра: (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds, квадратура по [a,t].
 *
 * Из-за ПЕРЕМЕННОГО верхнего предела (в отличие от Фредгольма) фиксированный набор
 * глобальных гауссовых узлов непригоден: область интегрирования зависит от t.
 * Поэтому интеграл считается напрямую (замыканиями) по составному разбиению [a,t]
 * (узлы сетки, попавшие в (a,t), плюс концы a и t).
 *
 * Производная по правилу Лейбница (для xi-функционалов и метрики):
 *   d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds.
 * ГРАНИЧНЫЙ член K(t,t) u(t) — специфика Вольтерра (у Фредгольма его нет).
 */
public class VolterraOperator(public val kernel: KernelV, public val grid: Grid, public val quad: GaussLegendre) {
    /** Левый конец отрезка — нижний предел интегрирования во всех формулах. */
    public val a: Double = grid.a

    /**
     * Допуск включения узла: узел сетки считается строго внутри (a, t), если `x < t - eps`.
     *
     * Значение берётся из ЕДИНОГО ИСТОЧНИКА [Grid.breakpointInclusionEps] (там же —
     * обоснование относительности и оговорка про мелкие отрезки), а не вычисляется
     * здесь повторно: раньше та же формула дублировалась в `SplineSpace`.
     *
     * Читается всеми тремя местами отбора ([subBreakpoints], кэшированный [apply],
     * [integrateRange]) — это не стиль, а ТРЕБОВАНИЕ КОРРЕКТНОСТИ: кэшированный и
     * некэшированный пути обязаны отбирать ОДИНАКОВОЕ число полных ячеек, иначе
     * ломается инвариант [IntegrandCache].
     */
    public val breakpointInclusionEps: Double = grid.breakpointInclusionEps

    /** Составное разбиение [a, t]: внутренние узлы сетки < t, затем сам t. */
    private fun subBreakpoints(t: Double): DoubleArray {
        val bp = grid.breakpoints
        val list = ArrayList<Double>()
        for (x in bp) { if (x < t - breakpointInclusionEps) list.add(x) else break }
        if (list.isEmpty()) list.add(a)
        list.add(t)
        return list.toDoubleArray()
    }

    /** (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds для произвольной u(s). */
    public fun apply(t: Double, u: (Double) -> Double): Double {
        // numerical-core 1.0.0 отвергает нечисловые точки разбиения исключением; контракт
        // оператора — распространять NaN аргумента, как это делает кэшированный путь.
        if (t.isNaN()) return Double.NaN
        if (t <= a) return 0.0
        return quad.integrate(subBreakpoints(t)) { s -> kernel.k(t, s) * u(s) }
    }

    /**
     * Гауссовы узлы ПОЛНЫХ ячеек сетки: `cellNodes[c][q]` — q-й узел составной
     * квадратуры на ячейке `[x_c, x_{c+1}]`.
     *
     * Ключевое наблюдение: при интегрировании по `[a,t]` от `t` зависит ТОЛЬКО последняя,
     * усечённая ячейка `[x_k, t]`; узлы всех полных ячеек одни и те же при любом `t`.
     * Формула здесь дословно повторяет [GaussLegendre.integrate] (`half`, `mid`,
     * `mid + half * refNodes[q]`) на тех же аргументах, поэтому узлы совпадают
     * ПОБИТОВО со значениями, которые вычислила бы сама квадратура.
     *
     * ИНВАРИАНТ (важно): это СНИМОК, материализуемый один раз, тогда как
     * некэшированный путь [apply] читает `grid.breakpoints` при каждом вызове.
     * Поэтому два пути эквивалентны ПРИ УСЛОВИИ неизменности содержимого
     * `grid.breakpoints` после первого обращения к `cellNodes`.
     *
     * Статус защиты этого условия: [Grid.breakpoints] — ГОРЯЧЕЕ поле, поэтому оно
     * СОЗНАТЕЛЬНО не возвращает копию (копирование на каждом из десятков тысяч
     * обращений свело бы на нет сам смысл кэша). Вместо этого действует явно
     * задокументированное соглашение «read-only» (см. KDoc [Grid.breakpoints]), и в проекте
     * нет ни одной записи в этот массив. Холодные же массивы соседних API
     * (`GaussLegendre.refNodesWeights`, `SplineSpace.weights/wInt/gramR`) отдаются копиями.
     *
     * О ПОРЯДКЕ ИНИЦИАЛИЗАЦИИ: выражение использует [refNodes], объявлённый НИЖЕ;
     * корректность обеспечена ИМЕННО ленивостью (к моменту первого обращения
     * конструктор уже завершён). Замена `by lazy` на немедленную инициализацию
     * без переноса [refNodes] выше дала бы `NullPointerException`.
     */
    private val cellNodes: Array<DoubleArray> by lazy {
        val bp = grid.breakpoints
        Array(bp.size - 1) { c ->
            val lo = bp[c]
            val hi = bp[c + 1]
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            DoubleArray(refNodes.size) { q -> mid + half * refNodes[q] }
        }
    }

    /** Эталонные узлы/веса квадратуры на [-1,1]: чтение в горячем цикле без аллокаций. */
    private val refNodes: DoubleArray = quad.refNodesWeights().first
    private val refWeights: DoubleArray = quad.refNodesWeights().second

    /**
     * Кэш значений подынтегральной функции `u(s)` в узлах ПОЛНЫХ ячеек сетки.
     *
     * Назначение — снять квадратичную стоимость применения оператора Вольтерры:
     * без кэша каждое обращение `apply(t, u)` при своём `t` заново вычисляло `u`
     * во всех `8 * (число полных ячеек)` узлах, хотя сами узлы от `t` не зависят.
     *
     * Кэш ЖЁСТКО СВЯЗАН с одной функцией `u` (она хранится в поле и другой быть не может),
     * поэтому подмена значений между разными подынтегральными функциями невозможна
     * конструктивно. Размер ограничен сеткой: `n * nodesPerSub` чисел (при n=64 — 512).
     *
     * Кэш ТАКЖЕ СВЯЗАН С ОПЕРАТОРОМ, создавшим его (см. [IntegrandCache.owner] и
     * проверку в [apply]). Проверка нужна именно потому, что САМ ТИП владельца не
     * различает: в Kotlin у `inner class` нет типа, параметризованного внешним
     * ЭКЗЕМПЛЯРОМ, поэтому выражение `op2.apply(t, op1.integrandCache(u))`
     * компилируется без ошибок и без проверки дало бы молча неверные числа.
     *
     * Кэшировать допустимо только ЧИСТУЮ `u` (детерминированную, без побочных эффектов).
     * Фактические потребители — ВСЕ вызовы [VolterraSecondKindSolver.applyL] (базисные `omega_i`
     * и их образы в `matrixM`/`matrixM2`, сплайны `evalSpline` в `sloan`/`kulkarni*`,
     * итеранты `kulkarniQuasi`/`combinedNystrom`, образы Nyström), все они чистые.
     * ВНИМАНИЕ: `vectorD` кэш НЕ использует — он идёт через старую перегрузку
     * `apply(t, u)` и `applyDeriv`; `applyLDeriv`/`applyLDeriv2` тоже не кэшируются
     * (у них другие ядра `kT`/`kTT`).
     *
     * ПОТОКОБЕЗОПАСНОСТЬ: значения ячейки публикуются целым массивом через
     * [AtomicReferenceArray], что даёт корректную публикацию (happens-before) без
     * блокировок. Гонка двух потоков на одной ячейке безвредна: `u` чиста, значит оба
     * вычислят побитово одинаковые числа, а победитель CAS определяет, чей массив увидят
     * остальные. Точки сериализации в горячем цикле нет — только volatile-чтение на ячейку.
     */
    public inner class IntegrandCache internal constructor(internal val u: (Double) -> Double) {
        /**
         * Экземпляр оператора, создавший этот кэш — единственный, чьи узлы соответствуют
         * хранимым значениям.
         *
         * Почему явное свойство, а не неявная ссылка `inner class`: неявная ссылка
         * на внешний экземпляр доступна только ИЗНУТРИ тела `IntegrandCache`
         * (`this@VolterraOperator`) и не читается снаружи, то есть из [apply], где и
         * нужна проверка. Поэтому ссылка зафиксирована в поле при создании.
         * Стоимость — одна ссылка на кэш (не на вызов и не на узел).
         */
        internal val owner: VolterraOperator = this@VolterraOperator

        private val cells = AtomicReferenceArray<DoubleArray>(cellNodes.size)

        /** Значения `u` в узлах полной ячейки `c`; вычисляются при первом обращении. */
        internal fun values(c: Int): DoubleArray {
            cells.get(c)?.let { return it }
            val nodes = cellNodes[c]
            val computed = DoubleArray(nodes.size) { q -> u(nodes[q]) }
            cells.compareAndSet(c, null, computed)
            return cells.get(c) ?: computed
        }
    }

    /** Создаёт кэш узловых значений для КОНКРЕТНОЙ подынтегральной функции. */
    public fun integrandCache(u: (Double) -> Double): IntegrandCache = IntegrandCache(u)

    /**
     * То же, что [apply], но значения `u` в узлах полных ячеек берутся из [cache].
     *
     * Арифметика повторена ДОСЛОВНО за [subBreakpoints] + [GaussLegendre.integrate]:
     * тот же отбор точек разбиения, тот же порядок обхода (ячейки слева направо, внутри
     * ячейки — узлы по возрастанию индекса), те же `half`/`mid`, то же простое
     * накопление `sum += half * w_q * (K(t,s_q) * u(s_q))` без выноса множителей.
     * Отличие ровно одно: `u(s_q)` на полных ячейках не вычисляется повторно.
     *
     * ТРЕБОВАНИЕ К ВЛАДЕЛЬЦУ: [cache] обязан быть создан ЭТИМ же экземпляром
     * оператора. Проверка выполняется ОДИН раз на входе (сравнение ссылок, вне
     * любого цикла; стоимость ничтожна на фоне `nodesPerSub * n` вызовов `kernel.k`)
     * и ничего не вычисляет, поэтому на числа не влияет. Без неё чужой кэш дал бы
     * МОЛЧА НЕВЕРНЫЕ ЧИСЛА: значения `u` брались бы в узлах СВОЕЙ сетки, а ядро
     * вычислялось бы в узлах сетки ЭТОГО оператора (или вовсе
     * `IndexOutOfBoundsException` при более грубой сетке владельца).
     *
     * @throws IllegalArgumentException если [cache] создан другим экземпляром оператора.
     */
    public fun apply(t: Double, cache: IntegrandCache): Double {
        require(cache.owner === this) {
            "Кэш узловых значений передан ДРУГОМУ экземпляру VolterraOperator, чем тот, " +
                "который его создал. Кэш хранит значения u(s) в гауссовых узлах сетки СВОЕГО " +
                "владельца, а ядро здесь вычислялось бы в узлах сетки этого оператора: " +
                "узлы сеток в общем случае НЕ СОВПАДАЮТ, и результат был бы молча неверным " +
                "(либо возникло бы IndexOutOfBoundsException при более грубой сетке владельца). " +
                "Кэш создавайте тем же оператором, на котором его применяете: " +
                "op.apply(t, op.integrandCache(u))."
        }
        if (t <= a) return 0.0
        val bp = grid.breakpoints
        // Число ведущих узлов сетки, попавших в разбиение (см. subBreakpoints).
        var included = 0
        while (included < bp.size && bp[included] < t - breakpointInclusionEps) included++
        var sum = 0.0
        // Полные ячейки [bp[c], bp[c+1]] — узлы и значения u берутся из кэша.
        for (c in 0 until included - 1) {
            val lo = bp[c]
            val hi = bp[c + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val nodes = cellNodes[c]
            val values = cache.values(c)
            for (q in refNodes.indices) {
                sum += half * refWeights[q] * (kernel.k(t, nodes[q]) * values[q])
            }
        }
        // Усечённая ячейка [x_k, t] (или [a, t], если узлов сетки левее t нет).
        val lo = if (included == 0) a else bp[included - 1]
        // Условие записано как ОТРИЦАНИЕ `hi <= lo` из [GaussLegendre.integrate] дословно,
        // а не как `t > lo`: для конечных `t` это тождественно, но при `t = NaN`
        // оба сравнения ложны, так что `t > lo` дало бы ровно 0.0, а некэшированный
        // путь (где `hi <= lo` тоже ложно и счёт продолжается) — NaN. Вариант с
        // `require(!t.isNaN())` отвергнут: он потребовал бы правки СТАРОГО `apply(t, u)`,
        // то есть смены поведения публичного API вне скоупа (его зовут также решатель
        // Урысона, `problems` и тесты). На конечных `t` числа не меняются: ветвь та же.
        if (!(t <= lo)) {
            val half = 0.5 * (t - lo)
            val mid = 0.5 * (t + lo)
            for (q in refNodes.indices) {
                val s = mid + half * refNodes[q]
                sum += half * refWeights[q] * (kernel.k(t, s) * cache.u(s))
            }
        }
        return sum
    }

    /**
     * Интеграл int_lo^hi g(s) ds по составному разбиению [lo,hi], делённому узлами сетки.
     * Используется для весов Nyström с ограничением на компактный носитель omega_j
     * ([x_j,x_{j+3}]) -> не более трёх подынтервалов, что снимает O(n)-стоимость на точку.
     */
    public fun integrateRange(lo: Double, hi: Double, g: (Double) -> Double): Double {
        if (hi <= lo) return 0.0
        val list = ArrayList<Double>()
        list.add(lo)
        for (x in grid.breakpoints) if (x > lo + breakpointInclusionEps && x < hi - breakpointInclusionEps) list.add(x)
        list.add(hi)
        return quad.integrate(list.toDoubleArray(), g)
    }

    /**
     * d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds (Лейбниц).
     * ВАЖНО: граничный член K(t,t)u(t) остаётся и при t=a (интеграл по [a,a] нулевой).
     * Этот член критичен для сведения V1->V2 на левом конце (g(a)=f'(a)/K(a,a)=u*(a)).
     */
    public fun applyDeriv(t: Double, u: (Double) -> Double): Double {
        if (t < a) return 0.0
        val boundary = kernel.k(t, t) * u(t)
        val integral = if (t.isNaN()) Double.NaN else if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kT(t, s) * u(s) }
        return boundary + integral
    }

    /**
     * Вторая производная образа:
     *
     *     (V u)''(t) = [2 K_t(t,t) + K_s(t,t)] u(t) + K(t,t) u'(t)
     *                  + \int_a^t K_tt(t,s) u(s) ds.
     *
     * Формула получается повторным применением правила Лейбница к [applyDeriv]:
     * дифференцирование граничного члена `K(t,t) u(t)` даёт `(K_t + K_s)(t,t) u(t)`
     * и `K(t,t) u'(t)`, а дифференцирование интеграла — ещё один член `K_t(t,t) u(t)`
     * и интеграл от `K_tt`.
     *
     * ВАЖНО: член `K(t,t) u'(t)` обязателен при `K(t,t) != 0`, поэтому кроме самой
     * функции передаётся и её первая производная. Диагональный член сохраняется
     * и при `t = a`, где интеграл по `[a,a]` обращается в ноль.
     *
     * @param u сама функция.
     * @param uD её первая производная.
     */
    public fun applyDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double {
        if (t < a) return 0.0
        val kd = kernel.k(t, t)
        val diag = 2.0 * kernel.kT(t, t) + kernel.kS(t, t)
        val boundary = diag * u(t) + kd * uD(t)
        val integral = if (t.isNaN()) Double.NaN else if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kTT(t, s) * u(s) }
        return boundary + integral
    }
}
