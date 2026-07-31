package solvers.volterra

import java.util.concurrent.atomic.AtomicReferenceArray
import kotlin.math.abs
import numerics.*
import numerics.functionals.*
import solvers.core.ImageTriple
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_TOLERANCE
import solvers.core.SecondKindSolverCore

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
class KernelV(
    val k: (Double, Double) -> Double,
    val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    val kS: (Double, Double) -> Double = { _, _ -> 0.0 },
    val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
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
class VolterraOperator(val kernel: KernelV, val grid: Grid, val quad: GaussLegendre) {
    /** Левый конец отрезка — нижний предел интегрирования во всех формулах. */
    val a = grid.a

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
    val breakpointInclusionEps: Double = grid.breakpointInclusionEps

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
    fun apply(t: Double, u: (Double) -> Double): Double {
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
    inner class IntegrandCache internal constructor(internal val u: (Double) -> Double) {
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
    fun integrandCache(u: (Double) -> Double): IntegrandCache = IntegrandCache(u)

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
    fun apply(t: Double, cache: IntegrandCache): Double {
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
    fun integrateRange(lo: Double, hi: Double, g: (Double) -> Double): Double {
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
    fun applyDeriv(t: Double, u: (Double) -> Double): Double {
        if (t < a) return 0.0
        val boundary = kernel.k(t, t) * u(t)
        val integral = if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kT(t, s) * u(s) }
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
    fun applyDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double {
        if (t < a) return 0.0
        val kd = kernel.k(t, t)
        val diag = 2.0 * kernel.kT(t, t) + kernel.kS(t, t)
        val boundary = diag * u(t) + kd * uD(t)
        val integral = if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kTT(t, s) * u(s) }
        return boundary + integral
    }
}

/**
 * Линейный решатель уравнения Вольтерры II рода u - L u = f, L = c_L * \mathcal V,
 * где (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds (ПЕРЕМЕННЫЙ верхний предел).
 *
 * c_L = 1 для уравнения II рода; при редукции I->II рода (см. [VolterraFirstKindSolver])
 * тоже c_L = 1, но с другим (редуцированным) ядром и правой частью.
 * Правая часть f и её производные задаются явно (fEff/fEffDeriv/fEffDeriv2),
 * чтобы решатель переиспользовался и для задач I рода.
 *
 * Матрицы дискретной задачи:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * @param throwOnDivergence поведение ИТЕРАЦИОННЫХ схем ([kulkarni] для
 *        квазиинтерполянтов, [combinedNystrom]) при недостижении сходимости:
 *        `true` (по умолчанию) — исключение, `false` — результат с
 *        `converged = false` и достигнутой невязкой в [numerics.SolutionFunc.residual].
 *        На прямые схемы не влияет.
 */
class VolterraSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    val op: VolterraOperator,
    cL: Double,
    fEff: (Double) -> Double,
    fEffDeriv: (Double) -> Double,
    fEffDeriv2: (Double) -> Double = { 0.0 },
    throwOnDivergence: Boolean = true,
) : SecondKindSolverCore<(Double) -> Double>(
    basis, funcs, cL, fEff, fEffDeriv, fEffDeriv2, throwOnDivergence,
) {
    // Числовые параметры итерационных схем (KULKARNI_QUASI_*, COMBINED_NYSTROM_*)
    // живут в [solvers.core.SecondKindDefaults]: у Фредгольма и Вольтерры они
    // совпадают, и расхождение при подборе нового значения было бы молчаливым.
    // Общая часть схем (`base`, `sloan`, `kulkarni`, сборка M/M2/g/d) — в
    // [SecondKindSolverCore]; здесь остаётся специфика Вольтерры.
    //
    // ВНИМАНИЕ (отличие от Фредгольма): у оператора Вольтерра область интегрирования
    // [a,t] зависит от t, поэтому предвычисление на фиксированных узлах невозможно.
    // Все применения L = c_L \mathcal V выражаются через замыкания op.apply / op.applyDeriv,
    // а «подготовленный операнд» — это сама функция, а не таблица её значений.

    /**
     * L g(t) = c_L (\mathcal V g)(t) и её производная (Лейбниц).
     *
     * Возвращаемое замыкание владеет СВОИМ кэшем узловых значений `g` на полных ячейках
     * (см. [VolterraOperator.IntegrandCache]): время жизни кэша совпадает со временем
     * жизни замыкания, а привязка к `g` фиксируется при создании, поэтому отдать значения
     * одной функции другой невозможно. Арифметика не меняется (см. [VolterraOperator.apply]).
     *
     * Метод СОЗНАТЕЛЬНО не переехал в общее ядро: кэш привязан к конкретному
     * [VolterraOperator] проверкой `require(cache.owner === this)`, и обобщённый тип
     * кэша ослабил бы эту проверку до runtime-каста.
     */
    private fun applyL(g: (Double) -> Double): (Double) -> Double {
        val cache = op.integrandCache(g)
        return { t -> cL * op.apply(t, cache) }
    }
    private fun applyLDeriv(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.applyDeriv(t, g) }

    /**
     * (L g)''(t) = c_L (\mathcal V g)''(t) по (V2''): требует g И g' (член K(t,t) g'(t)).
     * gD — первая производная самого операнда g.
     */
    private fun applyLDeriv2(g: (Double) -> Double, gD: (Double) -> Double): (Double) -> Double =
        { t -> cL * op.applyDeriv2(t, g, gD) }

    // --- Реализация точек расширения [SecondKindSolverCore] --------------------

    override val equationName: String get() = "Вольтерра"

    override val kulkarniQuasiHint: String
        get() = "Для квазиинтерполянтов (mu, lambda) нет свойства P^2 = P, поэтому " +
            "редукция Кулкарни неприменима и используется простая итерация"

    /**
     * Контрольные точки критерия останова: `4n+1` равноотстоящих точек отрезка.
     *
     * Глобальных узлов у оператора Вольтерра нет (они зависят от `t`), поэтому
     * нужна отдельная выборка. Она участвует ТОЛЬКО в критерии останова,
     * в самой итерации не используется.
     *
     * Поле ленивое, а не эагерное, по двум причинам. Во-первых, выборка нужна
     * только двум итерационным схемам из восьми. Во-вторых и главное: она читает
     * `n` и `grid` из базового класса через переопределённое свойство, а такое свойство
     * в принципе может быть прочитано до завершения конструктора наследника;
     * `by lazy` гарантирует, что вычисление произойдёт при ПЕРВОМ ОБРАЩЕНИИ из
     * метода, то есть гарантированно после того, как `n` уже инициализирован.
     * Эагерное поле здесь было бы корректным только случайно.
     */
    override val checkPoints: DoubleArray by lazy {
        DoubleArray(4 * n + 1) { grid.a + (grid.b - grid.a) * it / (4 * n) }
    }

    /** Предвычисление невозможно: операнд — сама функция. */
    override fun prepare(u: (Double) -> Double): (Double) -> Double = u

    override fun image(o: (Double) -> Double): (Double) -> Double = applyL(o)

    override fun imageDeriv(o: (Double) -> Double): (Double) -> Double = applyLDeriv(o)

    /** Член Лейбница `K(t,t) u'(t)` делает `uD` ОБЯЗАТЕЛЬНЫМ аргументом. */
    override fun imageDeriv2(o: (Double) -> Double, uD: (Double) -> Double): (Double) -> Double =
        applyLDeriv2(o, uD)

    override fun applyOperator(t: Double, u: (Double) -> Double): Double = op.apply(t, u)

    override fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double = op.applyDeriv(t, u)

    override fun applyOperatorDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double =
        op.applyDeriv2(t, u, uD)

    override fun omegaImages(i: Int): ImageTriple {
        val idx = i - 2
        val omega = { s: Double -> basis.omega(idx, s) }
        val omegaD = { s: Double -> basis.omegaDeriv(idx, s) }
        // Вторая производная образа требует и omega_i, и omega_i' (член K(t,t) omega_i').
        return ImageTriple(applyL(omega), applyLDeriv(omega), applyLDeriv2(omega, omegaD))
    }

    /**
     * Образ `L omega_i` строится ЗАНОВО, без переиспользования столбца из [matrixM].
     *
     * Это не избыточность, а сознательное решение: каждое замыкание [applyL] несёт
     * СВОЙ кэш подынтегральной функции, и передача готового образа извне изменила бы
     * число обращений к ядру и, возможно, младшие биты результата.
     */
    override fun doubleOmegaImages(i: Int): ImageTriple {
        val idx = i - 2
        val omega = { s: Double -> basis.omega(idx, s) }
        val image = applyL(omega)
        val imageDeriv = applyLDeriv(omega)
        // Вторая производная требует сам образ L omega_i и его производную.
        return ImageTriple(applyL(image), applyLDeriv(image), applyLDeriv2(image, imageDeriv))
    }

    // --- Nyström: сплайн-квадратура с зависящими от t весами --------------------

    /**
     * Опорные данные Nyström для Вольтерра: точки {eta_r} (по возрастанию t),
     * ValueFunctional-ы семейства и карта точка->индекс. В отличие от Фредгольма
     * веса W_j(t)=int_a^t omega_j зависят от t, поэтому агрегированные веса b_r(t)
     * вычисляются на лету (nystromB). Семейство xi (де Бура--Фикса) НЕ поддерживается:
     * его функционалы используют производную и не сводятся к линейной комбинации значений.
     */
    private class NystromSupport(
        val pts: DoubleArray,
        val vfs: Array<ValueFunctional>,
        val idx: HashMap<Double, Int>,
    )

    private fun nystromSupport(): NystromSupport {
        require(!funcs.usesDerivative) {
            "Nyström для семейства '${funcs.name}' не реализован: функционалы " +
                "де Бура--Фикса (xi) используют производную и не сводятся к значениям."
        }
        val vfs = Array(dim) { k ->
            funcs.chi(k - 2) as? ValueFunctional
                ?: error("Nyström: функционал '${funcs.name}' (j=${k - 2}) не является ValueFunctional.")
        }
        val ptSet = sortedSetOf<Double>()
        for (vf in vfs) for (s in vf.nodes) ptSet.add(s)
        val pts = ptSet.toDoubleArray()
        val idx = HashMap<Double, Int>(pts.size * 2)
        for (i in pts.indices) idx[pts[i]] = i
        return NystromSupport(pts, vfs, idx)
    }

    /**
     * Агрегированные веса b_r(t):
     * b_r(t) = sum_j sum_{q: s_{j,q}=eta_r} c_{j,q} W_j(t), W_j(t)=int_a^t omega_j.
     */
    private fun nystromB(sup: NystromSupport, t: Double): DoubleArray {
        val b = DoubleArray(sup.pts.size)
        for (k in 0 until dim) {
            val j = k - 2
            val lo = grid.x(j)                 // левый конец носителя omega_j (>= a)
            val hi = minOf(grid.x(j + 3), t)   // правый конец, усечённый верхним пределом t
            if (hi <= lo) continue             // носитель правее t -> W_j(t)=0 (причинность)
            val w = op.integrateRange(lo, hi) { s -> basis.omega(j, s) }
            val vf = sup.vfs[k]
            for (q in vf.nodes.indices) b[sup.idx.getValue(vf.nodes[q])] += vf.coeffs[q] * w
        }
        return b
    }

    /** u^N_h(t) = f(t) + cL sum_r b_r(t) K(t, eta_r) u_hat_r. */
    private fun nystromEval(sup: NystromSupport, uHat: DoubleArray, t: Double): Double {
        val b = nystromB(sup, t)
        var acc = 0.0
        for (r in sup.pts.indices) acc += b[r] * op.kernel.k(t, sup.pts[r]) * uHat[r]
        return fEff(t) + cL * acc
    }

    /** Решает (I - A^{N,V}) u_hat = f_hat: A^{N,V}_{rho,r}=cL b_r(eta_rho) K(eta_rho,eta_r) (2.3). */
    private fun nystromSolve(sup: NystromSupport): DoubleArray {
        val p = sup.pts.size
        val a = LinearAlgebra.zeros(p, p)
        for (rho in 0 until p) {
            val b = nystromB(sup, sup.pts[rho]) // t-зависимые веса при t=eta_rho
            for (r in 0 until p) a[rho][r] = -cL * b[r] * op.kernel.k(sup.pts[rho], sup.pts[r])
            a[rho][rho] += 1.0
        }
        return LinearAlgebra.solve(a, DoubleArray(p) { fEff(sup.pts[it]) })
    }

    /**
     * КЛАССИЧЕСКИЙ сплайн-Nyström для уравнения Вольтерры: квадратура с
     * t-зависимыми весами W_j(t)=int_a^t omega_j. Приводит к линейной системе
     * (I - A^{N,V}) u_hat = f_hat по значениям решения в опорных точках {eta_r}.
     * Приближение вне сплайнового пространства. Не поддерживает семейство xi.
     *
     * О СТРУКТУРЕ МАТРИЦЫ: ранее здесь утверждалось, что при упорядочении точек по
     * возрастанию матрица (блочно-)нижнетреугольна «по причинности». Это утверждение
     * УДАЛЕНО как НЕПОДТВЕРЖДЁННОЕ: b_r(eta_rho) агрегирует коэффициенты функционалов,
     * чьи опорные точки могут лежать правее eta_rho (носители omega_j перекрываются),
     * поэтому в общем случае верхние элементы не нулевые. Код всё равно решает систему
     * общим LU-разложением и на треугольность не полагается.
     *
     * ВАЖНО о порядке: это «голая» квадратура, сама по себе НЕ повышающая порядок;
     * см. [combinedNystrom]. Для уравнения Вольтерры теоретических оценок суперсходимости
     * в известной литературе нет ни для одного из вариантов (переменный верхний предел
     * даёт t-зависимые веса и усечение последней ячейки — требуется отдельный анализ).
     * Любые наблюдаемые порядки здесь — численное наблюдение, а не доказанный результат.
     */
    fun nystrom(): SolutionFunc {
        val sup = nystromSupport()
        val uHat = nystromSolve(sup)
        return SolutionFunc(eval = { t -> nystromEval(sup, uHat, t) })
    }

    /**
     * Итерированный Nyström: u_hat^N_h(t)=f(t)+(L u^N_h)(t) с ТОЧНЫМ оператором
     * Вольтерра L (замыкание applyL, как в sloan()). Одно интегрирование найденного
     * u^N_h, новой системы не требуется (аналог итерации Слоана).
     */
    fun iteratedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        val uHat = nystromSolve(sup)
        val uN = applyL { s -> nystromEval(sup, uHat, s) }
        return SolutionFunc(eval = { t -> fEff(t) + uN(t) })
    }

    /**
     * КОМБИНИРОВАННЫЙ оператор Nyström для уравнения Вольтерры:
     * u^N_h = f + L_n u^N_h, где L_n = P_chi L + (I - P_chi) L^N_h.
     *
     * На образе проектора действует ТОЧНЫЙ оператор, на дополнении — квадратура с
     * t-зависимыми весами. Отличие от [nystrom]: там решается u = f + L^N_h u.
     *
     * СТАТУС ИСТОЧНИКА: конструкция L_n взята из теории для уравнения Фредгольма
     * (см. [solvers.fredholm.FredholmSecondKindSolver.combinedNystrom]); для уравнения Вольтерры это
     * АДАПТАЦИЯ: доказательства суперсходимости в известной литературе НЕТ. Поведение
     * следует трактовать как численное наблюдение.
     *
     * @throws IllegalStateException если итерация не сошлась и [throwOnDivergence] равно `true`.
     */
    fun combinedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        // Критерий останова мерится на ТОМ ЖЕ множестве [checkPoints], что и в `kulkarniQuasi`.
        // Локальный пересчёт той же формулой был бы вторым критерием в одном классе:
        // при правке одного из них две схемы молча разъехались бы.
        val checkPoints = this.checkPoints
        var uAtCheck = DoubleArray(checkPoints.size) { uFun(checkPoints[it]) }
        var converged = false
        var performedIterations = 0
        var lastDiff = Double.NaN
        for (iter in 0 until COMBINED_NYSTROM_MAX_ITERATIONS) {
            val currentFun = uFun
            val currentAtPoints = DoubleArray(sup.pts.size) { currentFun(sup.pts[it]) }
            // Точный оператор L и его проекция P_chi(L u).
            val exactImage = applyL(currentFun)
            val projectedExact = funcs.projectorCoeffs(exactImage)
            // Квадратурный оператор L^N_h с t-зависимыми весами и его проекция.
            val quadratureImage = { t: Double ->
                val b = nystromB(sup, t)
                var acc = 0.0
                for (r in sup.pts.indices) acc += b[r] * op.kernel.k(t, sup.pts[r]) * currentAtPoints[r]
                cL * acc
            }
            val projectedQuadrature = funcs.projectorCoeffs(quadratureImage)
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(projectedExact, t) +
                    quadratureImage(t) - basis.evalSpline(projectedQuadrature, t)
            }
            val nextAtCheck = DoubleArray(checkPoints.size) { nextFun(checkPoints[it]) }
            var diff = 0.0
            for (k in nextAtCheck.indices) diff = maxOf(diff, abs(nextAtCheck[k] - uAtCheck[k]))
            uFun = nextFun
            uAtCheck = nextAtCheck
            performedIterations = iter + 1
            lastDiff = diff
            if (diff < COMBINED_NYSTROM_TOLERANCE) { converged = true; break }
        }
        reportConvergence(
            converged = converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Комбинированный Nyström (Вольтерра)",
            iterations = performedIterations,
            maxIterations = COMBINED_NYSTROM_MAX_ITERATIONS,
            residual = lastDiff,
            tolerance = COMBINED_NYSTROM_TOLERANCE,
            hint = "Для сходимости простой итерации требуется ||L_n|| < 1",
        )
        val resultFun = uFun
        return SolutionFunc(
            eval = { t -> resultFun(t) },
            converged = converged,
            iterations = performedIterations,
            residual = lastDiff,
        )
    }

    /**
     * Итерированный комбинированный Nyström: \hat u^N_h = f + L u^N_h с точным L.
     * Признак сходимости наследуется от [combinedNystrom].
     */
    fun iteratedCombinedNystrom(): SolutionFunc {
        val combined = combinedNystrom()
        val image = applyL { s -> combined.eval(s) }
        return SolutionFunc(
            eval = { t -> fEff(t) + image(t) },
            converged = combined.converged,
            iterations = combined.iterations,
            residual = combined.residual,
        )
    }
}

/**
 * Решатель уравнения Вольтерры ПЕРВОГО рода `(V u)(t) = \int_a^t K(t,s) u(s) ds = f(t)`
 * сведением к уравнению второго рода дифференцированием.
 *
 * Математическая идея. В отличие от уравнения Фредгольма первого рода (некорректного,
 * требующего регуляризации), задача Вольтерры при `K(t,t) != 0` КОРРЕКТНА. Дифференцируя
 * исходное уравнение по `t` по правилу Лейбница, получаем
 *
 *     K(t,t) u(t) + \int_a^t K_t(t,s) u(s) ds = f'(t),
 *
 * и после деления на `K(t,t)` приходим к уравнению второго рода
 *
 *     u(t) - (W u)(t) = g(t),   (W u)(t) = \int_a^t [-K_t(t,s)/K(t,t)] u(s) ds,
 *     g(t) = f'(t)/K(t,t),
 *
 * которое решается обычной схемой второго рода с `c_L = 1`. Источник метода указан
 * в `docs/REFERENCES.md` (раздел «Уравнения первого рода»).
 *
 * Случай `m = 1` (однократное дифференцирование) применим только при `K(t,t) != 0`.
 *
 * ДИАГНОСТИКА ВЫРОЖДЕНИЯ ДИАГОНАЛИ — два независимых рубежа:
 *
 *  1. *Предварительная проверка при создании* — по всем точкам, где деление реально
 *     выполняется на регулярной основе: узлы сетки, середины интервалов, гауссовы узлы
 *     составной квадратуры редуцированного оператора и точки шаблона конечной разности
 *     вокруг каждой из них. Даёт раннюю и дешёвую диагностику до начала счёта.
 *  2. *Защита в самой точке деления* ([safeDiagonal]) — срабатывает при ЛЮБОМ вычислении,
 *     в том числе в произвольной точке `t`, которую запросил пользователь у готового
 *     решения. Именно она даёт гарантию: молчаливого `NaN`/`Inf` не возникает нигде.
 *
 * Второй рубеж необходим, потому что множество точек деления не является конечным и
 * предвычислимым: оператор Вольтерры интегрирует по `[a,t]` с УСЕЧЁННОЙ последней
 * ячейкой, поэтому гауссовы узлы зависят от `t`, а итерация Слоана вычисляет `g(t)`
 * в любой запрошенной точке. Ранее в такой ситуации возвращался `NaN` без какого-либо
 * сигнала: например, при диагонали, положительной в узлах и серединах, но нулевой
 * в промежуточной точке, `base()` давала правдоподобное `E_h ~ 1.3e-5`, а `sloan()`
 * молча возвращала `NaN`.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство аппроксимационных функционалов.
 * @param kernel ядро ИСХОДНОГО уравнения первого рода.
 * @param rhsDeriv производная правой части `f'(t)` исходного уравнения.
 * @param smoothPart гладкая часть решения, известная аналитически; из-под конечной
 *        разности она выносится, чтобы не усиливать шум (см. пояснение к [gEffDeriv]).
 * @param smoothPartDeriv производная гладкой части.
 * @param throwOnDivergence политика обработки недостижения сходимости итерационными
 *        схемами внутреннего решателя; см. [VolterraSecondKindSolver.throwOnDivergence].
 * @throws IllegalArgumentException если `K(t,t)` близко к нулю в контрольных точках;
 *         если длина отрезка недостаточна для шаблона конечной разности; либо если
 *         выбрано семейство функционалов, требующее второй производной.
 * @throws IllegalStateException если `K(t,t)` обращается в ноль в точке деления,
 *         обнаруженной уже во время счёта (см. [safeDiagonal]).
 */
class VolterraFirstKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    kernel: KernelV,
    rhsDeriv: (Double) -> Double,
    smoothPart: (Double) -> Double,
    smoothPartDeriv: (Double) -> Double,
    val throwOnDivergence: Boolean = true,
) {
    private companion object {
        /**
         * Порог, ниже которого диагональ ядра `|K(t,t)|` считается нулевой и редукция
         * первого рода ко второму объявляется неприменимой. Значение выбрано много
         * больше машинного эпсилона, но много меньше типичных значений ядра: деление
         * на меньшую величину даёт неконтролируемое усиление погрешности.
         */
        const val KERNEL_DIAGONAL_TOLERANCE = 1e-12

        /**
         * Шаг конечной разности для численного дифференцирования.
         *
         * Для формулы ЧЕТВЁРТОГО порядка оптимум по сумме ошибки аппроксимации `O(h^4)`
         * и ошибки округления `O(eps/h)` достигается при `h ~ eps^{1/5} ~ 1e-3`.
         * При меньшем шаге (например, `1e-6`, оптимальном для второго порядка) начинает
         * доминировать ошибка округления и шум квадратуры.
         *
         * ШАГ ОСТАВЛЕН АБСОЛЮТНЫМ ОСОЗНАННО. Относительный шаг (доля от `b - a`) снял бы
         * ограничение на длину отрезка, но одновременно изменил бы численные результаты
         * на ВСЕХ существующих задачах, включая V1: величина шага входит в ошибку
         * аппроксимации `O(h^4)` и в ошибку округления `O(eps/h)`, поэтому смена шага
         * сдвигает `E_h` в последних значащих цифрах. Такая замена — обоснованное
         * изменение алгоритма, требующее осознанной пересъёмки эталона, и она не входит
         * в задачу «добавить недостающую диагностику». Вместо этого отрезки, на которых
         * шаблон не помещается, ЯВНО ЗАПРЕЩЕНЫ (см. [MIN_INTERVAL_STENCIL_STEPS]).
         */
        const val FINITE_DIFFERENCE_STEP = 1e-3

        /**
         * Минимальная длина отрезка `b - a`, выраженная в шагах [FINITE_DIFFERENCE_STEP].
         *
         * Оценка худшего случая по ветвям [deriv4]. Односторонняя ветвь выбирается для
         * точек, отстоящих от конца меньше чем на `2h`, а её шаблон тянется на `4h` в
         * противоположную сторону: суммарный охват достигает `2h + 4h = 6h`. Поэтому
         * при `b - a >= 6h` шаблон гарантированно остаётся внутри `[a,b]` при любом `t`,
         * а при меньшей длине — выходит за пределы, где ядро и оператор доопределены
         * нулём, что молча исказило бы производную.
         */
        const val MIN_INTERVAL_STENCIL_STEPS = 6

        /**
         * Относительный допуск при сравнении точки с концами отрезка: защищает выбор
         * ветви конечной разности от ошибок округления координат.
         */
        const val BOUNDARY_RELATIVE_TOLERANCE = 1e-9

        /** Порядок квадратуры для редуцированного оператора. */
        const val REDUCED_OPERATOR_QUADRATURE_ORDER = 8
    }

    private val grid = basis.grid
    private val quad = GaussLegendre(REDUCED_OPERATOR_QUADRATURE_ORDER)

    /** Ядро исходного уравнения I рода (нужно и в методах, не только в инициализаторах). */
    private val sourceKernel = kernel

    /**
     * Диагональ ядра `K(t,t)` С ПРОВЕРКОЙ — знаменатель редукции I рода ко II.
     *
     * ВСЕ деления на диагональ выполняются через эту функцию, поэтому вырождение не может
     * пройти незамеченным ни в одной точке — включая те, что невозможно перечислить
     * заранее (гауссовы узлы усечённой ячейки `[x_k, t]` и произвольные точки `t`,
     * запрошенные у готового решения).
     *
     * @throws IllegalStateException если `|K(t,t)|` ниже [KERNEL_DIAGONAL_TOLERANCE].
     */
    private fun safeDiagonal(t: Double): Double {
        val diagonal = sourceKernel.k(t, t)
        check(abs(diagonal) >= KERNEL_DIAGONAL_TOLERANCE) {
            "Решатель уравнения Вольтерры I рода требует K(t,t) != 0 (случай m=1): " +
                "в точке деления t=$t получено K(t,t)=$diagonal " +
                "(|K(t,t)|=${abs(diagonal)} < порога $KERNEL_DIAGONAL_TOLERANCE). " +
                "Эта точка не совпадает ни с узлом сетки, ни с серединой интервала, " +
                "поэтому предварительная проверка её не охватила."
        }
        return diagonal
    }

    /** Диагональ ядра `K(t,t)` — знаменатель редукции первого рода ко второму. */
    private val kernelDiagonal = { t: Double -> safeDiagonal(t) }

    init {
        // Отрезок обязан вмещать шаблон конечной разности: шаг абсолютный, а значит,
        // на коротком отрезке точки t ± k*h вышли бы за [a,b], где ядро и оператор
        // доопределены нулём — производная была бы искажена молча.
        val intervalLength = grid.b - grid.a
        val requiredLength = MIN_INTERVAL_STENCIL_STEPS * FINITE_DIFFERENCE_STEP
        require(intervalLength >= requiredLength) {
            "Решатель уравнения Вольтерры I рода неприменим на слишком коротком отрезке: " +
                "b - a = $intervalLength, а шаблон конечной разности четвёртого порядка требует " +
                "не менее $MIN_INTERVAL_STENCIL_STEPS шагов по $FINITE_DIFFERENCE_STEP, то есть " +
                "b - a >= $requiredLength. Шаг разности абсолютен и не масштабируется с длиной " +
                "отрезка, иначе точки шаблона выйдут за пределы области определения."
        }
        // Редукция делит на K(t,t), поэтому обращение диагонали в ноль недопустимо.
        // Проверяем ВСЕ точки, где деление выполняется на регулярной основе. Ранее
        // проверялись только узлы и середины, хотя главный потребитель деления — это
        // квадратура редуцированного оператора и шаблон конечной разности.
        for (t in diagonalCheckPoints()) {
            val diagonal = kernel.k(t, t)
            require(abs(diagonal) >= KERNEL_DIAGONAL_TOLERANCE) {
                "Решатель уравнения Вольтерры I рода требует K(t,t) != 0 (случай m=1); " +
                    "K(t,t)=$diagonal при t=$t (|K(t,t)|=${abs(diagonal)}) слишком мало"
            }
        }
        // Семейство xi^<0> требует ВТОРОЙ производной образа (Wu)'' и правой части g''.
        // После редукции ядро K_W само задано через численное дифференцирование, а его
        // производные K_W_s и K_W_tt аналитически недоступны: их получение потребовало бы
        // трёхкратного численного дифференцирования с неконтролируемым шумом. Ранее такой
        // вызов МОЛЧА возвращал неверный результат (обе производные считались нулевыми) —
        // теперь это явная ошибка вместо тихого искажения.
        require(!funcs.usesSecondDerivative) {
            "Решатель уравнения Вольтерры I рода не поддерживает семейство '${funcs.name}': " +
                "после редукции I->II рода вторая производная ядра недоступна аналитически. " +
                "Используйте theta, xi^<1>, xi^<2>, mu или lambda."
        }
    }

    /**
     * Точки, в которых редукция гарантированно делит на `K(t,t)` при любом сценарии.
     *
     * Собираются три группы:
     *
     *  1. узлы сетки и середины интервалов — опорные точки функционалов `theta`;
     *  2. гауссовы узлы составной квадратуры редуцированного оператора по полным ячейкам
     *     сетки — именно здесь `gEff` и редуцированное ядро вычисляются чаще всего;
     *  3. весь шаблон конечной разности `t ± k*h`, `k = 1..4`, вокруг каждой точки
     *     групп 1 и 2 — эти точки не совпадают ни с узлами, ни с серединами.
     *
     * Набор НЕ исчерпывающий и таким быть не может: оператор Вольтерры интегрирует по
     * `[a,t]` с усечённой последней ячейкой, поэтому его гауссовы узлы зависят от `t`.
     * Окончательную гарантию даёт [safeDiagonal], а этот список обеспечивает раннюю
     * диагностику ещё до начала счёта.
     */
    private fun diagonalCheckPoints(): DoubleArray {
        val breakpoints = grid.breakpoints
        val (referenceNodes, _) = quad.refNodesWeights()
        val base = ArrayList<Double>()
        for (i in breakpoints.indices) {
            base.add(breakpoints[i])
            if (i < breakpoints.size - 1) {
                val lo = breakpoints[i]
                val hi = breakpoints[i + 1]
                base.add(0.5 * (lo + hi))
                // Гауссовы узлы ячейки [x_i, x_{i+1}] составной квадратуры.
                val half = 0.5 * (hi - lo)
                val mid = 0.5 * (hi + lo)
                for (node in referenceNodes) base.add(mid + half * node)
            }
        }
        // Шаблон конечной разности вокруг каждой базовой точки; точки вне [a,b]
        // отбрасываются — там срабатывает односторонняя ветвь deriv4.
        val all = ArrayList<Double>(base)
        for (t in base) {
            for (k in 1..4) {
                val left = t - k * FINITE_DIFFERENCE_STEP
                val right = t + k * FINITE_DIFFERENCE_STEP
                if (left >= grid.a) all.add(left)
                if (right <= grid.b) all.add(right)
            }
        }
        return all.toDoubleArray()
    }

    /**
     * Первая производная по формуле четвёртого порядка.
     *
     * Внутри отрезка применяется пятиточечная центральная разность
     * `(-f(t+2h) + 8f(t+h) - 8f(t-h) + f(t-2h)) / (12h)`. Вблизи концов её шаблон вышел бы
     * за пределы `[a,b]`, где ядро и оператор доопределены нулём, что исказило бы
     * результат; поэтому там используются односторонние пятиточечные формулы того же
     * четвёртого порядка — «вперёд» у левого конца и «назад» у правого.
     */
    private fun deriv4(t: Double, f: (Double) -> Double): Double {
        val leftEnd = grid.a
        val rightEnd = grid.b
        val step = FINITE_DIFFERENCE_STEP
        val boundaryTolerance = BOUNDARY_RELATIVE_TOLERANCE * (rightEnd - leftEnd)
        return when {
            // Левый конец: центральный шаблон (t - 2h) вышел бы за a.
            t - 2 * step < leftEnd - boundaryTolerance ->
                (-25 * f(t) + 48 * f(t + step) - 36 * f(t + 2 * step) +
                    16 * f(t + 3 * step) - 3 * f(t + 4 * step)) / (12 * step)
            // Правый конец: центральный шаблон (t + 2h) вышел бы за b.
            t + 2 * step > rightEnd + boundaryTolerance ->
                (25 * f(t) - 48 * f(t - step) + 36 * f(t - 2 * step) -
                    16 * f(t - 3 * step) + 3 * f(t - 4 * step)) / (12 * step)
            // Внутренняя область: центральная пятиточечная разность.
            else ->
                (-f(t + 2 * step) + 8 * f(t + step) - 8 * f(t - step) + f(t - 2 * step)) / (12 * step)
        }
    }

    /**
     * Редуцированное ядро `K_W(t,s) = -K_t(t,s)/K(t,t)`.
     * Его производная по `t` аналитически недоступна и вычисляется конечной разностью.
     */
    private val reducedKernel = KernelV(
        k = { t, s -> -kernel.kT(t, s) / kernelDiagonal(t) },
        kT = { t, s -> deriv4(t) { argument -> -kernel.kT(argument, s) / kernelDiagonal(argument) } },
    )

    private val reducedOperator = VolterraOperator(reducedKernel, grid, quad)

    /** Правая часть редуцированного уравнения: `g(t) = f'(t)/K(t,t)`. */
    private val gEff = { t: Double -> rhsDeriv(t) / kernelDiagonal(t) }

    /**
     * Производная правой части `g'(t)`, нужная семействам функционалов с производной.
     *
     * Прямое численное дифференцирование всей `g` означало бы ВТОРОЕ дифференцирование
     * поверх `f'(t)`, которая сама получена аналитически по Лейбницу и содержит квадратуру:
     * шум квадратуры и ошибка округления при этом резко усиливаются.
     *
     * Поэтому используется разложение `g(t) = s(t) + r(t)`, где `s` — известная гладкая
     * часть решения, а `r = g - s` — малый остаток, несущий вклад квадратуры. Тогда
     * `g'(t) = s'(t) + r'(t)`: гладкая часть дифференцируется АНАЛИТИЧЕСКИ, а конечная
     * разность применяется только к остатку. Так из-под вычитания убран крупный гладкий
     * член, и катастрофическая потеря точности затрагивает лишь малую величину `|r|`.
     */
    private val gEffResidual = { t: Double -> gEff(t) - smoothPart(t) }
    private val gEffDeriv = { t: Double -> smoothPartDeriv(t) + deriv4(t, gEffResidual) }

    private val inner = VolterraSecondKindSolver(
        basis, funcs, reducedOperator, cL = 1.0, fEff = gEff, fEffDeriv = gEffDeriv,
        throwOnDivergence = throwOnDivergence,
    )

    /** Базовая коллокационная схема для редуцированного уравнения. */
    fun base(): SolutionFunc = inner.base()

    /** Итерация Слоана, применённая к редуцированному уравнению. */
    fun sloan(): SolutionFunc = inner.sloan()

    /** Схема Кулкарни для редуцированного уравнения. */
    fun kulkarni(): SolutionFunc = inner.kulkarni()

    /** Итерированная схема Кулкарни для редуцированного уравнения. */
    fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}
