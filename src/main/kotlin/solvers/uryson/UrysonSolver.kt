package solvers.uryson

import kotlin.math.abs
import numerics.GaussLegendre
import numerics.Grid
import numerics.LinearAlgebra
import numerics.MinimalSplineBasis
import numerics.NumericsContext
import numerics.ParallelAssembly
import numerics.SolutionFunc
import numerics.functionals.ApproxFunctional
import numerics.functionals.ProjFunctionals
import numerics.functionals.SupportPoints
import numerics.functionals.ValueFunctional
import numerics.reportConvergence

/**
 * Применяет функционал к функции значений, явно передавая нулевые производные.
 *
 * Проекционные функционалы `theta_j` производных не используют, но общий интерфейс
 * [ApproxFunctional] их принимает. Отдельное имя (вместо `apply`) выбрано, чтобы
 * исключить неоднозначность с одноимённой функцией-областью видимости из стандартной
 * библиотеки Kotlin.
 */
internal fun ApproxFunctional.applyTo(f: (Double) -> Double): Double = apply(f, { 0.0 }, { 0.0 })

/**
 * Возвращает функционал `theta_j` в виде линейной комбинации значений.
 *
 * Схемы Урысона (сборка `Xi`, якобиана и метод Nyström) работают напрямую с опорными
 * точками и коэффициентами функционала, поэтому нужен именно [ValueFunctional].
 * Семейство `theta` состоит из таких функционалов по построению.
 */
internal fun ProjFunctionals.valueFunctional(j: Int): ValueFunctional =
    chi(j) as? ValueFunctional
        ?: error("Функционал theta_$j не является линейной комбинацией значений (ValueFunctional).")

/**
 * Сплайн-пространство: веса, матрица Грама и интегральные характеристики базиса.
 *
 * Объединяет базис, сетку и квадратуру, предоставляя величины, общие для схем
 * второго и первого рода: веса дискретной нормы, веса метода Nyström и матрицу
 * стабилизатора Тихонова.
 *
 * @param basis базис минимальных сплайнов.
 * @param quad квадратура для интегралов по сеточным интервалам.
 */
class SplineSpace(
    val basis: MinimalSplineBasis,
    val quad: GaussLegendre,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    /**
     * Допуск отбрасывания узлов, совпавших с концами подынтервала.
     *
     * Значение берётся из ЕДИНОГО ИСТОЧНИКА [Grid.breakpointInclusionEps] (там же —
     * обоснование относительности и оговорка про мелкие отрезки), а не вычисляется
     * здесь повторно: раньше та же формула дублировалась в `VolterraOperator`, причём
     * была записана иначе (`coerceAtLeast` против `maxOf`).
     *
     * ОБЪЯВЛЕНО ДО [gramRInternal] НЕ СЛУЧАЙНО: инициализаторы свойств в Kotlin
     * выполняются в порядке объявления, а `buildGram()` вызывает [subBreakpoints],
     * читающий этот порог. При объявлении ПОСЛЕ порог был бы ещё `0.0`, и матрица
     * Грама считалась бы с ДРУГИМ разбиением — тихое изменение чисел без ошибки.
     */
    private val breakpointEps: Double = grid.breakpointInclusionEps

    /**
     * Веса дискретной нормы `w_j = (x_{j+3} - x_j)/3`, `j = -2..n-1` (индекс массива `j+2`).
     *
     * Пропорциональны длине носителя сплайна `omega_j`; их сумма равна длине отрезка,
     * что делает дискретную норму согласованной с `L^2`.
     */
    private val weightsInternal: DoubleArray = DoubleArray(dim) { (grid.x(it - 2 + 3) - grid.x(it - 2)) / 3.0 }

    /**
     * Веса дискретной нормы (КОПИЯ: мутация результата не затрагивает пространство).
     *
     * Поле ХОЛОДНОЕ: все вызывающие читают его один раз и сохраняют в своё поле
     * (см. `TikhonovSolver`), поэтому копия не попадает в горячий цикл.
     */
    val weights: DoubleArray get() = weightsInternal.copyOf()

    /** Веса метода Nyström `W_j = \int_a^b omega_j(s) ds`. */
    private val wIntInternal: DoubleArray = DoubleArray(dim) { k ->
        val j = k - 2
        quad.integrate(grid.breakpoints) { t -> basis.omega(j, t) }
    }

    /**
     * Веса метода Nyström (КОПИЯ, см. обоснование у [weights]).
     *
     * Единственный боевой читатель — `UrysonSecondKindSolver.nystrom`, где значение берётся
     * в локальную переменную ДО цикла Ньютона.
     */
    val wInt: DoubleArray get() = wIntInternal.copyOf()

    /**
     * Матрица Грама стабилизатора: `[R]_{i,j} = \int (omega_i omega_j + omega_i' omega_j') ds`.
     *
     * Это матрица скалярного произведения пространства Соболева `W^{1,2}`, то есть
     * стабилизатор Тихонова штрафует и саму функцию, и её первую производную.
     * Матрица симметрична, положительно определена и полосная: `|i-j| <= 2`, поскольку
     * носители сплайнов, отстоящих дальше, не пересекаются.
     */
    private val gramRInternal: Array<DoubleArray> = buildGram()

    /**
     * Матрица Грама стабилизатора (ГЛУБОКАЯ КОПИЯ, см. обоснование у [weights]).
     *
     * Копируются именно строки, а не только внешний массив: `copyOf()` на
     * `Array<DoubleArray>` даёт ПОВЕРХНОСТНУЮ копию, через которую содержимое
     * по-прежнему правилось бы насквозь — то есть защита была бы мнимой.
     */
    val gramR: Array<DoubleArray> get() = Array(gramRInternal.size) { gramRInternal[it].copyOf() }

    private fun buildGram(): Array<DoubleArray> {
        val r = LinearAlgebra.zeros(dim, dim)
        for (ki in 0 until dim) {
            val i = ki - 2
            for (kj in ki until dim) {
                val j = kj - 2
                if (abs(i - j) > 2) continue // полосная структура: носители не пересекаются
                val lo = maxOf(grid.x(i), grid.x(j))
                val hi = minOf(grid.x(i + 3), grid.x(j + 3))
                if (hi <= lo) continue
                val sub = subBreakpoints(lo, hi)
                val value = quad.integrate(sub) { t ->
                    basis.omega(i, t) * basis.omega(j, t) +
                        basis.omegaDeriv(i, t) * basis.omegaDeriv(j, t)
                }
                r[ki][kj] = value
                r[kj][ki] = value
            }
        }
        return r
    }

    /** Узлы сетки внутри `[lo, hi]` плюс концы — разбиение для составной квадратуры. */
    private fun subBreakpoints(lo: Double, hi: Double): DoubleArray {
        val pts = ArrayList<Double>()
        pts.add(lo)
        for (k in 0..n) {
            val xk = grid.x(k)
            if (xk > lo + breakpointEps && xk < hi - breakpointEps) pts.add(xk)
        }
        pts.add(hi)
        return pts.toDoubleArray()
    }

    /** Сумма весов `sum_j w_j`; по построению должна равняться длине отрезка `b - a`. */
    fun weightsSum(): Double = weightsInternal.sum()

    /** Квадратичная форма стабилизатора `Omega(x_h) = c^T R_h c`. */
    fun omegaReg(c: DoubleArray): Double {
        // Внутри класса читаем бэкинг-поле напрямую: копия здесь была бы лишней.
        val rc = LinearAlgebra.matVec(gramRInternal, c, ctx.backend)
        var s = 0.0
        for (i in c.indices) s += c[i] * rc[i]
        return s
    }

}

/**
 * Ядро нелинейного уравнения Урысона `K(t,s,u)` и его частная производная по `u`.
 *
 * Производная `dK/du` нужна для производной Фреше оператора, а через неё — для
 * аналитического якобиана метода Ньютона.
 */
interface Kernel {
    /** Значение ядра `K(t, s, u)`. */
    fun k(t: Double, s: Double, u: Double): Double

    /** Частная производная `dK/du(t, s, u)`. */
    fun dkdu(t: Double, s: Double, u: Double): Double
}

/**
 * Нелинейный интегральный оператор Урысона `(U x)(t) = \int_a^b K(t,s,x(s)) ds`.
 *
 * Интегралы вычисляются составной квадратурой Гаусса–Лежандра по сеточным интервалам.
 *
 * @param kernel ядро уравнения вместе с производной по `u`.
 * @param grid сетка, задающая отрезок интегрирования и разбиение для квадратуры.
 * @param quad квадратурная формула.
 */
class UrysohnOperator(val kernel: Kernel, val grid: Grid, val quad: GaussLegendre) {
    /** Значение `(U x)(t)` для произвольной функции `x(s)`. */
    fun apply(t: Double, x: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s, x(s)) }

    /**
     * Производная Фреше `(U'(x) h)(t) = \int_a^b dK/du(t,s,x(s)) h(s) ds`.
     *
     * Задаёт линеаризацию оператора в точке `x`; именно по этой формуле выводится
     * якобиан [CollocationCore.bMatrix], который вычисляется эффективнее — за один
     * проход по узлам квадратуры сразу для всех базисных функций.
     */
    fun frechet(t: Double, x: (Double) -> Double, h: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.dkdu(t, s, x(s)) * h(s) }

    /**
     * Глобальный набор узлов квадратуры по всей сетке: `\int h = sum_k gW[k] h(gNode[k])`.
     *
     * Предвычисление позволяет в схемах Кулкарни и Nyström не пересобирать разбиение
     * при каждом вычислении вложенных интегралов.
     *
     * READ-ONLY ПО СОГЛАШЕНИЮ: содержимое НЕЛЬЗЯ изменять. ГОРЯЧЕЕ поле — копия не
     * возвращается сознательно: массив читается в цикле [applyNodes] и на каждой
     * итерации квази-Ньютона. Записей в проекте нет.
     */
    val gNode: DoubleArray

    /** Веса квадратуры, соответствующие узлам [gNode]. READ-ONLY по соглашению (горячее). */
    val gW: DoubleArray

    init {
        val (referenceNodes, referenceWeights) = quad.refNodesWeights()
        val breakpoints = grid.breakpoints
        val nodes = ArrayList<Double>()
        val weights = ArrayList<Double>()
        for (m in 0 until breakpoints.size - 1) {
            val lo = breakpoints[m]
            val hi = breakpoints[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in referenceNodes.indices) {
                nodes.add(mid + half * referenceNodes[q])
                weights.add(half * referenceWeights[q])
            }
        }
        gNode = nodes.toDoubleArray()
        gW = weights.toDoubleArray()
    }

    /** Значение `(U x)(tau)` по предвычисленным значениям `x` в узлах [gNode]. */
    fun applyNodes(tau: Double, xNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k], xNodes[k])
        return s
    }
}



/**
 * Результат итераций Ньютона в пространстве коэффициентов.
 *
 * Ранее [UrysonSecondKindSolver.solveBase] возвращал `Pair<DoubleArray, Int>`, из которого
 * было невозможно узнать, сошлась ли итерация: число итераций, равное пределу,
 * одинаково возникает и при сходимости на последнем шаге, и при расходимости.
 *
 * @param coeffs найденные коэффициенты сплайна.
 * @param converged признак достижения сходимости.
 * @param iterations число ФАКТИЧЕСКИ ВЫПОЛНЕННЫХ шагов Ньютона (см. [NewtonRun]);
 *        `0` означает, что коэффициенты не изменились относительно начального
 *        приближения.
 * @param residual норма невязки. При `converged == true` — измеренная В ТОЙ ЖЕ
 *        точке, что возвращается в [coeffs]. При исчерпании предела шагов — невязка
 *        ПЕРЕД ПОСЛЕДНИМ шагом, а НЕ в возвращаемой точке: сознательное ограничение,
 *        полное обоснование — в KDoc [runNewtonIterations] и [NewtonRun.residual].
 */
class NewtonResult(
    val coeffs: DoubleArray,
    val converged: Boolean,
    val iterations: Int,
    val residual: Double,
)

/**
 * Ядро коллокационных вычислений: вектор `Xi(c) = Theta_h(U x_h)` и якобиан
 * `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
 *
 * Используется и схемами второго рода (метод Ньютона), и регуляризованной схемой
 * первого рода (метод Гаусса–Ньютона).
 */
class CollocationCore(
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val op: UrysohnOperator,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    val grid = basis.grid
    val n = grid.n
    private val quad = op.quad
    private val kernel = op.kernel

    /**
     * Эталонные узлы и веса квадратуры на [-1,1], полученные ОДИН РАЗ.
     *
     * [GaussLegendre.refNodesWeights] возвращает КОПИИ массивов, а [bMatrix] вызывается
     * НА КАЖДОЙ итерации Ньютона (`UrysonSecondKindSolver.newtonStep`) и Гаусса–Ньютона
     * (`TikhonovSolver.solveFixedAlpha`). Получение узлов внутри [bMatrix] давало бы две
     * аллокации на итерацию на ровном месте; здесь копия делается однократно.
     *
     * Значения и порядок арифметики те же: массивы неизменны и читаются только на чтение.
     */
    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    /**
     * Различные опорные точки всех функционалов `theta_j` (узлы и середины интервалов)
     * в ПОРЯДКЕ ПЕРВОГО ВХОЖДЕНИЯ при обходе `j = -2..n-1`.
     *
     * Порядок — часть контракта, а не деталь реализации: от него зависят нумерация
     * строк `G` в [bMatrix] и порядок суммирования в [xiVector]. READ-ONLY по соглашению.
     */
    val supportPts: DoubleArray

    /**
     * Индексация опорных точек по паре (номер функционала `j+2`, номер узла).
     *
     * Раньше здесь была `HashMap<Double, Int>` с поиском по ЗНАЧЕНИЮ точки: она требовала
     * побитового совпадения Double и работала лишь потому, что и заполнение, и чтение
     * шли из одного и того же массива `nodes` кэшированного функционала.
     */
    private val support: SupportPoints

    init {
        val (rn, rw) = quad.refNodesWeights()
        refNodes = rn
        refWeights = rw
        val vfs = Array(n + 2) { funcs.valueFunctional(it - 2) }
        support = SupportPoints.byFirstOccurrence(vfs, grid.breakpointInclusionEps)
        supportPts = support.points
    }

    /** Значения `(U x_h)` в опорных точках (по одному интегралу на точку). */
    fun uAtSupport(c: DoubleArray): DoubleArray =
        DoubleArray(supportPts.size) { p -> op.apply(supportPts[p]) { s -> basis.evalSpline(c, s) } }

    /** Вектор `Xi(c)_j = theta_j(U x_h)`, `j = -2..n-1` (индекс массива `j+2`). */
    fun xiVector(c: DoubleArray): DoubleArray {
        val uVals = uAtSupport(c)
        return DoubleArray(n + 2) { k ->
            val th = funcs.valueFunctional(k - 2)
            var s = 0.0
            for (q in th.nodes.indices) s += th.coeffs[q] * uVals[support.indexOf(k, q)]
            s
        }
    }

    /**
     * Якобиан `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
     *
     * Вычисляется в два прохода: сначала `G[p][i] = \int dK/du(tau_p, s, x_h(s)) omega_i(s) ds`
     * для всех опорных точек `tau_p` за один обход узлов квадратуры (на каждом интервале
     * ненулевыми являются лишь три базисных сплайна), затем строки `B` собираются как
     * линейные комбинации строк `G` с коэффициентами функционалов.
     */
    fun bMatrix(c: DoubleArray): Array<DoubleArray> {
        val np = supportPts.size
        val g = LinearAlgebra.zeros(np, n + 2)
        // Узлы/веса получены однократно в init: [refNodesWeights] отдаёт копии, а этот
        // метод вызывается на каждой итерации Ньютона / Гаусса–Ньютона.
        val nodes = refNodes
        val weights = refWeights
        for (m in 0 until n) {
            val lo = grid.x(m)
            val hi = grid.x(m + 1)
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in nodes.indices) {
                val s = mid + half * nodes[q]
                val weight = half * weights[q]
                val splineValue = basis.evalSpline(c, s)
                val activeOmega = basis.activeOmega(m, s) // omega_{m-2}, omega_{m-1}, omega_m
                for (p in 0 until np) {
                    val dk = kernel.dkdu(supportPts[p], s, splineValue) * weight
                    if (dk == 0.0) continue
                    val row = g[p]
                    row[m] += dk * activeOmega[0]
                    row[m + 1] += dk * activeOmega[1]
                    row[m + 2] += dk * activeOmega[2]
                }
            }
        }
        // Строки B независимы: каждая задача пишет в свою строку, g только читается.
        return ParallelAssembly.assembleRows(n + 2, n + 2, ctx.parallel) { k ->
            val th = funcs.valueFunctional(k - 2)
            val row = DoubleArray(n + 2)
            for (q in th.nodes.indices) {
                val gRow = g[support.indexOf(k, q)]
                val coefficient = th.coeffs[q]
                for (i in 0 until n + 2) row[i] += coefficient * gRow[i]
            }
            row
        }
    }
}

/**
 * Решатели нелинейного уравнения Урысона ВТОРОГО рода
 * `x(t) - lambda \int_a^b K(t,s,x(s)) ds = f(t)`.
 *
 * Реализованы четыре схемы: базовая коллокация, итерация Слоана, модификация
 * Кулкарни и сплайн-метод Nyström. Источники перечислены в `docs/REFERENCES.md`.
 *
 * Решатель не знает о модельных задачах: правая часть передаётся функцией [rhs],
 * а множитель — параметром [lambda]. Готовые фабрики для модельных задач находятся
 * в пакете `problems.uryson`.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство проекционных функционалов `theta_j`.
 * @param space сплайн-пространство (нужны веса Nyström `W_j`).
 * @param op оператор Урысона.
 * @param lambda множитель перед интегральным оператором.
 * @param rhs правая часть `f(t)`.
 * @param tol критерий останова по норме невязки и норме шага.
 * @param maxIter предел числа итераций Ньютона в БАЗОВОЙ схеме.
 * @param kulkarniMaxIter предел числа итераций квази-Ньютона в схеме Кулкарни.
 * @param nystromMaxIter предел числа итераций Ньютона в схеме Nyström.
 * @param throwOnDivergence поведение при недостижении сходимости Ньютона:
 *        `true` (по умолчанию) — исключение, `false` — результат с
 *        `converged = false`. Ранее все три схемы только писали предупреждение
 *        в лог и возвращали результат: при программном использовании библиотеки
 *        такое предупреждение оставалось незамеченным.
 */
class UrysonSecondKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val space: SplineSpace,
    val op: UrysohnOperator,
    val lambda: Double,
    val rhs: (Double) -> Double,
    val tol: Double = DEFAULT_TOLERANCE,
    val maxIter: Int = DEFAULT_MAX_ITERATIONS,
    val kulkarniMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    val nystromMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    val throwOnDivergence: Boolean = true,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // Семейство и сплайн-пространство считают ЧАСТИ ТОЙ ЖЕ задачи — тем же бэкендом.
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "space", space.ctx)
    }

    companion object {
        /** Критерий останова по умолчанию: близко к машинной точности. */
        const val DEFAULT_TOLERANCE = 1e-12

        /**
         * Предел итераций Ньютона в базовой схеме. Взят с большим запасом: метод
         * квадратично сходится за единицы итераций, а предел защищает лишь от
         * зацикливания на вырожденных данных.
         */
        const val DEFAULT_MAX_ITERATIONS = 10_000

        /**
         * Предел итераций для схем Кулкарни и Nyström.
         *
         * Он существенно меньше [DEFAULT_MAX_ITERATIONS], потому что цена одной
         * итерации здесь несопоставимо выше: схема Кулкарни на каждом шаге вычисляет
         * вложенные интегралы, а Nyström строит конечно-разностный якобиан, требующий
         * `P` полных вычислений правой части. Ранее это значение было зашито в код
         * константой, из-за чего параметр `maxIter` на данные схемы не влиял вопреки
         * документации.
         */
        const val DEFAULT_FIXED_POINT_MAX_ITERATIONS = 60

        /** Нижняя граница критерия останова для схем, использующих аналитический якобиан. */
        private const val NEWTON_TOLERANCE_FLOOR = 1e-13

        /** Нижняя граница критерия останова для схемы с конечно-разностным якобианом. */
        private const val FINITE_DIFFERENCE_TOLERANCE_FLOOR = 1e-12

        /**
         * Относительный шаг конечно-разностного якобиана в схеме Nyström.
         *
         * Значение близко к корню из машинного эпсилона (`sqrt(2.2e-16) ~ 1.5e-8`) —
         * это классический компромисс для односторонней разности между ошибкой
         * усечения (растёт с шагом) и ошибкой округления (растёт при его уменьшении).
         */
        private const val JACOBIAN_RELATIVE_STEP = 1e-7
    }

    val grid = basis.grid
    val n = grid.n

    /** Предвычисленные значения `theta_j(f)` для точной правой части. */
    private val thetaF: DoubleArray = DoubleArray(n + 2) { k -> funcs.valueFunctional(k - 2).applyTo(rhs) }

    private val collocation = CollocationCore(basis, funcs, op, ctx)

    /**
     * Базовая схема: `c = theta(f) + lambda Xi(c)`.
     *
     * Решается методом Ньютона для `F(c) = c - theta(f) - lambda Xi(c) = 0` с
     * аналитическим якобианом `J = I - lambda B(c)`. Ньютон выбран вместо простой
     * итерации, поскольку сходится и при отсутствии сжатия (например, при `lambda = 1`
     * и кубическом ядре).
     *
     * @return коэффициенты сплайна вместе со сведениями о сходимости Ньютона.
     * @throws IllegalStateException при недостижении сходимости, если [throwOnDivergence].
     */
    fun solveBase(): NewtonResult {
        val c = thetaF.copyOf()
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        val run = runNewtonIterations(
            x = c,
            maxSteps = maxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val xi = collocation.xiVector(current)
                DoubleArray(n + 2) { current[it] - thetaF[it] - lambda * xi[it] }
            },
            stepAt = analyticNewtonStep,
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Ньютон (базовая схема Урысона)",
            iterations = run.performedSteps,
            maxIterations = maxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        return NewtonResult(c, run.converged, run.performedSteps, run.residual)
    }

    /**
     * Пояснение для диагностики, если счёт прерван ЗАСТОЕМ ([NewtonRun.stalled]).
     *
     * Без него сообщение [reportConvergence] утверждало бы «не достигнуто за N
     * итераций (предел M)», из чего читатель заключил бы, что поможет повышение
     * предела. При застое это неверно: предел НЕ исчерпан, итерация просто
     * перестала двигаться.
     */
    private fun stallHint(run: NewtonRun): String? =
        if (!run.stalled) {
            null
        } else {
            "Счёт прерван ЗАСТОЕМ на шаге ${run.performedSteps}: норма шага упала ниже допуска, " +
                "а невязка осталась выше него. Предел итераций НЕ исчерпан, поэтому его повышение " +
                "не поможет: итерация перестала двигаться (вероятные причины — плохая " +
                "обусловленность якобиана либо его приближённость). Нужно менять сетку, " +
                "начальное приближение или требуемую точность"
        }

    /**
     * Шаг Ньютона с АНАЛИТИЧЕСКИМ якобианом `I - lambda B(c)`, ОБЩИЙ для базовой
     * схемы и схемы Кулкарни.
     *
     * Совпадение ЗДЕСЬ НЕ СЛУЧАЙНОЕ и потому вынесено в одно место: у Кулкарни
     * якобиан базовой схемы выступает ПРЕДОБУСЛАВЛИВАТЕЛЕМ квази-Ньютона ПО
     * ПРЕДПИСАНИЮ ИСТОЧНИКА (см. KDoc [kulkarni]) — то есть тождественность шага
     * является частью определения схемы, а не совпадением реализаций. Различие двух
     * схем живёт ЦЕЛИКОМ в `residualAt` (`F(c)` против `c - G_K(c)`), и общий шаг
     * это различие не размывает. Схема `nystrom` сюда НЕ входит: у неё другой,
     * конечно-разностный якобиан и другое пространство неизвестных.
     */
    private val analyticNewtonStep: (DoubleArray, DoubleArray) -> DoubleArray =
        { current, residual -> newtonStep(current, DoubleArray(n + 2) { -residual[it] }) }

    /** Шаг Ньютона с якобианом `J = I - lambda B(c)` (строки собираются независимо). */
    private fun newtonStep(c: DoubleArray, negativeResidual: DoubleArray): DoubleArray {
        val b = collocation.bMatrix(c)
        val jacobian = ParallelAssembly.assembleRows(n + 2, n + 2, ctx.parallel) { r ->
            val row = DoubleArray(n + 2) { col -> -lambda * b[r][col] }
            row[r] += 1.0
            row
        }
        return LinearAlgebra.solve(jacobian, negativeResidual, ctx.backend)
    }

    /**
     * Реконструкция правой части схемы Кулкарни по коэффициентам `c` сплайна `y_h`.
     *
     * Возвращает тройку:
     *  - `yhNodes` — значения `y_h` в узлах квадратуры [UrysohnOperator.gNode];
     *  - `gAtSupport` — функция `g(t) = f(t) + lambda (U y_h)(t)` (вычисляется в любой
     *    точке, в частности в опорных точках функционалов);
     *  - `gCoeffs` — коэффициенты проекции `P_theta g`.
     *
     * Блок нужен дважды: на каждой итерации квази-Ньютона (внутри `G_K`) и после
     * выхода из цикла — при восстановлении `x_h^K = y_h + (I - P_theta) g`.
     */
    private fun projectedRhs(c: DoubleArray): Triple<DoubleArray, (Double) -> Double, DoubleArray> {
        val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
        val gAtSupport = { t: Double -> rhs(t) + lambda * op.applyNodes(t, yhNodes) }
        val gCoeffs = funcs.projectorCoeffs(gAtSupport)
        return Triple(yhNodes, gAtSupport, gCoeffs)
    }

    /** Базовое приближение `x_h` как сплайн. */
    fun base(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        return SolutionFunc(
            eval = { t -> basis.evalSpline(c, t) },
            converged = newton.converged,
            iterations = newton.iterations,
            residual = newton.residual,
        )
    }

    /** Итерация Слоана: `\tilde x_h(t) = f(t) + lambda (U x_h)(t)`. */
    fun sloan(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        val splineSolution = { t: Double -> basis.evalSpline(c, t) }
        val eval = { t: Double -> rhs(t) + lambda * op.apply(t) { s -> splineSolution(s) } }
        return SolutionFunc(
            eval = eval,
            converged = newton.converged,
            iterations = newton.iterations,
            residual = newton.residual,
        )
    }

    /**
     * Модификация Кулкарни.
     *
     * Система для `y_h = P_theta x_h^K`: `c = theta(f) + lambda Theta(U(arg(c)))`, где
     * `arg = y_h + (I - P_theta)[f + lambda U(y_h)]`. Решается квази-Ньютоном, у которого
     * в роли предобуславливателя выступает якобиан БАЗОВОЙ схемы `I - lambda B(c)`.
     * Такой выбор предписан источником и обеспечивает сходимость при отсутствии сжатия,
     * где простая итерация расходится.
     *
     * Итоговое приближение восстанавливается как `x_h^K = y_h + (I - P_theta)[f + lambda U(y_h)]`.
     */
    fun kulkarni(): SolutionFunc {
        val fNodes = DoubleArray(op.gNode.size) { rhs(op.gNode[it]) }

        /** Правая часть системы Кулкарни `G_K(c)`. */
        fun gK(c: DoubleArray): DoubleArray {
            val (yhNodes, _, gCoeffs) = projectedRhs(c)
            val uyhNodes = DoubleArray(op.gNode.size) { op.applyNodes(op.gNode[it], yhNodes) }
            val gNodes = DoubleArray(op.gNode.size) { fNodes[it] + lambda * uyhNodes[it] }
            // Остаток проектора в узлах квадратуры: arg = y_h + (I - P_theta) g.
            val argNodes = DoubleArray(op.gNode.size) {
                yhNodes[it] + gNodes[it] - basis.evalSpline(gCoeffs, op.gNode[it])
            }
            return DoubleArray(n + 2) { k ->
                val th = funcs.valueFunctional(k - 2)
                var acc = 0.0
                for (q in th.nodes.indices) acc += th.coeffs[q] * op.applyNodes(th.nodes[q], argNodes)
                thetaF[k] + lambda * acc
            }
        }

        val c = thetaF.copyOf()
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        val run = runNewtonIterations(
            x = c,
            maxSteps = kulkarniMaxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val g = gK(current)
                DoubleArray(n + 2) { current[it] - g[it] }
            },
            stepAt = analyticNewtonStep,
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Квази-Ньютон (схема Кулкарни Урысона)",
            iterations = run.performedSteps,
            maxIterations = kulkarniMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        val (_, gAtSupport, gCoeffs) = projectedRhs(c)
        val eval = { t: Double -> basis.evalSpline(c, t) + (gAtSupport(t) - basis.evalSpline(gCoeffs, t)) }
        return SolutionFunc(
            eval = eval,
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    /**
     * Сплайн-метод Nyström: `x_h^N(t) = f(t) + lambda sum_j theta_j(g_t) W_j`,
     * где `g_t(s) = K(t, s, x_h^N(s))`.
     *
     * Неизвестными являются значения решения в опорных точках функционалов. Система
     * решается методом Ньютона с КОНЕЧНО-РАЗНОСТНЫМ якобианом: формула не содержит
     * вложенных интегралов, поэтому вычисление правой части дёшево, и разностный
     * якобиан оказывается выгоднее аналитического.
     *
     * Начальное приближение — проекция постоянной функции `P_theta(1)`. Нулевой старт
     * непригоден: для ядер с `dK/du(t,s,0) = 0` (например, кубических) якобиан в нуле
     * вырождается и метод не сдвигается с места.
     */
    fun nystrom(): SolutionFunc {
        // Порядок точек — ПОРЯДОК ПЕРВОГО ВХОЖДЕНИЯ при обходе `j = -2..n-1`: он задаёт
        // нумерацию неизвестных, порядок строк якобиана и вектор начального приближения.
        // Индексация — по паре (номер функционала, номер узла), а НЕ поиском по значению
        // точки в `HashMap<Double, Int>`, который требовал побитового совпадения Double.
        val vfs = Array(n + 2) { funcs.valueFunctional(it - 2) }
        val support = SupportPoints.byFirstOccurrence(vfs, grid.breakpointInclusionEps)
        val pts = support.points
        val wInt = space.wInt

        /** Правая часть схемы Nyström по значениям решения в опорных точках. */
        fun evalAtVals(t: Double, xVals: DoubleArray): Double {
            var acc = 0.0
            for (j in -2..n - 1) {
                val th = vfs[j + 2]
                var gtVal = 0.0
                for (q in th.nodes.indices) {
                    val supportPoint = th.nodes[q]
                    gtVal += th.coeffs[q] * op.kernel.k(t, supportPoint, xVals[support.indexOf(j + 2, q)])
                }
                acc += gtVal * wInt[j + 2]
            }
            return rhs(t) + lambda * acc
        }

        val p = pts.size
        val constantProjection = funcs.projectorCoeffs({ 1.0 })
        val x = DoubleArray(p) { basis.evalSpline(constantProjection, pts[it]) }
        val newtonTol = maxOf(tol, FINITE_DIFFERENCE_TOLERANCE_FLOOR)
        // Правая часть `G(x)` в ТЕКУЩЕЙ точке, передаваемая из расчёта невязки в расчёт
        // шага. Не восстанавливается как `x - F(x)` СОЗНАТЕЛЬНО: такое обратное вычитание
        // в IEEE 754 воспроизводит исходные биты лишь пока `x` и `G(x)` близки по порядку,
        // а при большом разбросе порядков теряет точность — это молча сдвинуло бы числа.
        // Повторное вычисление `G` тоже нежелательно: оно стоит `p` вычислений правой
        // части. Передача опирается на ГАРАНТИЮ ПОРЯДКА ВЫЗОВОВ, явно записанную в KDoc
        // параметра `stepAt` функции [runNewtonIterations]: `residualAt` всегда вызывается
        // непосредственно перед `stepAt` в той же точке. `null` вместо мёртвого
        // нулевого массива — не микрооптимизация, а КОНТРОЛЬ: если гарантию когда-нибудь
        // нарушат, `error` ниже упадёт громко, тогда как нулевой массив молча дал бы
        // неверный якобиан и правдоподобные числа.
        var currentG: DoubleArray? = null
        val run = runNewtonIterations(
            x = x,
            maxSteps = nystromMaxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val gx = DoubleArray(p) { evalAtVals(pts[it], current) }
                currentG = gx
                DoubleArray(p) { current[it] - gx[it] }
            },
            stepAt = { current, residual ->
                val gx = currentG
                    ?: error(
                        "Нарушена гарантия порядка вызовов runNewtonIterations: stepAt вызван без " +
                            "предшествующего residualAt, поэтому G(x) в текущей точке неизвестна.",
                    )
                val jacobian = LinearAlgebra.zeros(p, p)
                for (col in 0 until p) {
                    val saved = current[col]
                    // Шаг масштабируется величиной переменной, чтобы сохранять точность
                    // и при больших, и при близких к нулю значениях.
                    val step = JACOBIAN_RELATIVE_STEP * (abs(saved) + 1.0)
                    current[col] = saved + step
                    val perturbed = DoubleArray(p) { evalAtVals(pts[it], current) }
                    current[col] = saved
                    // F(x) = x - G(x), поэтому dF[row]/dx[col] = [row == col] - dG[row]/dx[col].
                    for (row in 0 until p) {
                        val identity = if (row == col) 1.0 else 0.0
                        jacobian[row][col] = (identity * step - (perturbed[row] - gx[row])) / step
                    }
                }
                LinearAlgebra.solve(jacobian, DoubleArray(p) { -residual[it] }, ctx.backend)
            },
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Ньютон (схема Nyström Урысона)",
            iterations = run.performedSteps,
            maxIterations = nystromMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        val xFinal = x
        return SolutionFunc(
            eval = { t -> evalAtVals(t, xFinal) },
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }
}

/**
 * Результат регуляризованного решения уравнения первого рода.
 *
 * @param coeffs коэффициенты найденного сплайна.
 * @param eval вычислитель приближённого решения.
 * @param alpha выбранный параметр регуляризации.
 * @param resid дискретная невязка при этом параметре.
 * @param omega значение стабилизатора `c^T R_h c`.
 */
class FirstKindSolution(
    val coeffs: DoubleArray,
    val eval: (Double) -> Double,
    val alpha: Double,
    val resid: Double,
    val omega: Double,
)

/**
 * Регуляризованная сплайн-коллокация для нелинейного уравнения Урысона ПЕРВОГО рода
 * `\int_a^b K(t,s,x(s)) ds = f(t)`.
 *
 * Задача некорректна, поэтому минимизируется функционал Тихонова
 * `||Theta_h(U x_h) - Theta_h(f^delta)||^2 + alpha c^T R_h c`, где стабилизатор
 * `R_h` задан нормой пространства `W^{1,2}`. Минимизация выполняется итерациями
 * Гаусса–Ньютона, а параметр `alpha` выбирается по принципу невязки Морозова.
 * Источники — в `docs/REFERENCES.md`.
 *
 * Решатель не знает о модельных задачах: зашумлённые данные передаются готовым
 * вектором `theta_j(f^delta)`, который строится средствами пакета `problems.uryson`.
 *
 * @param tau коэффициент запаса в принципе Морозова; теория требует лишь `tau > 1`.
 * @param gnTol критерий останова Гаусса–Ньютона по норме шага.
 * @param gnMaxIter предел числа итераций Гаусса–Ньютона при фиксированном `alpha`.
 * @param throwOnDivergence поведение при недостижении сходимости Гаусса–Ньютона.
 *        ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ — `false`, в отличие от остальных решателей.
 *        Причина: [solveFixedAlpha] вызывается внутри гомотопии [solveMorozov]
 *        десятки раз с тёплым стартом, и недостижение шагового критерия на
 *        ОТДЕЛЬНОМ `alpha` — штатная часть пути по параметру регуляризации
 *        (некорректная задача, промежуточные `alpha` заведомо плохо обусловлены),
 *        а не ошибка: итоговое решение выбирается по принципу невязки Морозова.
 *        Предупреждение в лог пишется в любом случае.
 */
class UrysonFirstKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val space: SplineSpace,
    val op: UrysohnOperator,
    val tau: Double = DEFAULT_TAU,
    val gnTol: Double = DEFAULT_GN_TOLERANCE,
    val gnMaxIter: Int = DEFAULT_GN_MAX_ITERATIONS,
    val throwOnDivergence: Boolean = false,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // КРИТИЧНО именно здесь: [solveMorozov] считает стабилизатор `Omega` через
        // `space.omegaReg` (то есть через `space.ctx.backend`), а систему Гаусса–Ньютона —
        // через собственный `ctx.backend`. При расхождении две части ОДНОГО критерия
        // Морозова считались бы разными реализациями LU — молча.
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "space", space.ctx)
    }

    companion object {
        /**
         * Коэффициент запаса в принципе невязки Морозова.
         *
         * Теория требует только `tau > 1`; конкретное значение — выбор реализации:
         * чем оно ближе к единице, тем меньше сглаживание, но тем выше чувствительность
         * к неточности оценки уровня шума.
         */
        const val DEFAULT_TAU = 1.1

        /** Критерий останова Гаусса–Ньютона по равномерной норме шага. */
        const val DEFAULT_GN_TOLERANCE = 1e-10

        /**
         * Предел итераций Гаусса–Ньютона при фиксированном `alpha`. Метод применяется
         * внутри гомотопии по параметру регуляризации, где каждый следующий запуск
         * стартует с предыдущего решения, поэтому большого числа итераций не требуется.
         */
        const val DEFAULT_GN_MAX_ITERATIONS = 50

        /** Верхняя граница показателя степени в логарифмической сетке параметра `alpha`. */
        private const val ALPHA_MAX_EXPONENT = 2.0

        /** Нижняя граница показателя степени в логарифмической сетке параметра `alpha`. */
        private const val ALPHA_MIN_EXPONENT = -12.0

        /**
         * Число шагов гомотопии по `alpha`. Вместе с границами показателя задаёт шаг
         * сетки `10^{-0.25}`: достаточно мелко, чтобы точка Морозова определялась
         * устойчиво, и достаточно грубо, чтобы весь путь считался за разумное время.
         */
        private const val ALPHA_PATH_STEPS = 56

        /**
         * Порог, ниже которого тёплый старт считается вырожденным и заменяется
         * проекцией постоянной функции. Нужен для ядер с `dK/du(t,s,0) = 0`, где
         * из нулевого приближения якобиан обращается в ноль.
         */
        private const val DEGENERATE_START_THRESHOLD = 1e-8
    }

    val grid = basis.grid
    val n = grid.n
    private val core = CollocationCore(basis, funcs, op, ctx)
    private val weights = space.weights
    private val gramR = space.gramR

    /** Вектор значений функционалов `theta_j(f)` для произвольной правой части. */
    fun thetaOf(f: (Double) -> Double): DoubleArray =
        DoubleArray(n + 2) { funcs.valueFunctional(it - 2).applyTo(f) }

    /**
     * Решает регуляризованную задачу при ФИКСИРОВАННОМ `alpha` методом Гаусса–Ньютона.
     *
     * Шаг определяется системой
     * `(B^T W_h B + alpha R_h) delta = -B^T W_h (Xi - theta(f^delta)) - alpha R_h c`.
     *
     * @param thetaFDelta вектор `theta_j(f^delta)` зашумлённых данных.
     * @param alpha параметр регуляризации, строго положительный.
     * @param c0 начальное приближение коэффициентов.
     */
    fun solveFixedAlpha(thetaFDelta: DoubleArray, alpha: Double, c0: DoubleArray): DoubleArray {
        require(alpha > 0.0) { "Параметр регуляризации alpha должен быть положительным, получено alpha=$alpha" }
        val c = c0.copyOf()
        var lastStep = Double.NaN
        repeat(gnMaxIter) {
            val xi = core.xiVector(c)
            val b = core.bMatrix(c)
            val btwb = LinearAlgebra.atWa(b, weights, ctx.backend)
            val lhs = LinearAlgebra.addScaled(btwb, gramR, alpha, ctx.backend)
            val r = DoubleArray(n + 2) { (xi[it] - thetaFDelta[it]) * weights[it] }
            val btr = LinearAlgebra.matTransVec(b, r, ctx.backend)
            val rc = LinearAlgebra.matVec(gramR, c, ctx.backend)
            val rhs = DoubleArray(n + 2) { -btr[it] - alpha * rc[it] }
            val delta = LinearAlgebra.solve(lhs, rhs, ctx.backend)
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < gnTol) return c
        }
        reportConvergence(
            converged = false,
            throwOnDivergence = throwOnDivergence,
            methodName = "Гаусс–Ньютон (Урысон, I род, alpha=$alpha)",
            iterations = gnMaxIter,
            maxIterations = gnMaxIter,
            residual = lastStep,
            tolerance = gnTol,
            hint = "На отдельном alpha это ожидаемо внутри гомотопии по параметру " +
                "регуляризации; итоговое alpha выбирается по принципу невязки Морозова",
        )
        return c
    }

    /** Дискретная невязка `res_h = ||Theta_h(U x_h) - Theta_h(f^delta)||` во взвешенной норме. */
    fun residual(c: DoubleArray, thetaFDelta: DoubleArray): Double {
        val xi = core.xiVector(c)
        var s = 0.0
        for (j in 0 until n + 2) {
            val d = xi[j] - thetaFDelta[j]
            s += weights[j] * d * d
        }
        return Math.sqrt(s)
    }

    /**
     * Выбирает `alpha` по принципу невязки Морозова: наибольшее значение, при котором
     * `res_h(alpha) <= tau * C_theta * sqrt(b - a) * delta`.
     *
     * Реализовано гомотопией по УБЫВАЮЩЕМУ `alpha` с тёплым стартом: решение при
     * очередном значении служит начальным приближением для следующего. Для некорректной
     * задачи это заметно стабилизирует Гаусса–Ньютона и делает невязку монотонной.
     *
     * Если цель недостижима на всём пути (например, на слишком грубой сетке),
     * возвращается решение с наименьшей достигнутой невязкой — без «раскачки» решения.
     *
     * @param thetaFDelta вектор `theta_j(f^delta)` зашумлённых данных.
     * @param delta уровень шума в норме `L^2`; при `delta = 0` путь проходится целиком.
     */
    fun solveMorozov(thetaFDelta: DoubleArray, delta: Double): FirstKindSolution {
        val barDelta = funcs.cChi() * Math.sqrt(grid.b - grid.a) * delta
        val target = tau * barDelta
        val initialGuess = funcs.projectorCoeffs({ 1.0 })
        var c = initialGuess.copyOf()
        var chosen: FirstKindSolution? = null
        var bestFallback: FirstKindSolution? = null
        var bestFallbackResidual = Double.MAX_VALUE

        for (i in 0..ALPHA_PATH_STEPS) {
            val exponent = ALPHA_MAX_EXPONENT +
                (ALPHA_MIN_EXPONENT - ALPHA_MAX_EXPONENT) * i / ALPHA_PATH_STEPS
            val alpha = Math.pow(10.0, exponent)
            val start = if (LinearAlgebra.normInf(c) < DEGENERATE_START_THRESHOLD) {
                initialGuess.copyOf()
            } else {
                c
            }
            c = solveFixedAlpha(thetaFDelta, alpha, start)
            val res = residual(c, thetaFDelta)
            if (delta == 0.0) {
                // Шума нет: критерий Морозова вырождается, идём до наименьшего alpha.
                val coeffs = c.copyOf()
                chosen = FirstKindSolution(
                    coeffs, { t -> basis.evalSpline(coeffs, t) }, alpha, res, space.omegaReg(coeffs),
                )
                continue
            }
            if (res <= target) {
                val coeffs = c.copyOf()
                chosen = FirstKindSolution(
                    coeffs, { t -> basis.evalSpline(coeffs, t) }, alpha, res, space.omegaReg(coeffs),
                )
                break
            }
            if (res < bestFallbackResidual) {
                bestFallbackResidual = res
                val coeffs = c.copyOf()
                bestFallback = FirstKindSolution(
                    coeffs, { t -> basis.evalSpline(coeffs, t) }, alpha, res, space.omegaReg(coeffs),
                )
            }
        }
        return chosen
            ?: bestFallback
            ?: error("Путь по параметру регуляризации пуст: проверьте ALPHA_PATH_STEPS.")
    }
}
