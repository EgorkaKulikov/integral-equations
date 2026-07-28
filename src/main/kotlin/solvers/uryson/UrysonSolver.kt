package solvers.uryson

import java.util.logging.Logger
import kotlin.math.abs
import numerics.GaussLegendre
import numerics.Grid
import numerics.LinearAlgebra
import numerics.MinimalSplineBasis
import numerics.ParallelAssembly
import numerics.functionals.ApproxFunctional
import numerics.functionals.ProjFunctionals
import numerics.functionals.ValueFunctional

/** Логгер решателей Урысона: предупреждения о недостижении сходимости и т. п. */
private val urysonLogger: Logger = Logger.getLogger("solvers.uryson.UrysonSolver")

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
class SplineSpace(val basis: MinimalSplineBasis, val quad: GaussLegendre) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    /**
     * Веса дискретной нормы `w_j = (x_{j+3} - x_j)/3`, `j = -2..n-1` (индекс массива `j+2`).
     *
     * Пропорциональны длине носителя сплайна `omega_j`; их сумма равна длине отрезка,
     * что делает дискретную норму согласованной с `L^2`.
     */
    val weights: DoubleArray = DoubleArray(dim) { (grid.x(it - 2 + 3) - grid.x(it - 2)) / 3.0 }

    /** Веса метода Nyström `W_j = \int_a^b omega_j(s) ds`. */
    val wInt: DoubleArray = DoubleArray(dim) { k ->
        val j = k - 2
        quad.integrate(grid.breakpoints) { t -> basis.omega(j, t) }
    }

    /**
     * Матрица Грама стабилизатора: `[R]_{i,j} = \int (omega_i omega_j + omega_i' omega_j') ds`.
     *
     * Это матрица скалярного произведения пространства Соболева `W^{1,2}`, то есть
     * стабилизатор Тихонова штрафует и саму функцию, и её первую производную.
     * Матрица симметрична, положительно определена и полосная: `|i-j| <= 2`, поскольку
     * носители сплайнов, отстоящих дальше, не пересекаются.
     */
    val gramR: Array<DoubleArray> = buildGram()

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
            if (xk > lo + BREAKPOINT_ABS_EPS && xk < hi - BREAKPOINT_ABS_EPS) pts.add(xk)
        }
        pts.add(hi)
        return pts.toDoubleArray()
    }

    /** Сумма весов `sum_j w_j`; по построению должна равняться длине отрезка `b - a`. */
    fun weightsSum(): Double = weights.sum()

    /** Квадратичная форма стабилизатора `Omega(x_h) = c^T R_h c`. */
    fun omegaReg(c: DoubleArray): Double {
        val rc = LinearAlgebra.matVec(gramR, c)
        var s = 0.0
        for (i in c.indices) s += c[i] * rc[i]
        return s
    }

    private companion object {
        /**
         * Абсолютный допуск для отбрасывания внутренних узлов сетки, совпадающих
         * (с точностью до машинного эпсилона) с концами подынтервала.
         */
        const val BREAKPOINT_ABS_EPS = 1e-15
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
     */
    val gNode: DoubleArray

    /** Веса квадратуры, соответствующие узлам [gNode]. */
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

/** Результат решения: вычислитель приближённого решения `x_h(t)` и число итераций. */
class SolutionFunc(val eval: (Double) -> Double, val iterations: Int)

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
) {
    val grid = basis.grid
    val n = grid.n
    private val quad = op.quad
    private val kernel = op.kernel

    /** Различные опорные точки всех функционалов `theta_j` (узлы и середины интервалов). */
    val supportPts: DoubleArray
    private val ptIdx: HashMap<Double, Int> = HashMap()

    init {
        val set = LinkedHashSet<Double>()
        for (j in -2..n - 1) for (p in funcs.valueFunctional(j).nodes) set.add(p)
        supportPts = set.toDoubleArray()
        for (i in supportPts.indices) ptIdx[supportPts[i]] = i
    }

    /** Индекс опорной точки; отсутствие точки означает рассогласование построения. */
    private fun indexOf(point: Double): Int = ptIdx[point]
        ?: error("Опорная точка $point не найдена среди точек функционалов theta_j.")

    /** Значения `(U x_h)` в опорных точках (по одному интегралу на точку). */
    fun uAtSupport(c: DoubleArray): DoubleArray =
        DoubleArray(supportPts.size) { p -> op.apply(supportPts[p]) { s -> basis.evalSpline(c, s) } }

    /** Вектор `Xi(c)_j = theta_j(U x_h)`, `j = -2..n-1` (индекс массива `j+2`). */
    fun xiVector(c: DoubleArray): DoubleArray {
        val uVals = uAtSupport(c)
        return DoubleArray(n + 2) { k ->
            val th = funcs.valueFunctional(k - 2)
            var s = 0.0
            for (q in th.nodes.indices) s += th.coeffs[q] * uVals[indexOf(th.nodes[q])]
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
        val (nodes, weights) = quad.refNodesWeights()
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
        return ParallelAssembly.assembleRows(n + 2, n + 2) { k ->
            val th = funcs.valueFunctional(k - 2)
            val row = DoubleArray(n + 2)
            for (q in th.nodes.indices) {
                val gRow = g[indexOf(th.nodes[q])]
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
 */
class SecondKindSolver(
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
) {
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

    private val collocation = CollocationCore(basis, funcs, op)

    /**
     * Базовая схема: `c = theta(f) + lambda Xi(c)`.
     *
     * Решается методом Ньютона для `F(c) = c - theta(f) - lambda Xi(c) = 0` с
     * аналитическим якобианом `J = I - lambda B(c)`. Ньютон выбран вместо простой
     * итерации, поскольку сходится и при отсутствии сжатия (например, при `lambda = 1`
     * и кубическом ядре).
     *
     * @return пара «коэффициенты сплайна, число выполненных итераций».
     */
    fun solveBase(): Pair<DoubleArray, Int> {
        val c = thetaF.copyOf()
        var iter = 0
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < maxIter) {
            val xi = collocation.xiVector(c)
            val f = DoubleArray(n + 2) { c[it] - thetaF[it] - lambda * xi[it] }
            lastResidual = LinearAlgebra.normInf(f)
            iter++
            if (lastResidual < newtonTol) { converged = true; break }
            val delta = newtonStep(c, DoubleArray(n + 2) { -f[it] })
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Ньютон (solveBase) не сошёлся за $iter итераций (предел maxIter=$maxIter): " +
                    "последняя невязка ||F||=$lastResidual, норма шага ||delta||=$lastStep",
            )
        }
        return c to iter
    }

    /** Шаг Ньютона с якобианом `J = I - lambda B(c)` (строки собираются независимо). */
    private fun newtonStep(c: DoubleArray, negativeResidual: DoubleArray): DoubleArray {
        val b = collocation.bMatrix(c)
        val jacobian = ParallelAssembly.assembleRows(n + 2, n + 2) { r ->
            val row = DoubleArray(n + 2) { col -> -lambda * b[r][col] }
            row[r] += 1.0
            row
        }
        return LinearAlgebra.solve(jacobian, negativeResidual)
    }

    /** Базовое приближение `x_h` как сплайн. */
    fun base(): SolutionFunc {
        val (c, iterations) = solveBase()
        return SolutionFunc({ t -> basis.evalSpline(c, t) }, iterations)
    }

    /** Итерация Слоана: `\tilde x_h(t) = f(t) + lambda (U x_h)(t)`. */
    fun sloan(): SolutionFunc {
        val (c, iterations) = solveBase()
        val splineSolution = { t: Double -> basis.evalSpline(c, t) }
        val eval = { t: Double -> rhs(t) + lambda * op.apply(t) { s -> splineSolution(s) } }
        return SolutionFunc(eval, iterations)
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
            val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
            val uyhNodes = DoubleArray(op.gNode.size) { op.applyNodes(op.gNode[it], yhNodes) }
            val gNodes = DoubleArray(op.gNode.size) { fNodes[it] + lambda * uyhNodes[it] }
            val gAtSupport = { t: Double -> rhs(t) + lambda * op.applyNodes(t, yhNodes) }
            val gCoeffs = funcs.projectorCoeffs(gAtSupport)
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
        var iter = 0
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < kulkarniMaxIter) {
            val g = gK(c)
            val residual = DoubleArray(n + 2) { c[it] - g[it] }
            lastResidual = LinearAlgebra.normInf(residual)
            iter++
            if (lastResidual < newtonTol) { converged = true; break }
            val delta = newtonStep(c, DoubleArray(n + 2) { -residual[it] })
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Квази-Ньютон (kulkarni) не сошёлся за $iter итераций " +
                    "(предел kulkarniMaxIter=$kulkarniMaxIter): последняя невязка ||F||=$lastResidual, " +
                    "норма шага ||delta||=$lastStep",
            )
        }
        val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
        val gAtSupport = { t: Double -> rhs(t) + lambda * op.applyNodes(t, yhNodes) }
        val gCoeffs = funcs.projectorCoeffs(gAtSupport)
        val eval = { t: Double -> basis.evalSpline(c, t) + (gAtSupport(t) - basis.evalSpline(gCoeffs, t)) }
        return SolutionFunc(eval, iter)
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
        val pointSet = LinkedHashSet<Double>()
        for (j in -2..n - 1) for (p in funcs.valueFunctional(j).nodes) pointSet.add(p)
        val pts = pointSet.toDoubleArray()
        val wInt = space.wInt
        val ptIndex = HashMap<Double, Int>()
        for (i in pts.indices) ptIndex[pts[i]] = i

        fun indexOf(point: Double): Int = ptIndex[point]
            ?: error("Опорная точка $point не найдена среди точек функционалов theta_j.")

        /** Правая часть схемы Nyström по значениям решения в опорных точках. */
        fun evalAtVals(t: Double, xVals: DoubleArray): Double {
            var acc = 0.0
            for (j in -2..n - 1) {
                val th = funcs.valueFunctional(j)
                var gtVal = 0.0
                for (q in th.nodes.indices) {
                    val supportPoint = th.nodes[q]
                    gtVal += th.coeffs[q] * op.kernel.k(t, supportPoint, xVals[indexOf(supportPoint)])
                }
                acc += gtVal * wInt[j + 2]
            }
            return rhs(t) + lambda * acc
        }

        val p = pts.size
        val constantProjection = funcs.projectorCoeffs({ 1.0 })
        val x = DoubleArray(p) { basis.evalSpline(constantProjection, pts[it]) }
        var iter = 0
        val newtonTol = maxOf(tol, FINITE_DIFFERENCE_TOLERANCE_FLOOR)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < nystromMaxIter) {
            val gx = DoubleArray(p) { evalAtVals(pts[it], x) }
            val residual = DoubleArray(p) { x[it] - gx[it] }
            lastResidual = LinearAlgebra.normInf(residual)
            iter++
            if (lastResidual < newtonTol) { converged = true; break }
            val jacobian = LinearAlgebra.zeros(p, p)
            for (col in 0 until p) {
                val saved = x[col]
                // Шаг масштабируется величиной переменной, чтобы сохранять точность
                // и при больших, и при близких к нулю значениях.
                val step = JACOBIAN_RELATIVE_STEP * (abs(saved) + 1.0)
                x[col] = saved + step
                val perturbed = DoubleArray(p) { evalAtVals(pts[it], x) }
                x[col] = saved
                // F(x) = x - G(x), поэтому dF[row]/dx[col] = [row == col] - dG[row]/dx[col].
                for (row in 0 until p) {
                    val identity = if (row == col) 1.0 else 0.0
                    jacobian[row][col] = (identity * step - (perturbed[row] - gx[row])) / step
                }
            }
            val delta = LinearAlgebra.solve(jacobian, DoubleArray(p) { -residual[it] })
            for (i in x.indices) x[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Ньютон (nystrom) не сошёлся за $iter итераций (предел nystromMaxIter=$nystromMaxIter): " +
                    "последняя невязка ||F||=$lastResidual, норма шага ||delta||=$lastStep",
            )
        }
        val xFinal = x
        return SolutionFunc({ t -> evalAtVals(t, xFinal) }, iter)
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
 */
class FirstKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val space: SplineSpace,
    val op: UrysohnOperator,
    val tau: Double = DEFAULT_TAU,
    val gnTol: Double = DEFAULT_GN_TOLERANCE,
    val gnMaxIter: Int = DEFAULT_GN_MAX_ITERATIONS,
) {
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
    private val core = CollocationCore(basis, funcs, op)
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
            val btwb = LinearAlgebra.atWa(b, weights)
            val lhs = LinearAlgebra.addScaled(btwb, gramR, alpha)
            val r = DoubleArray(n + 2) { (xi[it] - thetaFDelta[it]) * weights[it] }
            val btr = LinearAlgebra.matTransVec(b, r)
            val rc = LinearAlgebra.matVec(gramR, c)
            val rhs = DoubleArray(n + 2) { -btr[it] - alpha * rc[it] }
            val delta = LinearAlgebra.solve(lhs, rhs)
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < gnTol) return c
        }
        urysonLogger.warning(
            "Гаусс–Ньютон (solveFixedAlpha, alpha=$alpha) не сошёлся за $gnMaxIter итераций: " +
                "норма шага ||delta||=$lastStep",
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
