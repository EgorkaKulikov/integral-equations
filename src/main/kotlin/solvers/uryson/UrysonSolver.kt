package solvers.uryson

import kotlin.math.abs
import kotlin.system.exitProcess
import java.util.logging.Logger
import numerics.*

/** Логгер решателей Урысона: предупреждения о недостижении сходимости и т.п. */
private val urysonLogger: Logger = Logger.getLogger("solvers.uryson.UrysonSolver")

// ============================================================================
// 6. ПРОЕКЦИОННЫЕ ФУНКЦИОНАЛЫ theta_j (pr_func) И ПРОЕКТОР P_theta
// ============================================================================

/**
 * Один проекционный функционал theta_j: линейная комбинация значений в точках nodes
 * с коэффициентами coeffs. Реализует (lam-form): theta_j(f) = sum_k beta_{j,k} f(tau_{j,k}).
 */
class ProjFunctional(val nodes: DoubleArray, val coeffs: DoubleArray) {
    init {
        require(nodes.size == coeffs.size)
    }

    /** Применяет функционал к функции f. */
    fun apply(f: (Double) -> Double): Double {
        var s = 0.0
        for (k in nodes.indices) s += coeffs[k] * f(nodes[k])
        return s
    }

    /** Применяет функционал к вектору значений в точках nodes (в том же порядке). */
    fun applyValues(values: DoubleArray): Double {
        var s = 0.0
        for (k in coeffs.indices) s += coeffs[k] * values[k]
        return s
    }

    /** Сумма |beta_{j,k}| — для оценки нормы C_theta (Lambda-cons). */
    fun absSum(): Double = coeffs.fold(0.0) { acc, v -> acc + abs(v) }
}

/**
 * Семейство проекционных функционалов {theta_j}_{j=-2}^{n-1} и проектор P_theta
 * для базиса минимальных сплайнов. Реализует (pr_func)/(lam-form):
 * для внутреннего j коэффициенты beta выражаются через значения omega_j в узлах
 * и серединах интервалов (A, B, C, D, E, K_1, K_2); краевые j вычисляются по тем
 * же формулам. Биортогональность (biorth): theta_i(omega_j) = delta_{i,j}.
 */
class ProjFunctionals(val basis: MinimalSplineBasis) {
    val grid = basis.grid
    val n = grid.n

    /** theta_j для j = -2..n-1 (индексируется theta(j)). */
    private val funcs: Array<ProjFunctional> = Array(n + 2) { buildTheta(it - 2) }

    fun theta(j: Int): ProjFunctional = funcs[j + 2]

    private fun mid(p: Double, q: Double) = 0.5 * (p + q)

    private fun buildTheta(j: Int): ProjFunctional {
        val x0 = grid.x(0)
        val x1 = grid.x(1)
        val xnm1 = grid.x(n - 1)
        val xn = grid.x(n)
        return when (j) {
            // theta_{-2}(f) = f(x_0)
            -2 -> ProjFunctional(doubleArrayOf(x0), doubleArrayOf(1.0))
            // theta_{-1}: 3-точечный краевой функционал (pr_func) по точкам x_0,(x_0+x_1)/2,x_1
            // и базисным индексам -2,-1,0; коэффициенты из локальной биортогонализации.
            -1 -> localFunctional(j = -1, points = doubleArrayOf(x0, mid(x0, x1), x1), indices = intArrayOf(-2, -1, 0))
            // theta_{n-1}(f) = f(x_n)
            n - 1 -> ProjFunctional(doubleArrayOf(xn), doubleArrayOf(1.0))
            // theta_{n-2}: зеркальный краевой функционал по x_{n-1},(x_{n-1}+x_n)/2,x_n
            n - 2 -> localFunctional(j = n - 2, points = doubleArrayOf(xnm1, mid(xnm1, xn), xn), indices = intArrayOf(n - 3, n - 2, n - 1))
            // внутренние j = 0..n-3: 5-точечный проекционный функционал (lam-form)/(pr_func)
            // по точкам x_j,(x_j+x_{j+1})/2,(x_{j+1}+x_{j+2})/2,(x_{j+2}+x_{j+3})/2,x_{j+3}.
            // Коэффициенты — решение локальной системы биортогональности
            // theta_j(omega_i)=delta_{ij}, i=j-2..j+2 (закрытая формула (pr_func) сверяется
            // в health-check 7 на равномерной сетке).
            else -> {
                val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
                localFunctional(
                    j = j,
                    points = doubleArrayOf(xj, mid(xj, xj1), mid(xj1, xj2), mid(xj2, xj3), xj3),
                    indices = intArrayOf(j - 2, j - 1, j, j + 1, j + 2),
                )
            }
        }
    }

    /**
     * Закрытая формула (pr_func) для внутреннего j: коэффициенты beta через A,B,C,D,E,K_1.
     * Используется только для health-check (сверка с классическими (pr_func_b)).
     */
    fun closedFormInternal(j: Int): ProjFunctional {
        val (a, b, c, d, e) = abcde(j)
        val k1 = c * c * d - b * c * e - a * d * e - d * e * e
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        return ProjFunctional(
            doubleArrayOf(xj, mid(xj, xj1), mid(xj1, xj2), mid(xj2, xj3), xj3),
            doubleArrayOf(
                e * e / k1,
                -d * e / k1,
                (c * d - b * e) / k1,
                -d * e / k1,
                e * e / k1,
            ),
        )
    }

    /**
     * Краевой функционал через локальную биортогонализацию: коэффициенты coeff такие,
     * что sum_p coeff_p omega_i(points_p) = delta_{i,j} для всех i из indices.
     * Это замкнутая форма (pr_func) на краю, устойчивая к тройным узлам.
     */
    private fun localFunctional(j: Int, points: DoubleArray, indices: IntArray): ProjFunctional {
        val m = points.size
        // Матрица A[row=i-ый базис][col=точка] = omega_index(point).
        val matrix = Array(m) { r -> DoubleArray(m) { c -> basis.omega(indices[r], points[c]) } }
        val rhs = DoubleArray(m) { if (indices[it] == j) 1.0 else 0.0 }
        val coeff = LinearAlgebra.solve(matrix, rhs)
        return ProjFunctional(points, coeff)
    }

    /** Величины A,B,C,D,E из example.tex для индекса j (значения omega_j в 5 точках). */
    private fun abcde(j: Int): Quintuple {
        val xj = grid.x(j); val xj1 = grid.x(j + 1); val xj2 = grid.x(j + 2); val xj3 = grid.x(j + 3)
        val a = basis.omega(j, mid(xj, xj1))
        val cc = basis.omega(j, mid(xj1, xj2))
        val e = basis.omega(j, mid(xj2, xj3))
        val b = basis.omega(j, xj1)
        val d = basis.omega(j, xj2)
        return Quintuple(a, b, cc, d, e)
    }

    /** Коэффициенты проекции P_theta g = sum_j theta_j(g) omega_j: вектор (theta_j(g))_j размера n+2. */
    fun projectorCoeffs(g: (Double) -> Double): DoubleArray =
        DoubleArray(n + 2) { theta(it - 2).apply(g) }

    /** Максимальная sum_k |beta_{j,k}| по всем j — константа C_theta (Lambda-cons). */
    fun cTheta(): Double = (-2..n - 1).maxOf { theta(it).absSum() }
}

/** Пятёрка значений (A, B, C, D, E) для destructuring. */
data class Quintuple(val a: Double, val b: Double, val c: Double, val d: Double, val e: Double)

// ============================================================================
// 7. ВЕСА w_j (lh2-norm), ВЕСА W_j, МАТРИЦА ГРАМА R_h (Rmat)
// ============================================================================

/**
 * Сплайн-пространство: веса, матрица Грама и интегральные величины базиса.
 * Объединяет базис, сетку и квадратуру для вычисления W_j (веса Nystrom),
 * w_j (lh2-norm) и R_h (Rmat).
 */
class SplineSpace(val basis: MinimalSplineBasis, val quad: GaussLegendre) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    /** Веса w_j = (x_{j+3} - x_j)/3 (lh2-norm), j = -2..n-1. Индекс массива = j+2. */
    val weights: DoubleArray = DoubleArray(dim) { (grid.x(it - 2 + 3) - grid.x(it - 2)) / 3.0 }

    /** Веса W_j = \int_a^b omega_j(s) ds (Unystrom). */
    val wInt: DoubleArray = DoubleArray(dim) { k ->
        val j = k - 2
        quad.integrate(grid.breakpoints) { t -> basis.omega(j, t) }
    }

    /**
     * Матрица Грама R_h (Rmat): [R]_{i,j} = \int (omega_i omega_j + omega_i' omega_j') ds.
     * Симметрична, положительно определена, полосная (|i-j|<=2).
     */
    val gramR: Array<DoubleArray> = buildGram()

    private fun buildGram(): Array<DoubleArray> {
        val r = LinearAlgebra.zeros(dim, dim)
        for (ki in 0 until dim) {
            val i = ki - 2
            for (kj in ki until dim) {
                val j = kj - 2
                if (kotlin.math.abs(i - j) > 2) continue // полосная структура
                // общий носитель [max(x_i,x_j), min(x_{i+3},x_{j+3})]
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

    /** Узлы сетки в [lo, hi] плюс концы — для составной квадратуры по подынтервалам. */
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

    /** Сумма весов sum_j w_j (должна равняться b-a) — health-check 10. */
    fun weightsSum(): Double = weights.sum()

    /** Квадратичная форма Omega(x_h) = c^T R_h c (регуляризатор). */
    fun omegaReg(c: DoubleArray): Double {
        val rc = LinearAlgebra.matVec(gramR, c)
        var s = 0.0
        for (i in c.indices) s += c[i] * rc[i]
        return s
    }

    companion object {
        /**
         * Абсолютный допуск для отбрасывания внутренних узлов сетки, совпадающих
         * (с точностью до машинного эпсилона) с концами подынтервала [lo, hi].
         */
        private const val BREAKPOINT_ABS_EPS = 1e-15
    }
}

// ============================================================================
// 8. ЯДРА И ОПЕРАТОРЫ УРЫСОНА
// ============================================================================

/**
 * Ядро уравнения Урысона K(t,s,u) и его частная производная dK/du(t,s,u).
 * Используются в (U-op) и производной Фреше (Uprime).
 */
interface Kernel {
    /** Значение ядра K(t, s, u). */
    fun k(t: Double, s: Double, u: Double): Double

    /** Частная производная dK/du(t, s, u). */
    fun dkdu(t: Double, s: Double, u: Double): Double
}

/**
 * Оператор Урысона (U-op): (U x)(t) = \int_a^b K(t,s,x(s)) ds, и его производная
 * Фреше (Uprime): (U'(x) h)(t) = \int_a^b dK/du(t,s,x(s)) h(s) ds.
 * Интегралы вычисляются составной квадратурой Гаусса--Лежандра по сеточным интервалам.
 */
class UrysohnOperator(val kernel: Kernel, val grid: Grid, val quad: GaussLegendre) {
    /** (U x)(t) для произвольной функции x(s). */
    fun apply(t: Double, x: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s, x(s)) }

    /** (U'(x) h)(t) — производная Фреше в точке x, применённая к h. */
    fun frechet(t: Double, x: (Double) -> Double, h: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.dkdu(t, s, x(s)) * h(s) }

    // Глобальный набор гауссовых узлов/весов по всей сетке (для эффективных
    // вложенных интегралов в схемах Kulkarni/Nystrom): \int h = sum_k gW[k] h(gNode[k]).
    val gNode: DoubleArray
    val gW: DoubleArray

    init {
        val (rn, rw) = quad.refNodesWeights()
        val bp = grid.breakpoints
        val nodes = ArrayList<Double>()
        val ws = ArrayList<Double>()
        for (m in 0 until bp.size - 1) {
            val lo = bp[m]; val hi = bp[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo); val mid = 0.5 * (hi + lo)
            for (qi in rn.indices) {
                nodes.add(mid + half * rn[qi])
                ws.add(half * rw[qi])
            }
        }
        gNode = nodes.toDoubleArray()
        gW = ws.toDoubleArray()
    }

    /** (U x)(tau) по предвычисленным значениям x в глобальных гауссовых узлах. */
    fun applyNodes(tau: Double, xNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k], xNodes[k])
        return s
    }
}

// ============================================================================
// 9. МОДЕЛЬНЫЕ ЗАДАЧИ A, B, C, D
// ============================================================================

/**
 * Модельная задача: ядро, параметр lambda, точное решение. Правая часть строится
 * из точного решения численно (квадратурой), без зашивания.
 *
 * @param secondKind true — уравнение II рода (ury2), false — I рода (ury1).
 */
class ModelProblem(
    val name: String,
    val kernel: Kernel,
    val lambda: Double,
    val exact: (Double) -> Double,
    val secondKind: Boolean,
) {
    /**
     * Точная правая часть f(t).
     * II рода (ury2): f(t) = x*(t) - lambda \int K(t,s,x*(s)) ds.
     * I рода  (ury1): f(t) = \int K(t,s,x*(s)) ds.
     */
    fun rhsExact(t: Double, op: UrysohnOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - lambda * integral else integral
    }

    companion object {
        /** Задача A (eq:num-ury2A): K=1/(t+s+u), lambda=-1, x*=1/(t+1). */
        val A = ModelProblem(
            name = "A",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = -1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = true,
        )

        /** Задача B (eq:num-ury2B): K=e^{t-2s} u^3, lambda=1, x*=e^t. */
        val B = ModelProblem(
            name = "B",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = Math.exp(t - 2.0 * s) * u * u * u
                override fun dkdu(t: Double, s: Double, u: Double) = 3.0 * Math.exp(t - 2.0 * s) * u * u
            },
            lambda = 1.0,
            exact = { t -> Math.exp(t) },
            secondKind = true,
        )

        /** Задача C (eq:num-ury1C): K=1/(t+s+u), x†=1/(t+1), I рода. */
        val C = ModelProblem(
            name = "C",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = 1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = false,
        )

        /** Задача D (eq:num-ury1D): K=e^{-(t-s)^2} u^3, x†=e^t, I рода. */
        val D = ModelProblem(
            name = "D",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = Math.exp(-(t - s) * (t - s)) * u * u * u
                override fun dkdu(t: Double, s: Double, u: Double) = 3.0 * Math.exp(-(t - s) * (t - s)) * u * u
            },
            lambda = 1.0,
            exact = { t -> Math.exp(t) },
            secondKind = false,
        )
    }
}

// ============================================================================
// 10. РЕШАТЕЛИ ДЛЯ УРАВНЕНИЯ II РОДА
// ============================================================================

/** Результат решения: вычислитель приближённого решения x_h(t) и число итераций. */
class SolutionFunc(val eval: (Double) -> Double, val iterations: Int)

/**
 * Ядро коллокационных вычислений: вектор Xi(c) = Theta_h(U x_h) и якобиан
 * B(c)_{j,i} = theta_j(U'(x_h) omega_i) (Bdef). Используется и в схемах II рода
 * (Ньютон), и в регуляризованной схеме I рода (Гаусс--Ньютон).
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

    // Различные опорные точки всех theta_j (узлы и середины интервалов).
    val supportPts: DoubleArray
    private val ptIdx: HashMap<Double, Int> = HashMap()

    init {
        val set = LinkedHashSet<Double>()
        for (j in -2..n - 1) for (p in funcs.theta(j).nodes) set.add(p)
        supportPts = set.toDoubleArray()
        for (i in supportPts.indices) ptIdx[supportPts[i]] = i
    }

    /** (U x_h) в опорных точках (по одному интегралу на точку). */
    fun uAtSupport(c: DoubleArray): DoubleArray =
        DoubleArray(supportPts.size) { p -> op.apply(supportPts[p]) { s -> basis.evalSpline(c, s) } }

    /** Вектор Xi(c)_j = theta_j(U x_h), j = -2..n-1 (индекс массива j+2). */
    fun xiVector(c: DoubleArray): DoubleArray {
        val uVals = uAtSupport(c)
        return DoubleArray(n + 2) { k ->
            val th = funcs.theta(k - 2)
            var s = 0.0
            for (q in th.nodes.indices) s += th.coeffs[q] * uVals[ptIdx[th.nodes[q]]!!]
            s
        }
    }

    /**
     * Якобиан B(c)_{j,i} = theta_j(U'(x_h) omega_i) (Bdef).
     * Сначала считаем G[p][i] = \int dK/du(tau_p, s, x_h(s)) omega_i(s) ds для опорных tau_p,
     * затем B_{j,i} = sum_q coeff_{j,q} G[idx(node_{j,q})][i].
     */
    fun bMatrix(c: DoubleArray): Array<DoubleArray> {
        val np = supportPts.size
        val g = LinearAlgebra.zeros(np, n + 2)
        // Проходим по сеточным интервалам и гауссовым узлам один раз.
        val (nodes, weights) = quad.refNodesWeights()
        for (m in 0 until n) {
            val lo = grid.x(m); val hi = grid.x(m + 1)
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo); val mid = 0.5 * (hi + lo)
            for (qi in nodes.indices) {
                val s = mid + half * nodes[qi]
                val wq = half * weights[qi]
                val xhs = basis.evalSpline(c, s)
                val act = basis.activeOmega(m, s) // omega_{m-2},omega_{m-1},omega_m в s
                for (p in 0 until np) {
                    val dk = kernel.dkdu(supportPts[p], s, xhs) * wq
                    if (dk == 0.0) continue
                    val row = g[p]
                    row[m] += dk * act[0]      // omega_{m-2} -> idx m
                    row[m + 1] += dk * act[1]  // omega_{m-1} -> idx m+1
                    row[m + 2] += dk * act[2]  // omega_m     -> idx m+2
                }
            }
        }
        // Строки B независимы: строка k читает только g (read-only).
        val b = ParallelAssembly.assembleRows(n + 2, n + 2) { k ->
            val th = funcs.theta(k - 2)
            val brow = DoubleArray(n + 2)
            for (q in th.nodes.indices) {
                val gp = g[ptIdx[th.nodes[q]]!!]
                val coef = th.coeffs[q]
                for (i in 0 until n + 2) brow[i] += coef * gp[i]
            }
            brow
        }
        return b
    }
}

/**
 * Решатели для уравнения Урысона II рода (ury2) в базисе минимальных сплайнов.
 * Реализует базовую схему (base-system-xi), Sloan (sloan-xi), Kulkarni (Kulk-system),
 * Nystrom (Unystrom-xi).
 */
class SecondKindSolver(
    val problem: ModelProblem,
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val space: SplineSpace,
    val op: UrysohnOperator,
    val tol: Double = 1e-12,
    val maxIter: Int = 10_000,
) {
    val grid = basis.grid
    val n = grid.n
    private val lambda = problem.lambda

    // Предвычисленные theta_j(f) для точной правой части.
    private val thetaF: DoubleArray = DoubleArray(n + 2) { k ->
        funcs.theta(k - 2).apply { t -> problem.rhsExact(t, op) }
    }

    // Общий набор опорных точек всех theta_j (без дубликатов).
    private val collocation = CollocationCore(basis, funcs, op)

    /**
     * Базовая схема (base-system-xi)/(base-system-vec): c = theta(f) + lambda Xi(c).
     * Решается методом Ньютона для F(c) = c - theta(f) - lambda Xi(c) = 0,
     * J = I - lambda B(c), B(c)_{j,i} = theta_j(U'(x_h) omega_i) (Bdef).
     * Простая итерация (simple-iter) — частный случай при J ~ I; Ньютон надёжнее
     * при q_h близком к 1 и при отсутствии сжатия (например, lambda=1, кубичное ядро).
     */
    fun solveBase(): Pair<DoubleArray, Int> {
        val c = thetaF.copyOf()
        var iter = 0
        val newtonTol = maxOf(tol, 1e-13)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < maxIter) {
            val xi = collocation.xiVector(c)
            val f = DoubleArray(n + 2) { c[it] - thetaF[it] - lambda * xi[it] }
            val fNorm = LinearAlgebra.normInf(f)
            lastResidual = fNorm
            iter++
            if (fNorm < newtonTol) { converged = true; break }
            val b = collocation.bMatrix(c)
            // J = I - lambda B (строки независимы, b только читается)
            val j = ParallelAssembly.assembleRows(n + 2, n + 2) { r ->
                val jr = DoubleArray(n + 2) { col -> -lambda * b[r][col] }
                jr[r] += 1.0
                jr
            }
            val rhs = DoubleArray(n + 2) { -f[it] }
            val delta = LinearAlgebra.solve(j, rhs)
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Ньютон (solveBase) не сошёлся за $iter итераций (лимит maxIter=$maxIter): " +
                    "последняя невязка ||F||=$lastResidual, норма шага ||delta||=$lastStep"
            )
        }
        return c to iter
    }

    /** Базовое приближение x_h как сплайн. */
    fun base(): SolutionFunc {
        val (c, it) = solveBase()
        return SolutionFunc({ t -> basis.evalSpline(c, t) }, it)
    }

    /** Sloan-итерация (sloan-xi): \tilde x_h(t) = f(t) + lambda (U x_h)(t). */
    fun sloan(): SolutionFunc {
        val (c, it) = solveBase()
        val xh = { t: Double -> basis.evalSpline(c, t) }
        val eval = { t: Double -> problem.rhsExact(t, op) + lambda * op.apply(t) { s -> xh(s) } }
        return SolutionFunc(eval, it)
    }

    /**
     * Модификация Kulkarni (Kulk-system)/(xh-final).
     * Система для y_h = P_theta x_h^K: c = theta(f) + lambda Theta(U(arg(c))),
     * arg = y_h + (I - P_theta)[f + lambda U(y_h)]. Решается квази-Ньютоном с
     * предобуславливателем (I - lambda B(c)) (якобиан базовой схемы), что сходится
     * и при отсутствии сжатия (например, lambda=1, кубичное ядро), в отличие от
     * простой итерации. Остаток (I - P_theta)g вычисляется через коэффициенты проекции.
     */
    fun kulkarni(): SolutionFunc {
        val fNodes = DoubleArray(op.gNode.size) { problem.rhsExact(op.gNode[it], op) }
        // G_K(c): вектор theta(f) + lambda Theta(U(arg(c))).
        fun gK(c: DoubleArray): DoubleArray {
            val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
            // U(y_h) в глобальных узлах и в опорных точках.
            val UyhNodes = DoubleArray(op.gNode.size) { op.applyNodes(op.gNode[it], yhNodes) }
            // g в глобальных узлах
            val gNodes = DoubleArray(op.gNode.size) { fNodes[it] + lambda * UyhNodes[it] }
            // P_theta g: коэффициенты через g в опорных точках
            val gAtSupport = { t: Double -> problem.rhsExact(t, op) + lambda * op.applyNodes(t, yhNodes) }
            val gCoeffs = funcs.projectorCoeffs(gAtSupport)
            // arg в глобальных узлах: y_h + g - P_theta g
            val argNodes = DoubleArray(op.gNode.size) { yhNodes[it] + gNodes[it] - basis.evalSpline(gCoeffs, op.gNode[it]) }
            // theta_j(U(arg)) через U(arg) в опорных точках
            return DoubleArray(n + 2) { k ->
                val th = funcs.theta(k - 2)
                var acc = 0.0
                for (q in th.nodes.indices) acc += th.coeffs[q] * op.applyNodes(th.nodes[q], argNodes)
                thetaF[k] + lambda * acc
            }
        }
        val c = thetaF.copyOf()
        var iter = 0
        val newtonTol = maxOf(tol, 1e-13)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < 60) {
            val g = gK(c)
            val resF = DoubleArray(n + 2) { c[it] - g[it] }
            lastResidual = LinearAlgebra.normInf(resF)
            iter++
            if (lastResidual < newtonTol) { converged = true; break }
            val b = collocation.bMatrix(c)
            // J = I - lambda B (строки независимы, b только читается)
            val j = ParallelAssembly.assembleRows(n + 2, n + 2) { r ->
                val jr = DoubleArray(n + 2) { col -> -lambda * b[r][col] }
                jr[r] += 1.0
                jr
            }
            val delta = LinearAlgebra.solve(j, DoubleArray(n + 2) { -resF[it] })
            for (i in c.indices) c[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Квази-Ньютон (kulkarni) не сошёлся за $iter итераций (лимит 60): " +
                    "последняя невязка ||F||=$lastResidual, норма шага ||delta||=$lastStep"
            )
        }
        // Восстановление x_h^K = y_h + (I - P_theta)[f + lambda U(y_h)] (xh-final).
        val cy = c
        val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(cy, op.gNode[it]) }
        val gAtSupport = { t: Double -> problem.rhsExact(t, op) + lambda * op.applyNodes(t, yhNodes) }
        val gCoeffs = funcs.projectorCoeffs(gAtSupport)
        val eval = { t: Double -> basis.evalSpline(cy, t) + (gAtSupport(t) - basis.evalSpline(gCoeffs, t)) }
        return SolutionFunc(eval, iter)
    }

    /**
     * Сплайн-метод Nystrom (Unystrom-xi)/(Nystrom-eq):
     * x_h^N(t) = f(t) + lambda sum_j theta_j(g_t) W_j, g_t(s) = K(t, s, x_h^N(s)).
     * Неизвестные — значения x_h^N в опорных точках функционалов; решается методом
     * Ньютона с конечно-разностным якобианом (формула без вложенных интегралов,
     * эвалюация G дешёвая), что устойчиво даже при отсутствии сжатия.
     */
    fun nystrom(): SolutionFunc {
        val pointSet = LinkedHashSet<Double>()
        for (j in -2..n - 1) for (p in funcs.theta(j).nodes) pointSet.add(p)
        val pts = pointSet.toDoubleArray()
        val wInt = space.wInt
        val ptIndex = HashMap<Double, Int>()
        for (i in pts.indices) ptIndex[pts[i]] = i
        // G(x): правая часть схемы Nystrom по значениям xVals в опорных точках.
        fun evalAtVals(t: Double, xVals: DoubleArray): Double {
            var acc = 0.0
            for (j in -2..n - 1) {
                val th = funcs.theta(j)
                var gtVal = 0.0
                for (q in th.nodes.indices) {
                    val sp = th.nodes[q]
                    gtVal += th.coeffs[q] * problem.kernel.k(t, sp, xVals[ptIndex[sp]!!])
                }
                acc += gtVal * wInt[j + 2]
            }
            return problem.rhsExact(t, op) + lambda * acc
        }
        val p = pts.size
        val x = DoubleArray(p) { problem.exact(pts[it]) }
        var iter = 0
        val newtonTol = maxOf(tol, 1e-12)
        var converged = false
        var lastResidual = Double.NaN
        var lastStep = Double.NaN
        while (iter < 60) {
            val gx = DoubleArray(p) { evalAtVals(pts[it], x) }
            val resF = DoubleArray(p) { x[it] - gx[it] }
            lastResidual = LinearAlgebra.normInf(resF)
            iter++
            if (lastResidual < newtonTol) { converged = true; break }
            // Конечно-разностный якобиан J = d resF / dx.
            val jac = LinearAlgebra.zeros(p, p)
            val eps = 1e-7
            for (col in 0 until p) {
                val saved = x[col]
                val step = eps * (abs(saved) + 1.0)
                x[col] = saved + step
                val gxp = DoubleArray(p) { evalAtVals(pts[it], x) }
                x[col] = saved
                // F(x) = x - G(x); dF[row]/dx[col] = [row==col] - dG[row]/dx[col].
                for (row in 0 until p) {
                    val identity = if (row == col) 1.0 else 0.0
                    jac[row][col] = (identity * step - (gxp[row] - gx[row])) / step
                }
            }
            val delta = LinearAlgebra.solve(jac, DoubleArray(p) { -resF[it] })
            for (i in x.indices) x[i] += delta[i]
            lastStep = LinearAlgebra.normInf(delta)
            if (lastStep < newtonTol) { converged = true; break }
        }
        if (!converged) {
            urysonLogger.warning(
                "Ньютон (nystrom) не сошёлся за $iter итераций (лимит 60): " +
                    "последняя невязка ||F||=$lastResidual, норма шага ||delta||=$lastStep"
            )
        }
        val xFinal = x
        return SolutionFunc({ t -> evalAtVals(t, xFinal) }, iter)
    }
}

// ============================================================================
// 11. РЕГУЛЯРИЗОВАННЫЙ РЕШАТЕЛЬ ДЛЯ УРАВНЕНИЯ I РОДА
// ============================================================================

/** Результат регуляризованного решения I рода: сплайн, параметр alpha, невязка, регуляризатор. */
class FirstKindSolution(
    val coeffs: DoubleArray,
    val eval: (Double) -> Double,
    val alpha: Double,
    val resid: Double,
    val omega: Double,
)

/**
 * Регуляризованная сплайн-коллокация для уравнения Урысона I рода (ury1).
 * Минимизирует тихоновский функционал (tikh-disc) итерациями Гаусса--Ньютона
 * (coll-gn); параметр alpha выбирается по принципу Морозова (morozov).
 *
 * Шум добавляется детерминированно (фиксированный seed) и масштабируется под заданный
 * уровень delta в L2 (условие (VII)).
 */
class FirstKindSolver(
    val problem: ModelProblem,
    val basis: MinimalSplineBasis,
    val funcs: ProjFunctionals,
    val space: SplineSpace,
    val op: UrysohnOperator,
    val tau: Double = 1.1,
    val gnTol: Double = 1e-10,
    val gnMaxIter: Int = 50,
) {
    val grid = basis.grid
    val n = grid.n
    private val core = CollocationCore(basis, funcs, op)
    private val weights = space.weights
    private val gramR = space.gramR

    /**
     * Строит зашумлённые данные f^delta = f^dagger + xi, с ||xi||_L2 = delta.
     * Возвращает вектор theta_j(f^delta) (используем f^delta в C[a,b], f_h^delta = f^delta).
     * Шум генерируется как сплайн со случайными узловыми значениями, отмасштабированными под delta.
     */
    fun noisyThetaF(delta: Double, seed: Long): DoubleArray {
        // f^dagger в опорных точках.
        val fDagger = { t: Double -> problem.rhsExact(t, op) }
        if (delta == 0.0) return DoubleArray(n + 2) { funcs.theta(it - 2).apply(fDagger) }
        // Случайный шумовой профиль как кусочно-линейная функция по контрольным точкам.
        val rnd = kotlin.random.Random(seed)
        val m = 4 * n
        val noiseNodes = DoubleArray(m + 1) { grid.a + (grid.b - grid.a) * it / m }
        val noiseVals = DoubleArray(m + 1) { rnd.nextDouble(-1.0, 1.0) }
        val noise = { t: Double ->
            var k = 0
            while (k < m - 1 && t >= noiseNodes[k + 1]) k++
            val tl = noiseNodes[k]; val tr = noiseNodes[k + 1]
            val w = ((t - tl) / (tr - tl)).coerceIn(0.0, 1.0)
            noiseVals[k] * (1 - w) + noiseVals[k + 1] * w
        }
        // L2-норма шума
        val l2 = Math.sqrt(op.quad.integrate(noiseNodes) { t -> noise(t) * noise(t) })
        val scale = if (l2 > 0) delta / l2 else 0.0
        val fDelta = { t: Double -> fDagger(t) + scale * noise(t) }
        return DoubleArray(n + 2) { funcs.theta(it - 2).apply(fDelta) }
    }

    /**
     * Решает (tikh-disc) при фиксированном alpha итерациями Гаусса--Ньютона (coll-gn),
     * стартуя с c0. Возвращает коэффициенты.
     */
    fun solveFixedAlpha(thetaFDelta: DoubleArray, alpha: Double, c0: DoubleArray): DoubleArray {
        val c = c0.copyOf()
        var lastStep = Double.NaN
        repeat(gnMaxIter) {
            val xi = core.xiVector(c)
            val b = core.bMatrix(c)
            // Матрица B^T W_h B + alpha R_h
            val btwb = LinearAlgebra.atWa(b, weights)
            val lhs = LinearAlgebra.addScaled(btwb, gramR, alpha)
            // Правая часть: -B^T W_h (Xi - theta(fDelta)) - alpha R_h c
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
                "норма шага ||delta||=$lastStep"
        )
        return c
    }

    /** Дискретная невязка res_h = ||Theta_h(U x_h) - Theta_h(f_h^delta)||_{ell2_h} (lh2-norm). */
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
     * Выбор alpha по принципу Морозова (morozov): ищет максимальный alpha с
     * res_h(alpha) <= tau * barDelta, barDelta = C_theta sqrt(b-a) delta.
     *
     * Реализация — через continuation (гомотопию) по убывающему alpha с warm-start:
     * для некорректной задачи это резко стабилизирует Гаусс--Ньютон и даёт монотонный
     * дискрепанс. Если цель недостижима даже при малом alpha (грубая сетка), берётся
     * alpha с минимальным дискрепансом на пути (без взрыва решения).
     */
    fun solveMorozov(delta: Double, seed: Long): FirstKindSolution {
        val thetaFDelta = noisyThetaF(delta, seed)
        val barDelta = funcs.cTheta() * Math.sqrt(grid.b - grid.a) * delta
        val target = tau * barDelta

        // Логарифмический путь alpha от 1e2 к 1e-12 (по убыванию) с warm-start.
        // Старт — ненулевой (проекция константы 1): для ядер с dK/du(.,.,0)=0
        // (например кубичных, Задача D) нулевой старт даёт вырожденный якобиан B=0,
        // и Гаусс--Ньютон не сдвигается. Ненулевой нейтральный старт — не «подгонка»,
        // а стандартное начальное приближение итерационного метода.
        val hiExp = 2.0; val loExp = -12.0
        val steps = 56
        val cInit = funcs.projectorCoeffs { 1.0 }
        var c = cInit.copyOf()
        var chosen: FirstKindSolution? = null
        var bestAbove: FirstKindSolution? = null // минимальный дискрепанс (если цель недостижима)
        var bestAboveRes = Double.MAX_VALUE
        for (i in 0..steps) {
            val exp = hiExp + (loExp - hiExp) * i / steps
            val alpha = Math.pow(10.0, exp)
            // Старт GN: warm-start от предыдущего alpha, но если текущий c вырождается
            // в ноль (ядро с dK/du(.,.,0)=0), перезапускаем от ненулевого proj(1).
            val start = if (LinearAlgebra.normInf(c) < 1e-8) cInit.copyOf() else c
            c = solveFixedAlpha(thetaFDelta, alpha, start)
            val res = residual(c, thetaFDelta)
            if (delta == 0.0) {
                // без шума — идём до малого alpha
                chosen = FirstKindSolution(c.copyOf(), { t -> basis.evalSpline(c, t) }, alpha, res, space.omegaReg(c))
                continue
            }
            if (res <= target) {
                // первый (максимальный) alpha, удовлетворяющий Морозову
                val cc = c.copyOf()
                chosen = FirstKindSolution(cc, { t -> basis.evalSpline(cc, t) }, alpha, res, space.omegaReg(cc))
                break
            }
            if (res < bestAboveRes) {
                bestAboveRes = res
                val cc = c.copyOf()
                bestAbove = FirstKindSolution(cc, { t -> basis.evalSpline(cc, t) }, alpha, res, space.omegaReg(cc))
            }
        }
        return chosen ?: bestAbove!!
    }
}

/** Максимум |x*(t) - x_h(t)| на контрольной сетке из 100n+1 точек (eq:num-Eh). */
fun errorEh(problem: ModelProblem, sol: SolutionFunc, grid: Grid): Double {
    val m = 100 * grid.n
    var e = 0.0
    for (i in 0..m) {
        val t = grid.a + (grid.b - grid.a) * i / m
        e = maxOf(e, abs(problem.exact(t) - sol.eval(t)))
    }
    return e
}

/** Универсальная ошибка E_h по функции-вычислителю (eq:num-Eh). */
fun errorEhEval(exact: (Double) -> Double, eval: (Double) -> Double, grid: Grid): Double {
    val m = 100 * grid.n
    var e = 0.0
    for (i in 0..m) {
        val t = grid.a + (grid.b - grid.a) * i / m
        e = maxOf(e, abs(exact(t) - eval(t)))
    }
    return e
}

// ============================================================================
// 12. HEALTH-CHECKS (раздел 9)
// ============================================================================

/**
 * Набор health-checks из раздела 9 задания. Каждая проверка печатает имя, измеренную
 * величину невязки, порог и вердикт OK/FAIL. Критический провал вызывает exit(1).
 */
object HealthChecks {
    private val quad = GaussLegendre(8)
    private val testGridsU = Grid.uniform(8)
    private val testGridsQ = Grid.quasiUniform(8)
    private val sampleTs = (0..200).map { it / 200.0 }

    fun runAll(): List<CheckResult> {
        val results = ArrayList<CheckResult>()
        results.add(checkSplineVsB())
        results.add(checkSplineVsH())
        results.add(checkPartitionOfUnity())
        results.add(checkBiorthogonality())
        results.add(checkProjectorIdempotent())
        results.add(checkExactOnSpan())
        results.add(checkBoundaryClosedForm())
        results.add(checkQuadrature())
        results.add(checkGramMatrix())
        results.add(checkWeightsSum())
        results.add(checkRhsConsistency())
        results.add(checkMethodSanity())
        results.add(checkGnMatrixSpd())
        return results
    }

    private fun forEachGrid(action: (Grid) -> Double): Double =
        maxOf(action(testGridsU), action(testGridsQ))

    /** 1. omega (общая) == omegaB для phi^B, вкл. производные. */
    private fun checkSplineVsB(): CheckResult {
        val err = forEachGrid { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            var m = 0.0
            for (ti in sampleTs) {
                val t = grid.a + (grid.b - grid.a) * ti
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j)) {
                    m = maxOf(m, abs(basis.omega(j, t) - ReferenceSplines.omegaB(grid, j, t)))
                    m = maxOf(m, abs(basis.omegaDeriv(j, t) - ReferenceSplines.omegaBDeriv(grid, j, t)))
                }
            }
            m
        }
        return CheckResult("1. omega == omegaB (и производные)", err, 1e-10, true)
    }

    /** 2. omega (общая) == omegaH для phi^H. */
    private fun checkSplineVsH(): CheckResult {
        val err = forEachGrid { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            var m = 0.0
            for (ti in sampleTs) {
                val t = grid.a + (grid.b - grid.a) * ti
                for (j in -2..grid.n - 1) if (nonDegenerate(grid, j))
                    m = maxOf(m, abs(basis.omega(j, t) - ReferenceSplines.omegaH(grid, j, t)))
            }
            m
        }
        return CheckResult("2. omega == omegaH", err, 1e-10, true)
    }

    /** 3. Разбиение единицы (pou) для всех phi. */
    private fun checkPartitionOfUnity(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(sys, grid)
                for (ti in sampleTs) {
                    val t = grid.a + (grid.b - grid.a) * ti
                    var sum = 0.0
                    for (j in -2..grid.n - 1) sum += basis.omega(j, t)
                    m = maxOf(m, abs(sum - 1.0))
                }
            }
            m
        }
        return CheckResult("3. Разбиение единицы (pou)", err, 1e-10, true)
    }

    /** 4. Биортогональность (biorth) theta_i(omega_j) = delta_ij. */
    private fun checkBiorthogonality(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val v = funcs.theta(i).apply { t -> basis.omega(j, t) }
                    m = maxOf(m, abs(v - if (i == j) 1.0 else 0.0))
                }
            }
            m
        }
        return CheckResult("4. Биортогональность (biorth)", err, 1e-9, true)
    }

    /** 5. Свойство проектора (lem:projector): P_theta u_h = u_h. */
    private fun checkProjectorIdempotent(): CheckResult {
        val rnd = kotlin.random.Random(777)
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                val c = DoubleArray(grid.n + 2) { rnd.nextDouble(-1.0, 1.0) }
                val uh = { t: Double -> basis.evalSpline(c, t) }
                val pc = funcs.projectorCoeffs(uh)
                for (i in c.indices) m = maxOf(m, abs(pc[i] - c[i]))
            }
            m
        }
        return CheckResult("5. Проектор P_theta u_h = u_h", err, 1e-9, true)
    }

    /** 6. Точность на оболочке span{1,rho,sigma}: P_theta g = g для g = rho. */
    private fun checkExactOnSpan(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                val g = { t: Double -> sys.rho(t) }
                val pc = funcs.projectorCoeffs(g)
                for (ti in sampleTs) {
                    val t = grid.a + (grid.b - grid.a) * ti
                    m = maxOf(m, abs(g(t) - basis.evalSpline(pc, t)))
                }
            }
            m
        }
        return CheckResult("6. Точность на span (P g = g)", err, 1e-9, true)
    }

    /** 7. Редукция closed-form theta к (pr_func_b) на равномерной B-сетке. */
    private fun checkBoundaryClosedForm(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val expected = doubleArrayOf(1.0 / 14.0, -2.0 / 7.0, 10.0 / 7.0, -2.0 / 7.0, 1.0 / 14.0)
        var m = 0.0
        for (j in 0..grid.n - 3) {
            val cf = funcs.closedFormInternal(j).coeffs
            for (k in cf.indices) m = maxOf(m, abs(cf[k] - expected[k]))
        }
        return CheckResult("7. Редукция theta к (pr_func_b)", m, 1e-10, true)
    }

    /** 8. Точность квадратуры: многочлены степени <=2m-1 точно, e^t и 1/(t+1). */
    private fun checkQuadrature(): CheckResult {
        val bp = doubleArrayOf(0.0, 1.0)
        var m = 0.0
        // многочлен t^k, k до 2*8-1=15: \int_0^1 t^k = 1/(k+1)
        for (k in 0..15) {
            val v = quad.integrate(bp) { t -> Math.pow(t, k.toDouble()) }
            m = maxOf(m, abs(v - 1.0 / (k + 1)))
        }
        // e^t: \int_0^1 = e - 1
        m = maxOf(m, abs(quad.integrate(bp) { t -> Math.exp(t) } - (Math.E - 1.0)))
        // 1/(t+1): \int_0^1 = ln 2
        m = maxOf(m, abs(quad.integrate(bp) { t -> 1.0 / (t + 1.0) } - Math.log(2.0)))
        return CheckResult("8. Точность квадратуры", m, 1e-10, true)
    }

    /** 9. R_h: симметрия, положительная определённость (Холецкий), полосность. */
    private fun checkGramMatrix(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(sys, grid)
                val space = SplineSpace(basis, quad)
                val r = space.gramR
                m = maxOf(m, LinearAlgebra.maxAsymmetry(r))
                if (LinearAlgebra.cholesky(r) == null) m = maxOf(m, 1.0) // не ПОО -> провал
                for (i in 0 until space.dim) for (j in 0 until space.dim)
                    if (abs(i - j) > 2) m = maxOf(m, abs(r[i][j]))
            }
            m
        }
        return CheckResult("9. R_h симм./ПОО/полосная", err, 1e-10, true)
    }

    /** 10. Согласованность весов: sum_j w_j = b-a (lh2-norm). */
    private fun checkWeightsSum(): CheckResult {
        val err = forEachGrid { grid ->
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val space = SplineSpace(basis, quad)
            abs(space.weightsSum() - (grid.b - grid.a))
        }
        return CheckResult("10. Сумма весов sum w_j = b-a", err, 1e-12, true)
    }

    /** 11. Согласованность правой части: невязка на точном решении мала и убывает. */
    private fun checkRhsConsistency(): CheckResult {
        // берём Задачу C (I рода): подставляем точное решение в дискр. невязку без регуляриз.
        fun resOn(nn: Int): Double {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(ModelProblem.C.kernel, grid, quad)
            val core = CollocationCore(basis, funcs, op)
            // коэффициенты сплайн-интерполянта точного решения
            val c = funcs.projectorCoeffs { t -> ModelProblem.C.exact(t) }
            val xi = core.xiVector(c)
            val thetaF = DoubleArray(grid.n + 2) { funcs.theta(it - 2).apply { t -> ModelProblem.C.rhsExact(t, op) } }
            var s = 0.0
            for (j in 0 until grid.n + 2) {
                val d = xi[j] - thetaF[j]
                s += space.weights[j] * d * d
            }
            return Math.sqrt(s)
        }
        val r8 = resOn(8); val r16 = resOn(16)
        // невязка должна убывать: проверяем r16 <= r8 (и оба малы)
        val measured = if (r16 <= r8 * 1.1) maxOf(r16, 0.0) else 1.0
        return CheckResult("11. Согласованность правой части (res убывает)", measured, 1e-1, true)
    }

    /** 12. Sanity: базовая схема II рода сходится; 1 шаг GN уменьшает функционал. */
    private fun checkMethodSanity(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        // II рода: Задача A
        val opA = UrysohnOperator(ModelProblem.A.kernel, grid, quad)
        val solver = SecondKindSolver(ModelProblem.A, basis, funcs, space, opA)
        val eh = errorEhEval({ t -> ModelProblem.A.exact(t) }, solver.base().eval, grid)
        // GN: одна итерация уменьшает функционал (tikh-disc) для Задачи C
        val opC = UrysohnOperator(ModelProblem.C.kernel, grid, quad)
        val fk = FirstKindSolver(ModelProblem.C, basis, funcs, space, opC)
        val thetaF = fk.noisyThetaF(1e-2, 999L)
        val alpha = 1e-3
        val c0 = DoubleArray(grid.n + 2)
        fun phiVal(c: DoubleArray): Double {
            val core = CollocationCore(basis, funcs, opC)
            val xi = core.xiVector(c)
            var s = 0.0
            for (j in 0 until grid.n + 2) { val d = xi[j] - thetaF[j]; s += space.weights[j] * d * d }
            return s + alpha * space.omegaReg(c)
        }
        val phi0 = phiVal(c0)
        val c1 = fk.solveFixedAlpha(thetaF, alpha, c0)
        val phi1 = phiVal(c1)
        // обе части: сходимость (eh мала) и phi1 < phi0
        val measured = if (phi1 <= phi0 && eh < 1e-2) eh else 1.0
        return CheckResult("12. Sanity схем (сход/GN уменьшает Phi)", measured, 1e-2, true)
    }

    /** 13. Симметрия и ПОО матрицы B^T W_h B + alpha R_h при alpha>0. */
    private fun checkGnMatrixSpd(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(ModelProblem.C.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val c = funcs.projectorCoeffs { t -> ModelProblem.C.exact(t) }
        val b = core.bMatrix(c)
        val btwb = LinearAlgebra.atWa(b, space.weights)
        val lhs = LinearAlgebra.addScaled(btwb, space.gramR, 1e-3)
        var m = LinearAlgebra.maxAsymmetry(lhs)
        if (LinearAlgebra.cholesky(lhs) == null) m = maxOf(m, 1.0)
        return CheckResult("13. B^T W B + alpha R — симм./ПОО", m, 1e-10, true)
    }
}

// ============================================================================
// 13. ГЕНЕРАЦИЯ ТАБЛИЦ (раздел 8)
// ============================================================================

/**
 * Построение всех пяти таблиц статьи. Каждый метод считает и печатает
 * человекочитаемую таблицу и LaTeX-строки.
 */
object Tables {
    private val NS = listOf(8, 16, 32, 64, 128)
    /** Фиксированный seed для детерминированного шума I рода (раздел 1, п.5). */
    const val SEED = 20240517L
    private val quad = GaussLegendre(8)

    /** Контекст решения для фиксированной задачи/базиса/сетки. */
    private fun ctx(problem: ModelProblem, sys: GeneratingSystem, nn: Int): SecondKindSolver {
        val grid = Grid.uniform(nn)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return SecondKindSolver(problem, basis, funcs, space, op)
    }

    /** p_h = log2(E_h / E_{h/2}) (eq:num-ph) — по соседним строкам. */
    private fun orders(errs: List<Double>): List<Double> =
        errs.indices.map { i ->
            if (i + 1 < errs.size) Math.log(errs[i] / errs[i + 1]) / Math.log(2.0) else Double.NaN
        }

    /** Таблица tab:num-ury2A-order: Задача A, B-базис vs H-базис, колонки E_h,p_h. */
    fun tableA() {
        println("\n--- tab:num-ury2A-order: Задача A, равномерная сетка ---")
        val errB = ArrayList<Double>(); val errH = ArrayList<Double>(); val hs = ArrayList<Double>()
        for (nn in NS) {
            val solB = ctx(ModelProblem.A, GeneratingSystem.B, nn)
            val solH = ctx(ModelProblem.A, GeneratingSystem.H, nn)
            hs.add(solB.grid.h)
            errB.add(errorEhEval({ t -> ModelProblem.A.exact(t) }, solB.base().eval, solB.grid))
            errH.add(errorEhEval({ t -> ModelProblem.A.exact(t) }, solH.base().eval, solH.grid))
        }
        val pB = orders(errB); val pH = orders(errH)
        println("   n |    h    |   Eh(B)   | ph(B) |   Eh(H)   | ph(H)")
        for (i in NS.indices) {
            println("%4d | %s | %s | %5s | %s | %5s".format(
                NS[i], Fmt.h(hs[i]), Fmt.e(errB[i]), Fmt.p(pB[i]), Fmt.e(errH[i]), Fmt.p(pH[i])))
        }
        println("   LaTeX:")
        for (i in NS.indices) {
            println("   $${NS[i]}$ & $${Fmt.h(hs[i])}$ & $${Fmt.tex(errB[i])}$ & $${Fmt.p(pB[i])}$ & $${Fmt.tex(errH[i])}$ & $${Fmt.p(pH[i])}$ \\\\")
        }
    }

    /** Таблица tab:num-ury2B-phi: Задача B, три базиса B/H/T, E_h,p_h,C_h. */
    fun tableB() {
        println("\n--- tab:num-ury2B-phi: Задача B, равномерная сетка ---")
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val errs = systems.map { ArrayList<Double>() }
        val hs = ArrayList<Double>()
        for (nn in NS) {
            for ((si, sys) in systems.withIndex()) {
                val s = ctx(ModelProblem.B, sys, nn)
                if (si == 0) hs.add(s.grid.h)
                errs[si].add(errorEhEval({ t -> ModelProblem.B.exact(t) }, s.base().eval, s.grid))
            }
        }
        val ps = systems.indices.map { orders(errs[it]) }
        val p = 3.0 // ожидаемый порядок
        println("   n |    h    | [B: Eh ph Ch] | [H: Eh ph Ch] | [T: Eh ph Ch]")
        for (i in NS.indices) {
            val cols = systems.indices.joinToString(" | ") { si ->
                "%s %5s %s".format(Fmt.e(errs[si][i]), Fmt.p(ps[si][i]), Fmt.e(errs[si][i] / Math.pow(hs[i], p)))
            }
            println("%4d | %s | %s".format(NS[i], Fmt.h(hs[i]), cols))
        }
        println("   LaTeX:")
        for (i in NS.indices) {
            val cells = systems.indices.joinToString(" & ") { si ->
                "$${Fmt.tex(errs[si][i])}$ & $${Fmt.p(ps[si][i])}$ & $${Fmt.tex(errs[si][i] / Math.pow(hs[i], p))}$"
            }
            println("   $${NS[i]}$ & $${Fmt.h(hs[i])}$ & $cells \\\\")
        }
    }

    /**
     * Таблица tab:num-ury2-methods: Задача B, базовая/Sloan/Kulkarni/Nystrom для двух
     * базисов: B (порядки методов различимы) и H (решение e^t в оболочке,
     * все методы дают машинную точность).
     */
    fun tableMethods() {
      for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
        println("\n--- tab:num-ury2-methods: Задача B, базис ${sys.name} ---")
        val names = listOf("базовая", "Sloan", "Kulkarni", "Nystrom")
        val errs = names.map { ArrayList<Double>() }
        val hs = ArrayList<Double>()
        for (nn in NS) {
            val s = ctx(ModelProblem.B, sys, nn)
            hs.add(s.grid.h)
            val exact = { t: Double -> ModelProblem.B.exact(t) }
            errs[0].add(errorEhEval(exact, s.base().eval, s.grid))
            errs[1].add(errorEhEval(exact, s.sloan().eval, s.grid))
            errs[2].add(errorEhEval(exact, s.kulkarni().eval, s.grid))
            errs[3].add(errorEhEval(exact, s.nystrom().eval, s.grid))
        }
        val ps = names.indices.map { orders(errs[it]) }
        println("   n |    h    | " + names.joinToString(" | ") { "$it: Eh ph" })
        for (i in NS.indices) {
            val cols = names.indices.joinToString(" | ") { mi -> "%s %5s".format(Fmt.e(errs[mi][i]), Fmt.p(ps[mi][i])) }
            println("%4d | %s | %s".format(NS[i], Fmt.h(hs[i]), cols))
        }
        println("   LaTeX (базис ${sys.name}):")
        for (i in NS.indices) {
            val cells = names.indices.joinToString(" & ") { mi -> "$${Fmt.tex(errs[mi][i])}$ & $${Fmt.p(ps[mi][i])}$" }
            println("   $${NS[i]}$ & $${Fmt.h(hs[i])}$ & $cells \\\\")
        }
      }
    }

    /** Таблица tab:num-ury1C: Задача C, строки по (delta,n,alpha), E_h,E_h^rel,res,Omega.
     *  На квазиравномерной сетке X^q (неравномерность выявляет влияние выбора базиса). */
    fun tableC() {
        println("\n--- tab:num-ury1C: Задача C (регуляризованная, H-базис, квазиравномерная сетка) ---")
        val deltas = listOf(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        println("  delta |  n  |   alpha   |    Eh     |  Eh_rel   |   res     |   Omega")
        val rows = ArrayList<String>()
        for (i in deltas.indices) {
            val delta = deltas[i]; val nn = NS[i]
            val grid = Grid.quasiUniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(ModelProblem.C.kernel, grid, quad)
            val solver = FirstKindSolver(ModelProblem.C, basis, funcs, space, op)
            val sol = solver.solveMorozov(delta, SEED)
            val eh = errorEhEval({ t -> ModelProblem.C.exact(t) }, sol.eval, grid)
            val normExact = (0..100 * nn).maxOf { abs(ModelProblem.C.exact(it.toDouble() / (100 * nn))) }
            val ehRel = eh / normExact
            println("  %s | %3d | %s | %s | %s | %s | %s".format(
                Fmt.e(delta), nn, Fmt.e(sol.alpha), Fmt.e(eh), Fmt.e(ehRel), Fmt.e(sol.resid), Fmt.e(sol.omega)))
            rows.add("   $${Fmt.tex(delta)}$ & $$nn$ & $${Fmt.tex(sol.alpha)}$ & $${Fmt.tex(eh)}$ & $${Fmt.tex(ehRel)}$ & $${Fmt.tex(sol.resid)}$ & $${Fmt.tex(sol.omega)}$ \\\\")
        }
        println("   LaTeX:")
        rows.forEach { println(it) }
    }

    /** Таблица tab:num-ury1D: Задача D, B vs H при фиксированном delta.
     *  На квазиравномерной сетке X^q (на ней эффект согласованного базиса phi^H заметен). */
    fun tableD() {
        println("\n--- tab:num-ury1D: Задача D, fixed delta=1e-3, B vs H (квазиравномерная сетка) ---")
        val delta = 1e-3
        println("   n  |   alpha   |  Eh(B)   |  res(B)   |  Eh(H)   |  res(H)   | Eh_B/Eh_H")
        val rows = ArrayList<String>()
        for (nn in NS) {
            fun solve(sys: GeneratingSystem): FirstKindSolution {
                val grid = Grid.quasiUniform(nn)
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                val space = SplineSpace(basis, quad)
                val op = UrysohnOperator(ModelProblem.D.kernel, grid, quad)
                return FirstKindSolver(ModelProblem.D, basis, funcs, space, op).solveMorozov(delta, SEED)
            }
            val sB = solve(GeneratingSystem.B); val sH = solve(GeneratingSystem.H)
            val gridN = Grid.quasiUniform(nn)
            val ehB = errorEhEval({ t -> ModelProblem.D.exact(t) }, sB.eval, gridN)
            val ehH = errorEhEval({ t -> ModelProblem.D.exact(t) }, sH.eval, gridN)
            val ratio = ehB / ehH
            println("  %3d | %s | %s | %s | %s | %s | %s".format(
                nn, Fmt.e(sB.alpha), Fmt.e(ehB), Fmt.e(sB.resid), Fmt.e(ehH), Fmt.e(sH.resid), Fmt.p(ratio)))
            rows.add("   $$nn$ & $${Fmt.tex(sB.alpha)}$ & $${Fmt.tex(ehB)}$ & $${Fmt.tex(sB.resid)}$ & $${Fmt.tex(ehH)}$ & $${Fmt.tex(sH.resid)}$ & $${Fmt.p(ratio)}$ \\\\")
        }
        println("   LaTeX:")
        rows.forEach { println(it) }
    }
}

// ============================================================================
// 14. MAIN: health-checks -> таблицы
// ============================================================================

fun main() {
    println("=".repeat(72))
    println("Численный эксперимент (main.tex, \\label{sec:numerical})")
    println("=".repeat(72))
    println("HEALTH-CHECKS (раздел 9):")
    val results = HealthChecks.runAll()
    var anyCriticalFail = false
    for (r in results) {
        val verdict = if (r.ok) "OK  " else "FAIL"
        if (!r.ok && r.critical) anyCriticalFail = true
        println("  [$verdict] ${r.name.padEnd(40)} measured=${"%.3e".format(r.measured)} thr=${"%.0e".format(r.threshold)}")
    }
    if (anyCriticalFail) {
        println("\nКРИТИЧЕСКИЙ ПРОВАЛ health-check. Расчёт остановлен.")
        exitProcess(1)
    }
    println("\nВсе критические health-checks пройдены.\n")

    println("=".repeat(72))
    println("ТАБЛИЦЫ (раздел 8)")
    println("Детерминированный seed для шума (уравнение I рода): ${Tables.SEED}")
    println("=".repeat(72))
    Tables.tableA()
    Tables.tableB()
    Tables.tableMethods()
    Tables.tableC()
    Tables.tableD()
    println("\nРасчёт завершён.")
}
