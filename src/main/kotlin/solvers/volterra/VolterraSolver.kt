package solvers.volterra

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.system.exitProcess
import numerics.*
import numerics.functionals.*

// ============================================================================
// 7. ЯДРО И ОПЕРАТОР ВОЛЬТЕРРА \mathcal V u(t) = \int_a^t K(t,s) u(s) ds
// ============================================================================

/** Ядро K(t,s) линейного уравнения Вольтерра и (при необходимости для xi) dK/dt. */
class KernelV(val k: (Double, Double) -> Double, val kT: (Double, Double) -> Double = { _, _ -> 0.0 })

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
    val a = grid.a
    val b = grid.b

    /** Составное разбиение [a, t]: внутренние узлы сетки < t, затем сам t. */
    private fun subBreakpoints(t: Double): DoubleArray {
        val bp = grid.breakpoints
        val list = ArrayList<Double>()
        for (x in bp) { if (x < t - 1e-15) list.add(x) else break }
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
}

// ============================================================================
// 8. МОДЕЛЬНЫЕ ЗАДАЧИ
// ============================================================================

/**
 * Модельная задача: ядро, точное решение, род. Правая часть строится из точного
 * решения численно (квадратурой), без зашивания.
 * II рода: f = u* - \mathcal K u*;  I рода: f = \mathcal K u*.
 */
class ModelProblem(
    val name: String,
    val kernel: KernelV,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val a: Double = 0.0,
    val b: Double = 1.0,
) {
    /** Точная правая часть f(t). */
    fun rhsExact(t: Double, op: VolterraOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /** d/dt f(t) (для xi-функционалов): II рода u*' - d/dt \mathcal K u*. */
    fun rhsExactDeriv(t: Double, op: VolterraOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    companion object {
        /** V2span: K=e^{t-2s}, u*=t^2 ∈ span{1,t,t^2}=phi^B (машинная точность на B). */
        val V2span = ModelProblem(
            name = "V2span",
            kernel = KernelV({ t, s -> Math.exp(t - 2.0 * s) }, { t, s -> Math.exp(t - 2.0 * s) }),
            exact = { t -> t * t }, exactDeriv = { t -> 2.0 * t },
            secondKind = true,
        )

        /** V2: K=1/(1+t+s), u*=1/(t+1) (проверка порядка на полиномиальном базисе). */
        val V2 = ModelProblem(
            name = "V2",
            kernel = KernelV({ t, s -> 1.0 / (1.0 + t + s) }, { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) }),
            exact = { t -> 1.0 / (t + 1.0) }, exactDeriv = { t -> -1.0 / ((t + 1.0) * (t + 1.0)) },
            secondKind = true,
        )

        /** V2exp: K=e^{-(t-s)^2}, u*=e^t (согласован с phi^H). */
        val V2exp = ModelProblem(
            name = "V2exp",
            kernel = KernelV({ t, s -> Math.exp(-(t - s) * (t - s)) },
                { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) }),
            exact = { t -> Math.exp(t) }, exactDeriv = { t -> Math.exp(t) },
            secondKind = true,
        )

        /**
         * V2win: K=t-s (K(t,t)=0 — слабое/сглаживающее ядро), u*=cos t.
         * Разведочный пример (tests/VolterraKernelSearch.kt), где ОДНОШАГОВЫЙ Кулкарни
         * СТРОГО точнее Слоана (на ~2 порядка при n=32): p_h база≈3, Слоан≈4,
         * Кулкарни≈5, итер.Кулкарни≈6. Причина: K(t,t)=0 усиливает сглаживание \mathcal V,
         * и член суперсходимости (I-P)\mathcal V(I-P) работает в полную силу.
         */
        val V2win = ModelProblem(
            name = "V2win",
            kernel = KernelV({ t, s -> t - s }, { _, _ -> 1.0 }),
            exact = { t -> Math.cos(t) }, exactDeriv = { t -> -Math.sin(t) },
            secondKind = true,
        )

        /**
         * V1: K=1+t-s (K(t,t)=1≠0), u*=cos t, уравнение I рода \mathcal V u = f.
         * Корректна (r3): сводится к V2 однократным дифференцированием (m=1, т.к. K(t,t)≠0).
         */
        val V1 = ModelProblem(
            name = "V1",
            kernel = KernelV({ t, s -> 1.0 + t - s }, { _, _ -> 1.0 }),
            exact = { t -> Math.cos(t) }, exactDeriv = { t -> -Math.sin(t) },
            secondKind = false,
        )
    }
}
// ============================================================================
// 9. ЯДРО КОЛЛОКАЦИИ + РЕШАТЕЛЬ УРАВНЕНИЯ II РОДА (ЛИНЕЙНО)
// ============================================================================

/** Результат решения: вычислитель u_h(t). */
class SolutionFunc(val eval: (Double) -> Double)

/**
 * Линейный решатель уравнения II рода u - L u = f, L = c_L * \mathcal K
 * (c_L = 1 для F2; c_L = -1/alpha для F1 Wazwaz, \mathcal K_eff = -(1/alpha)\mathcal K).
 * Правая часть f и её производная задаются явно (fEff/fEffDeriv) для переиспользования в F1.
 *
 * Матрицы (discrete-problem.md):
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 */
class SecondKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    val op: VolterraOperator,
    val cL: Double,
    val fEff: (Double) -> Double,
    val fEffDeriv: (Double) -> Double,
) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    // ВНИМАНИЕ (отличие от Фредгольма): у оператора Вольтерра область интегрирования
    // [a,t] зависит от t, поэтому предвычисление на фиксированных узлах невозможно.
    // Все применения L = c_L \mathcal V выражаются через замыкания op.apply / op.applyDeriv.

    /** L g(t) = c_L (\mathcal V g)(t) и её производная (Лейбниц). */
    private fun applyL(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.apply(t, g) }
    private fun applyLDeriv(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.applyDeriv(t, g) }

    /** chi_j(g) по значениям g и g' (обёртка). */
    private fun chiOf(g: (Double) -> Double, gD: (Double) -> Double): DoubleArray =
        DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD) }

    /** Матрица M_{j,i} = chi_j(L omega_i). Для xi учитывается производная (L omega_i)'. */
    fun matrixM(): Array<DoubleArray> {
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) {
            val idx = i - 2
            val omega = { s: Double -> basis.omega(idx, s) }
            val Lom = applyL(omega)
            val LomD = applyLDeriv(omega)
            for (j in 0 until dim) m[j][i] = funcs.chi(j - 2).apply(Lom, LomD)
        }
        return m
    }

    /** Матрица M2_{j,i} = chi_j(L(L omega_i)) (двойное применение L). */
    fun matrixM2(): Array<DoubleArray> {
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) {
            val idx = i - 2
            val Lom = applyL { s -> basis.omega(idx, s) }
            val LLom = applyL(Lom)
            val LLomD = applyLDeriv(Lom)
            for (j in 0 until dim) m[j][i] = funcs.chi(j - 2).apply(LLom, LLomD)
        }
        return m
    }

    /** g_j = chi_j(f). */
    fun vectorG(): DoubleArray = chiOf(fEff, fEffDeriv)

    /** d_j = chi_j(L f). */
    fun vectorD(): DoubleArray {
        val Lf = { t: Double -> cL * op.apply(t) { s -> fEff(s) } }
        val LfD = { t: Double -> cL * op.applyDeriv(t) { s -> fEff(s) } }
        return chiOf(Lf, LfD)
    }

    /** Базовая схема: (I - M) c = g. */
    fun solveBaseCoeffs(): DoubleArray {
        val m = matrixM()
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) { for (c in 0 until dim) a[r][c] = -m[r][c]; a[r][r] += 1.0 }
        return LinearAlgebra.solve(a, vectorG())
    }

    fun base(): SolutionFunc {
        val c = solveBaseCoeffs()
        return SolutionFunc { t -> basis.evalSpline(c, t) }
    }

    /** Слоан: ~u_h(t) = f(t) + (L u_h)(t). u_h — сплайн, L применяется замыканием. */
    fun sloan(): SolutionFunc {
        val c = solveBaseCoeffs()
        val Luh = applyL { s -> basis.evalSpline(c, s) }
        return SolutionFunc { t -> fEff(t) + Luh(t) }
    }

    /**
     * Кулкарни для проекторов (theta, xi): (I - M - M2 + M^2) c = (I - M) g + d;
     * u_h^K = y_h + (I - P_chi)[f + L y_h]. Для квазиинтерполянтов (mu, lambda) без
     * редукции — прямая итерация конечноранговым U^K_h [численное наблюдение].
     */
    fun kulkarni(): SolutionFunc {
        return if (funcs.isProjector) kulkarniProjector() else kulkarniQuasi()
    }

    private fun kulkarniProjector(): SolutionFunc {
        val m = matrixM(); val m2 = matrixM2(); val g = vectorG(); val d = vectorD()
        val mm = LinearAlgebra.matMat(m, m)
        // A = I - M - M2 + M^2
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) {
            for (c in 0 until dim) a[r][c] = -m[r][c] - m2[r][c] + mm[r][c]
            a[r][r] += 1.0
        }
        // rhs = (I - M) g + d
        val mg = LinearAlgebra.matVec(m, g)
        val rhs = DoubleArray(dim) { g[it] - mg[it] + d[it] }
        val c = LinearAlgebra.solve(a, rhs) // коэффициенты y_h
        // u_h^K = y_h + (I - P_chi)[f + L y_h]; (I - P_chi)w = w - P_chi w.
        val yh = { s: Double -> basis.evalSpline(c, s) }
        val Lyh = applyL(yh)
        val LyhD = applyLDeriv(yh)
        val wFun = { t: Double -> fEff(t) + Lyh(t) }
        val wDFun = { t: Double -> fEffDeriv(t) + LyhD(t) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun)
        return SolutionFunc { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) }
    }

    /**
     * Кулкарни для mu, lambda: итерация u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)). Работа в узлах квадратуры.
     * [численное наблюдение]: разрешимость/сходимость не гарантированы (нет P^2=P).
     */
    private val sampleXs: DoubleArray = DoubleArray(20 * n + 1) { grid.a + (grid.b - grid.a) * it / (20 * n) }

    private fun kulkarniQuasi(): SolutionFunc {
        var uVals = DoubleArray(sampleXs.size) { fEff(sampleXs[it]) }
        repeat(200) {
            val cur = uVals
            val uFun = { t: Double -> evalNodalLinear(t, cur) }
            val pc = funcs.projectorCoeffs(uFun)
            val pcFun = { s: Double -> basis.evalSpline(pc, s) }
            val LuFun = applyL(uFun)                       // L u
            val pLu = funcs.projectorCoeffs(LuFun)        // P_chi(L u)
            val LPu = applyL(pcFun)                        // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(LPu)         // P_chi(L(P_chi u))
            val next = DoubleArray(sampleXs.size) { k ->
                val t = sampleXs[k]
                fEff(t) + basis.evalSpline(pLu, t) + LPu(t) - basis.evalSpline(pLPu, t)
            }
            var diff = 0.0
            for (k in next.indices) diff = maxOf(diff, abs(next[k] - cur[k]))
            uVals = next
            if (diff < 1e-12) return@repeat
        }
        val finalVals = uVals
        return SolutionFunc { t -> evalNodalLinear(t, finalVals) }
    }

    /** Кусочно-линейное восстановление по значениям на sampleXs (для mu/lambda). */
    private fun evalNodalLinear(t: Double, nodes: DoubleArray): Double {
        val xs = sampleXs
        if (t <= xs[0]) return nodes[0]
        if (t >= xs[xs.size - 1]) return nodes[xs.size - 1]
        var lo = 0; var hi = xs.size - 1
        while (hi - lo > 1) { val mid = (lo + hi) / 2; if (xs[mid] <= t) lo = mid else hi = mid }
        val w = (t - xs[lo]) / (xs[hi] - xs[lo])
        return nodes[lo] * (1 - w) + nodes[hi] * w
    }

    /** Итерированный Кулкарни: ^u_h^K = f + L u_h^K. */
    fun iteratedKulkarni(): SolutionFunc {
        val uK = kulkarni()
        val Luk = applyL { s -> uK.eval(s) }
        return SolutionFunc { t -> fEff(t) + Luk(t) }
    }
}
// ============================================================================
// 10. РЕШАТЕЛЬ УРАВНЕНИЯ I РОДА — СВЕДЕНИЕ ДИФФЕРЕНЦИРОВАНИЕМ (r3)
// ============================================================================

/**
 * V1 — уравнение Вольтерра I рода: (\mathcal V u)(t) = \int_a^t K(t,s)u(s)ds = f(t).
 * В отличие от Фредгольма I рода (некорректного, требует Wazwaz), здесь
 * задача КОРРЕКТНА (r3) и сводится к II роду дифференцированием.
 *
 * При K(t,t) ≠ 0 (m=1) дифференцируем \mathcal V u = f по t (Лейбниц):
 *   K(t,t) u(t) + \int_a^t K_t(t,s) u(s) ds = f'(t).
 * Деля на K(t,t), получаем V2:
 *   u(t) - \mathcal W u(t) = g(t),   \mathcal W u = \int_a^t [-K_t(t,s)/K(t,t)] u(s) ds,
 *   g(t) = f'(t)/K(t,t).
 * Далее — та же схема II рода (база/Слоан/Кулкарни) с c_L = 1.
 * Производная g'(t) (для xi) и редуцированное K_t — конечные разности.
 */
class FirstKindSolver(
    val problem: ModelProblem,
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    op: VolterraOperator,
) {
    private val grid = basis.grid
    private val quad = GaussLegendre(8)
    private val Ktt = { t: Double -> problem.kernel.k(t, t) }
    // Редуцированное ядро W: K_W(t,s) = -K_t(t,s)/K(t,t); K_W_t — центральной разностью.
    private val hFD = 1e-6
    private val kernelW = KernelV(
        k = { t, s -> -problem.kernel.kT(t, s) / Ktt(t) },
        kT = { t, s ->
            val kp = -problem.kernel.kT(t + hFD, s) / Ktt(t + hFD)
            val km = -problem.kernel.kT(t - hFD, s) / Ktt(t - hFD)
            (kp - km) / (2 * hFD)
        },
    )
    private val opW = VolterraOperator(kernelW, grid, quad)
    // g(t) = f'(t)/K(t,t), f — правая часть I рода; f'(t) = d/dt \mathcal V u* (Лейбниц).
    private val gEff = { t: Double -> problem.rhsExactDeriv(t, op) / Ktt(t) }
    private val gEffDeriv = { t: Double -> (gEff(t + hFD) - gEff(t - hFD)) / (2 * hFD) }
    private val inner = SecondKindSolver(basis, funcs, opW, cL = 1.0, fEff = gEff, fEffDeriv = gEffDeriv)

    fun base(): SolutionFunc = inner.base()
    fun sloan(): SolutionFunc = inner.sloan()
    fun kulkarni(): SolutionFunc = inner.kulkarni()
    fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}

/** Фабрика решателя II рода для модельной задачи (c_L = 1, f = u* - \mathcal V u*). */
fun secondKindSolver(problem: ModelProblem, basis: MinimalSplineBasis, funcs: FunctionalFamily,
                     op: VolterraOperator): SecondKindSolver =
    SecondKindSolver(basis, funcs, op, cL = 1.0,
        fEff = { t -> problem.rhsExact(t, op) },
        fEffDeriv = { t -> problem.rhsExactDeriv(t, op) })

// ============================================================================
// 12. HEALTH-CHECKS
// ============================================================================


/**
 * Набор health-checks (миррор UrysonSolver). Критический провал -> exit(1).
 */
object HealthChecks {
    private val quad = GaussLegendre(8)
    private val gridsU = Grid.uniform(8)
    private val gridsQ = Grid.quasiUniform(8)
    private val sampleTs = (0..200).map { it / 200.0 }
    private val allSys = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)

    fun runAll(): List<CheckResult> = listOf(
        checkSplineVsB(), checkSplineVsH(), checkPartitionOfUnity(),
        checkBiorthTheta(), checkBiorthXi(), checkProjectorIdempotent(),
        checkExactOnSpan(), checkThetaReduction(), checkQuadrature(),
        checkRhsConsistency(), checkBaseSanity(),
    )

    private fun forEachGrid(action: (Grid) -> Double): Double = maxOf(action(gridsU), action(gridsQ))

    /** 1. omega == omegaB (+производные). */
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
        return CheckResult("1. omega == omegaB (и произв.)", err, 1e-10, true)
    }

    /** 2. omega == omegaH. */
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

    /** 3. Разбиение единицы. */
    private fun checkPartitionOfUnity(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
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
        return CheckResult("3. Разбиение единицы", err, 1e-10, true)
    }

    /** 4. Биортогональность theta_i(omega_j)=delta. */
    private fun checkBiorthTheta(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = ProjFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val v = funcs.chi(i).apply({ t -> basis.omega(j, t) }, { t -> basis.omegaDeriv(j, t) })
                    m = maxOf(m, abs(v - if (i == j) 1.0 else 0.0))
                }
            }
            m
        }
        return CheckResult("4. Биортогональность theta", err, 1e-9, true)
    }

    /** 5. Биортогональность xi_i(omega_j)=delta (с производной). */
    private fun checkBiorthXi(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val funcs = DeBoorFixFunctionals(basis)
                for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
                    val v = funcs.chi(i).apply({ t -> basis.omega(j, t) }, { t -> basis.omegaDeriv(j, t) })
                    m = maxOf(m, abs(v - if (i == j) 1.0 else 0.0))
                }
            }
            m
        }
        return CheckResult("5. Биортогональность xi", err, 1e-9, true)
    }

    /** 6. Идемпотентность P_chi u_h = u_h для theta, xi (mu,lambda не обязаны). */
    private fun checkProjectorIdempotent(): CheckResult {
        val rnd = kotlin.random.Random(777)
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                for (funcs in listOf(ProjFunctionals(basis), DeBoorFixFunctionals(basis))) {
                    val c = DoubleArray(grid.n + 2) { rnd.nextDouble(-1.0, 1.0) }
                    val uh = { t: Double -> basis.evalSpline(c, t) }
                    val uhD = { t: Double -> basis.evalSplineDeriv(c, t) }
                    val pc = funcs.projectorCoeffs(uh, uhD)
                    for (i in c.indices) m = maxOf(m, abs(pc[i] - c[i]))
                }
            }
            m
        }
        return CheckResult("6. Идемпотентность theta,xi", err, 1e-8, true)
    }

    /** 7. Точность на span{1,rho,sigma}: P_chi g = g для g=rho (все семейства). */
    private fun checkExactOnSpan(): CheckResult {
        val err = forEachGrid { grid ->
            var m = 0.0
            for (sys in allSys) {
                val basis = MinimalSplineBasis(sys, grid)
                val fams: List<FunctionalFamily> = listOf(
                    ProjFunctionals(basis), DeBoorFixFunctionals(basis),
                    AveragingFunctionals(basis), ThreePointFunctionals(basis),
                )
                for (funcs in fams) {
                    val pc = funcs.projectorCoeffs({ t -> sys.rho(t) }, { t -> sys.rhoD(t) })
                    for (ti in sampleTs) {
                        val t = grid.a + (grid.b - grid.a) * ti
                        m = maxOf(m, abs(sys.rho(t) - basis.evalSpline(pc, t)))
                    }
                }
            }
            m
        }
        return CheckResult("7. Точность на span (все chi)", err, 1e-8, true)
    }

    /** 8. Редукция theta к (pr_func_b) {1/14,-2/7,10/7,-2/7,1/14}. */
    private fun checkThetaReduction(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val expected = doubleArrayOf(1.0 / 14.0, -2.0 / 7.0, 10.0 / 7.0, -2.0 / 7.0, 1.0 / 14.0)
        var m = 0.0
        for (j in 0..grid.n - 3) {
            val cf = funcs.closedFormInternal(j).coeffs
            for (k in cf.indices) m = maxOf(m, abs(cf[k] - expected[k]))
        }
        return CheckResult("8. Редукция theta к (pr_func_b)", m, 1e-10, true)
    }

    /** 9. Точность квадратуры: t^k до 15, e^t, 1/(t+1). */
    private fun checkQuadrature(): CheckResult {
        val bp = doubleArrayOf(0.0, 1.0)
        var m = 0.0
        for (k in 0..15) m = maxOf(m, abs(quad.integrate(bp) { t -> Math.pow(t, k.toDouble()) } - 1.0 / (k + 1)))
        m = maxOf(m, abs(quad.integrate(bp) { t -> Math.exp(t) } - (Math.E - 1.0)))
        m = maxOf(m, abs(quad.integrate(bp) { t -> 1.0 / (t + 1.0) } - Math.log(2.0)))
        return CheckResult("9. Точность квадратуры", m, 1e-10, true)
    }

    /** 10. Согласованность правой части: на span-задаче F2span база(B) даёт машинную точность. */
    private fun checkRhsConsistency(): CheckResult {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V2span.kernel, grid, quad)
        val solver = secondKindSolver(ModelProblem.V2span, basis, funcs, op)
        val eh = errorEh({ t -> ModelProblem.V2span.exact(t) }, solver.base().eval, grid)
        return CheckResult("10. Точность на span-задаче (база B)", eh, 1e-8, true)
    }

    /** 11. Sanity: базовая схема II рода сходится (E_8 > E_16 для F2/H). */
    private fun checkBaseSanity(): CheckResult {
        fun ehAt(nn: Int): Double {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val op = VolterraOperator(ModelProblem.V2.kernel, grid, quad)
            val solver = secondKindSolver(ModelProblem.V2, basis, funcs, op)
            return errorEh({ t -> ModelProblem.V2.exact(t) }, solver.base().eval, grid)
        }
        val e8 = ehAt(8); val e16 = ehAt(16)
        val measured = if (e16 <= e8 && e8 < 1e-1) e16 else 1.0
        return CheckResult("11. Sanity базовой схемы (сходимость)", measured, 1e-1, true)
    }
}
// ============================================================================
// 13. ФОРМАТИРОВАНИЕ + ТАБЛИЦЫ + MAIN
// ============================================================================

object Tables {
    // Для Вольтерра matrixM2 = O(dim^2 * Q^2) (двойной \int_a^t), поэтому сетка ограничена n<=64.
    private val NS = listOf(8, 16, 32, 64)
    private val quad = GaussLegendre(8)

    private fun makeSolver(p: ModelProblem, sys: GeneratingSystem, fam: String, nn: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(nn)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = family(fam, basis)
        val op = VolterraOperator(p.kernel, grid, quad)
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi" -> DeBoorFixFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** T1: V2exp, базисы B/H/T, базовая схема theta: E_h, p_h, C_h. */
    fun tablePhi(p: ModelProblem) {
        println("\n--- T1[${p.name}]: theta, базисы B/H/T (E_h,p_h,C_h) ---")
        for (sys in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val errs = ArrayList<Double>(); val hs = ArrayList<Double>()
            for (nn in NS) { val (s, grid) = makeSolver(p, sys, "theta", nn)
                hs.add(grid.h); errs.add(errorEh({ t -> p.exact(t) }, s.base().eval, grid)) }
            val ps = orders(errs)
            println("  базис ${sys.name}:")
            for (i in NS.indices) println("   n=%4d h=%s E_h=%s p_h=%s C_h=%s".format(
                NS[i], Fmt.h(hs[i]), Fmt.e(errs[i]), Fmt.p(ps[i]), Fmt.e(errs[i] / Math.pow(hs[i], 3.0))))
        }
    }

    /** T2[p]: сравнение база/Слоан/Кулкарни/итер.Кулкарни (theta). */
    fun tableMethods(p: ModelProblem, sys: GeneratingSystem) {
        println("\n--- T2[${p.name}]: базис ${sys.name}, theta: база/Слоан/Кулкарни/итер.Кулкарни (E_h,p_h) ---")
        val names = listOf("база", "Слоан", "Кулк", "ит.Кулк")
        val errs = names.map { ArrayList<Double>() }
        for (nn in NS) {
            val (s, grid) = makeSolver(p, sys, "theta", nn)
            val ex = { t: Double -> p.exact(t) }
            errs[0].add(errorEh(ex, s.base().eval, grid))
            errs[1].add(errorEh(ex, s.sloan().eval, grid))
            errs[2].add(errorEh(ex, s.kulkarni().eval, grid))
            errs[3].add(errorEh(ex, s.iteratedKulkarni().eval, grid))
        }
        val ps = names.indices.map { orders(errs[it]) }
        for (i in NS.indices) println("   n=%4d | ".format(NS[i]) +
            names.indices.joinToString(" | ") { mi -> "%s:%s(%s)".format(names[mi], Fmt.e(errs[mi][i]), Fmt.p(ps[mi][i])) })
    }

    /** Сравнение семейств theta/xi/mu/lambda (базовая схема). */
    fun tableFamilies(p: ModelProblem, sys: GeneratingSystem) {
        println("\n--- Семейства[${p.name}]: базис ${sys.name}, базовая схема (E_h,p_h) ---")
        for (fam in listOf("theta", "xi", "mu", "lambda")) {
            val errs = ArrayList<Double>()
            for (nn in NS) { val (s, grid) = makeSolver(p, sys, fam, nn)
                errs.add(errorEh({ t -> p.exact(t) }, s.base().eval, grid)) }
            val ps = orders(errs)
            println("  %-7s: ".format(fam) + NS.indices.joinToString(" ") { i -> "%s(%s)".format(Fmt.e(errs[i]), Fmt.p(ps[i])) })
        }
    }

    /** V1 (сведение дифференцированием, m=1): база/Слоан/Кулкарни, базис B. */
    fun tableF1() {
        println("\n--- V1 (I рода, сведение дифференцированием m=1), базис B, theta: база/Слоан/Кулк ---")
        val p = ModelProblem.V1
        for (nn in listOf(8, 16, 32, 64)) {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = VolterraOperator(p.kernel, grid, quad)
            val solver = FirstKindSolver(p, basis, ProjFunctionals(basis), op)
            val ehB = errorEh({ t -> p.exact(t) }, solver.base().eval, grid)
            val ehS = errorEh({ t -> p.exact(t) }, solver.sloan().eval, grid)
            val ehK = errorEh({ t -> p.exact(t) }, solver.kulkarni().eval, grid)
            println("   n=%4d E_h(база)=%s E_h(Слоан)=%s E_h(Кулк)=%s".format(nn, Fmt.e(ehB), Fmt.e(ehS), Fmt.e(ehK)))
        }
        println("   [V1 корректна (r3); K(t,t)=1≠0 ⇒ m=1; g=f'/K(t,t), ядро W=-K_t/K(t,t)]")
    }
}

fun main() {
    println("=".repeat(72))
    println("VolterraSolver — линейные уравнения Вольтерра (new-01)")
    println("=".repeat(72))
    println("HEALTH-CHECKS:")
    var anyFail = false
    for (r in HealthChecks.runAll()) {
        val verdict = if (r.ok) "OK  " else "FAIL"
        if (!r.ok && r.critical) anyFail = true
        println("  [$verdict] ${r.name.padEnd(38)} measured=${"%.3e".format(r.measured)} thr=${"%.0e".format(r.threshold)}")
    }
    if (anyFail) { println("\nКРИТИЧЕСКИЙ ПРОВАЛ health-check. Расчёт остановлен."); exitProcess(1) }
    println("\nВсе критические health-checks пройдены.\n")

    println("=".repeat(72)); println("ТАБЛИЦЫ"); println("=".repeat(72))
    // Численные примеры II рода (каждый — полный набор сравнений):
    //   V2   : K=1/(1+t+s), u*=1/(t+1) (рациональный, K(t,t)!=0);
    //   V2exp: K=e^{-(t-s)^2}, u*=e^t (K(t,t)!=0);
    //   V2win: K=t-s, u*=cos t (K(t,t)=0) — одношаговый Кулкарни СТРОГО точнее Слоана.
    val secondKindExamples = listOf(
        ModelProblem.V2 to GeneratingSystem.B,
        ModelProblem.V2exp to GeneratingSystem.B,
        ModelProblem.V2win to GeneratingSystem.B,
    )
    for ((p, sys) in secondKindExamples) {
        Tables.tablePhi(p)
        Tables.tableMethods(p, sys)
        Tables.tableFamilies(p, sys)
    }
    // Один пример I рода (сведение дифференцированием).
    Tables.tableF1()
    println("\nРасчёт завершён.")
}

