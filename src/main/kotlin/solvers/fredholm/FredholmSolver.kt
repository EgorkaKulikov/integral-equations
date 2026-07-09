package solvers.fredholm

import kotlin.math.abs
import kotlin.system.exitProcess
import numerics.*
import numerics.functionals.*

// ============================================================================
// 7. ЯДРО И ОПЕРАТОР ФРЕДГОЛЬМА \mathcal K u(t) = \int_a^b K(t,s) u(s) ds
// ============================================================================

/** Ядро K(t,s) линейного уравнения Фредгольма и (при необходимости для xi) dK/dt. */
class KernelF(val k: (Double, Double) -> Double, val kT: (Double, Double) -> Double = { _, _ -> 0.0 })

/**
 * Оператор Фредгольма: (\mathcal K u)(t) = \int_a^b K(t,s) u(s) ds, квадратура по [a,b].
 * Предвычисляются глобальные гауссовы узлы gNode/gW: \int h = sum_k gW[k] h(gNode[k]).
 */
class FredholmOperator(val kernel: KernelF, val grid: Grid, val quad: GaussLegendre) {
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
            for (qi in rn.indices) { nodes.add(mid + half * rn[qi]); ws.add(half * rw[qi]) }
        }
        gNode = nodes.toDoubleArray()
        gW = ws.toDoubleArray()
    }

    /** (\mathcal K u)(t) для произвольной u(s). */
    fun apply(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s) * u(s) }

    /** d/dt (\mathcal K u)(t) = \int_a^b dK/dt(t,s) u(s) ds (для xi-функционалов). */
    fun applyDeriv(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kT(t, s) * u(s) }

    /** (\mathcal K u)(tau) по предвычисленным значениям u в глобальных узлах. */
    fun applyNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k]) * uNodes[k]
        return s
    }

    /** d/dt (\mathcal K u)(tau) по предвычисленным uNodes. */
    fun applyDerivNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kT(tau, gNode[k]) * uNodes[k]
        return s
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
    val kernel: KernelF,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val a: Double = 0.0,
    val b: Double = 1.0,
) {
    /** Точная правая часть f(t). */
    fun rhsExact(t: Double, op: FredholmOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /** d/dt f(t) (для xi-функционалов): II рода u*' - d/dt \mathcal K u*. */
    fun rhsExactDeriv(t: Double, op: FredholmOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    companion object {
        /** F2-poly: K=e^{t-2s}, u*=t^2 ∈ span{1,t,t^2}=phi^B (машинная точность на B). */
        val F2span = ModelProblem(
            name = "F2span",
            kernel = KernelF({ t, s -> Math.exp(t - 2.0 * s) }, { t, s -> Math.exp(t - 2.0 * s) }),
            exact = { t -> t * t }, exactDeriv = { t -> 2.0 * t },
            secondKind = true,
        )

        /** F2-B: K=1/(1+t+s), u*=1/(t+1) (проверка порядка на полиномиальном базисе). */
        val F2 = ModelProblem(
            name = "F2",
            kernel = KernelF({ t, s -> 1.0 / (1.0 + t + s) }, { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) }),
            exact = { t -> 1.0 / (t + 1.0) }, exactDeriv = { t -> -1.0 / ((t + 1.0) * (t + 1.0)) },
            secondKind = true,
        )

        /** F2-H: K=e^{-(t-s)^2}, u*=e^t (согласован с phi^H). */
        val F2exp = ModelProblem(
            name = "F2exp",
            kernel = KernelF({ t, s -> Math.exp(-(t - s) * (t - s)) },
                { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) }),
            exact = { t -> Math.exp(t) }, exactDeriv = { t -> Math.exp(t) },
            secondKind = true,
        )

        /** F1: K=e^{-(t-s)^2}, u*=e^t, уравнение I рода (Wazwaz). */
        val F1 = ModelProblem(
            name = "F1",
            kernel = KernelF({ t, s -> Math.exp(-(t - s) * (t - s)) },
                { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) }),
            exact = { t -> Math.exp(t) }, exactDeriv = { t -> Math.exp(t) },
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
    val op: FredholmOperator,
    val cL: Double,
    val fEff: (Double) -> Double,
    val fEffDeriv: (Double) -> Double,
) {
    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    // L omega_i в глобальных гауссовых узлах: LomegaNodes[i][k] = c_L (\mathcal K omega_i)(gNode[k]).
    private val ng = op.gNode.size
    private val LomegaNodes: Array<DoubleArray> = Array(dim) { ki ->
        val i = ki - 2
        DoubleArray(ng) { k -> cL * op.apply(op.gNode[k]) { s -> basis.omega(i, s) } }
    }
    // omega_i в глобальных узлах (для второго применения L).
    private val omegaNodes: Array<DoubleArray> = Array(dim) { ki ->
        val i = ki - 2
        DoubleArray(ng) { k -> basis.omega(i, op.gNode[k]) }
    }

    /** chi_j(g) по значениям g и g' (обёртка). */
    private fun chiOf(g: (Double) -> Double, gD: (Double) -> Double): DoubleArray =
        DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD) }

    /** Матрица M_{j,i} = chi_j(L omega_i). Для xi учитывается производная (L omega_i)'. */
    fun matrixM(): Array<DoubleArray> {
        // Столбцы M независимы по i; cols[i] = столбец i.
        val cols = ParallelAssembly.assembleRows(dim, dim) { i ->
            val ln = LomegaNodes[i]
            val on = omegaNodes[i]
            val Lom = { t: Double -> cL * op.applyNodes(t, on) }
            val LomD = { t: Double -> cL * op.applyDerivNodes(t, on) }
            require(ln.size == ng)
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(Lom, LomD) }
        }
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) for (j in 0 until dim) m[j][i] = cols[i][j]
        return m
    }

    /** Матрица M2_{j,i} = chi_j(L(L omega_i)) (двойное применение L). */
    fun matrixM2(): Array<DoubleArray> {
        val cols = ParallelAssembly.assembleRows(dim, dim) { i ->
            val ln = LomegaNodes[i] // (L omega_i) в узлах
            val LLom = { t: Double -> cL * op.applyNodes(t, ln) }
            val LLomD = { t: Double -> cL * op.applyDerivNodes(t, ln) }
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(LLom, LLomD) }
        }
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) for (j in 0 until dim) m[j][i] = cols[i][j]
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

    /** Слоан: ~u_h(t) = f(t) + (L u_h)(t). */
    fun sloan(): SolutionFunc {
        val c = solveBaseCoeffs()
        val uNodes = DoubleArray(ng) { basis.evalSpline(c, op.gNode[it]) }
        return SolutionFunc { t -> fEff(t) + cL * op.applyNodes(t, uNodes) }
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
        val yNodes = DoubleArray(ng) { basis.evalSpline(c, op.gNode[it]) }
        val wFun = { t: Double -> fEff(t) + cL * op.applyNodes(t, yNodes) }
        val wDFun = { t: Double -> fEffDeriv(t) + cL * op.applyDerivNodes(t, yNodes) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun)
        return SolutionFunc { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) }
    }

    /**
     * Кулкарни для mu, lambda: итерация u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)). Работа в узлах квадратуры.
     * [численное наблюдение]: разрешимость/сходимость не гарантированы (нет P^2=P).
     */
    private fun kulkarniQuasi(): SolutionFunc {
        // Итерант хранится как НЕПРЕРЫВНАЯ функция u^{(m)}(t): реконструкция того же
        // порядка, что и базис (через basis.evalSpline и точную квадратуру op.applyNodes),
        // без понижающей кусочно-линейной интерполяции узловых значений.
        var uFun: (Double) -> Double = { t -> fEff(t) }
        var uNodes = DoubleArray(ng) { fEff(op.gNode[it]) }
        for (iter in 0 until 200) {
            val curFun = uFun
            val curNodes = uNodes
            // P_chi u: коэффициенты chi_j(u) по непрерывной u^{(m)} (сохраняет порядок).
            val pc = funcs.projectorCoeffs(curFun)
            val pcNodes = DoubleArray(ng) { basis.evalSpline(pc, op.gNode[it]) }
            // L u в узлах (точная квадратура по узловым значениям текущего итеранта).
            val luFun = { t: Double -> cL * op.applyNodes(t, curNodes) }
            val pLu = funcs.projectorCoeffs(luFun) // P_chi(L u) коэффициенты
            val lpu = { t: Double -> cL * op.applyNodes(t, pcNodes) } // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(lpu) // P_chi(L(P_chi u))
            // Непрерывная реконструкция следующего итеранта u^{(m+1)}(t).
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(pLu, t) + lpu(t) - basis.evalSpline(pLPu, t)
            }
            val nextNodes = DoubleArray(ng) { nextFun(op.gNode[it]) }
            var diff = 0.0
            for (k in 0 until ng) diff = maxOf(diff, abs(nextNodes[k] - curNodes[k]))
            uFun = nextFun
            uNodes = nextNodes
            if (diff < 1e-13) break
        }
        val finalFun = uFun
        return SolutionFunc { t -> finalFun(t) }
    }

    /** Итерированный Кулкарни: ^u_h^K = f + L u_h^K. */
    fun iteratedKulkarni(): SolutionFunc {
        val uK = kulkarni()
        val uNodes = DoubleArray(ng) { uK.eval(op.gNode[it]) }
        return SolutionFunc { t -> fEff(t) + cL * op.applyNodes(t, uNodes) }
    }
}
// ============================================================================
// 10. РЕШАТЕЛЬ УРАВНЕНИЯ I РОДА — МЕТОД WAZWAZ (НЕ ТИХОНОВ)
// ============================================================================

/**
 * F1 по Wazwaz (r2.tex): (alpha I + \mathcal K) u_alpha = f  <=>
 * u_alpha - \mathcal K_eff u_alpha = f/alpha, \mathcal K_eff = -(1/alpha)\mathcal K.
 * Далее — та же схема II рода (база/Слоан/Кулкарни) с c_L = -1/alpha и правой
 * частью f/alpha. alpha = 10^{-10} — ВЫБОР (как в r2). РИСК: M ∝ alpha^{-1},
 * M2 ∝ alpha^{-2} (R2 numerical-scheme.md) — обусловленность растёт при alpha->0.
 */
class FirstKindSolver(
    val problem: ModelProblem,
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    val op: FredholmOperator,
    val alpha: Double = 1e-10,
) {
    private val cL = -1.0 / alpha
    private val fEff = { t: Double -> problem.rhsExact(t, op) / alpha }
    private val fEffDeriv = { t: Double -> problem.rhsExactDeriv(t, op) / alpha }
    private val inner = SecondKindSolver(basis, funcs, op, cL, fEff, fEffDeriv)

    fun base(): SolutionFunc = inner.base()
    fun sloan(): SolutionFunc = inner.sloan()
    fun kulkarni(): SolutionFunc = inner.kulkarni()
    fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}

/** Фабрика решателя II рода для модельной задачи (c_L = 1, f = u* - \mathcal K u*). */
fun secondKindSolver(problem: ModelProblem, basis: MinimalSplineBasis, funcs: FunctionalFamily,
                     op: FredholmOperator): SecondKindSolver =
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
        val op = FredholmOperator(ModelProblem.F2span.kernel, grid, quad)
        val solver = secondKindSolver(ModelProblem.F2span, basis, funcs, op)
        val eh = errorEh({ t -> ModelProblem.F2span.exact(t) }, solver.base().eval, grid)
        return CheckResult("10. Точность на span-задаче (база B)", eh, 1e-8, true)
    }

    /** 11. Sanity: базовая схема II рода сходится (E_8 > E_16 для F2/H). */
    private fun checkBaseSanity(): CheckResult {
        fun ehAt(nn: Int): Double {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val op = FredholmOperator(ModelProblem.F2.kernel, grid, quad)
            val solver = secondKindSolver(ModelProblem.F2, basis, funcs, op)
            return errorEh({ t -> ModelProblem.F2.exact(t) }, solver.base().eval, grid)
        }
        val e8 = ehAt(8); val e16 = ehAt(16)
        val ratio = if (e16 > 0) e8 / e16 else Double.POSITIVE_INFINITY
        val ratioMin = 4.0   // фактор >= 2^2 => наблюдаемый порядок >= 2
        val sane = e8 < 1e-1 && e16 <= e8 && ratio >= ratioMin
        val measured = if (sane) ratioMin / ratio else 1e9
        return CheckResult("11. Sanity базовой схемы (порядок>=2)", measured, 1.0, true)
    }
}
// ============================================================================
// 13. ФОРМАТИРОВАНИЕ + ТАБЛИЦЫ + MAIN
// ============================================================================

object Tables {
    // n=128 не требуется (решение пользователя); сетки ограничены n<=64.
    private val NS = listOf(8, 16, 32, 64)
    private val quad = GaussLegendre(8)

    private fun makeSolver(p: ModelProblem, sys: GeneratingSystem, fam: String, nn: Int): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(nn)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = family(fam, basis)
        val op = FredholmOperator(p.kernel, grid, quad)
        return SecondKindSolver(basis, funcs, op, 1.0,
            { t -> p.rhsExact(t, op) }, { t -> p.rhsExactDeriv(t, op) }) to grid
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi" -> DeBoorFixFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** T1[p]: базисы B/H/T, базовая схема theta: E_h, p_h, C_h. */
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

    /** F1 (Wazwaz, alpha=1e-10): база/Слоан, базис H. */
    fun tableF1() {
        println("\n--- F1 Wazwaz (alpha=1e-10), базис H, theta: база/Слоан ---")
        val p = ModelProblem.F1
        for (nn in listOf(8, 16, 32)) {
            val grid = Grid.uniform(nn)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val op = FredholmOperator(p.kernel, grid, quad)
            val solver = FirstKindSolver(p, basis, ProjFunctionals(basis), op, alpha = 1e-10)
            val ehB = errorEh({ t -> p.exact(t) }, solver.base().eval, grid)
            val ehS = errorEh({ t -> p.exact(t) }, solver.sloan().eval, grid)
            println("   n=%4d E_h(база)=%s E_h(Слоан)=%s".format(nn, Fmt.e(ehB), Fmt.e(ehS)))
        }
        println("   [риск R2: обусловленность M propto alpha^{-1}; выбор alpha — эвристика r2]")
    }
}

fun main() {
    println("=".repeat(72))
    println("FredholmSolver — линейные уравнения Фредгольма (new-01)")
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
    // Два численных примера II рода (каждый — полный набор сравнений):
    //   F2  : K=1/(1+t+s), u*=1/(t+1) (рациональный); F2exp: K=e^{-(t-s)^2}, u*=e^t.
    val secondKindExamples = listOf(
        ModelProblem.F2 to GeneratingSystem.B,
        ModelProblem.F2exp to GeneratingSystem.B,
    )
    for ((p, sys) in secondKindExamples) {
        Tables.tablePhi(p)
        Tables.tableMethods(p, sys)
        Tables.tableFamilies(p, sys)
    }
    // Один пример I рода (Wazwaz).
    Tables.tableF1()
    println("\nРасчёт завершён.")
}

