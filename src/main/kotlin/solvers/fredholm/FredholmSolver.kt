package solvers.fredholm

import kotlin.math.abs
import numerics.*
import numerics.functionals.*
import solvers.core.ImageTriple
import solvers.core.IterationStopCriterion
import solvers.core.RhsWithDerivatives
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_TOLERANCE
import solvers.core.SecondKindSolverCore

/**
 * Ядро K(t,s) линейного уравнения Фредгольма вместе с аналитическими частными
 * производными.
 *
 * @param k само ядро `K(t,s)`.
 * @param kT производная `K_t(t,s)`; требуется семействам функционалов
 *        де Бура–Фикса `xi^<1>`, `xi^<2>`.
 * @param kTT вторая производная `K_tt(t,s)`; требуется семейству `xi^<0>`.
 *
 * Значения по умолчанию равны нулю и допустимы ТОЛЬКО тогда, когда соответствующая
 * производная действительно тождественно нулевая, либо когда выбранное семейство
 * функционалов её не использует: иначе система будет построена неверно без какой-либо
 * диагностики.
 */
class KernelF(
    val k: (Double, Double) -> Double,
    val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
)

/**
 * Оператор Фредгольма `(K u)(t) = \int_a^b K(t,s) u(s) ds` с постоянными пределами
 * интегрирования.
 *
 * При создании предвычисляются глобальные гауссовы узлы [gNode] и веса [gW]
 * составной квадратуры, так что `\int h = sum_k gW[k] * h(gNode[k])`. Это позволяет
 * многократно применять оператор к уже вычисленным в этих узлах значениям функции
 * (см. [applyNodes]), не пересчитывая её заново.
 *
 * @param kernel ядро уравнения.
 * @param grid сетка, задающая отрезок `[a,b]` и точки разбиения для составной квадратуры.
 * @param quad квадратурная формула Гаусса–Лежандра на ячейке.
 */
class FredholmOperator(val kernel: KernelF, val grid: Grid, val quad: GaussLegendre) {
    /**
     * Глобальные узлы составной квадратуры.
     *
     * READ-ONLY ПО СОГЛАШЕНИЮ: содержимое НЕЛЬЗЯ изменять. ГОРЯЧЕЕ поле — копия не
     * возвращается сознательно: массив читается во внутренних циклах [applyNodes],
     * [applyDerivNodes], [applyDeriv2Nodes] и при сборке матриц — копирование на каждом
     * обращении дало бы квадратичный рост аллокаций. Записей в проекте нет.
     */
    val gNode: DoubleArray

    /** Веса квадратуры при узлах [gNode]. READ-ONLY по соглашению (горячее, см. [gNode]). */
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

    /** d^2/dt^2 (\mathcal K u)(t) = \int_a^b d^2K/dt^2(t,s) u(s) ds (для xi^<0>). */
    fun applyDeriv2(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kTT(t, s) * u(s) }

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

    /** d^2/dt^2 (\mathcal K u)(tau) по предвычисленным uNodes (для xi^<0>). */
    fun applyDeriv2Nodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kTT(tau, gNode[k]) * uNodes[k]
        return s
    }
}


/**
 * Линейный решатель уравнения II рода u - L u = f, L = c_L * \mathcal K
 * (c_L = 1 для F2; c_L = -1/alpha для F1 Wazwaz, \mathcal K_eff = -(1/alpha)\mathcal K).
 * Правая часть f и её производные задаются явно ([RhsWithDerivatives]) для переиспользования в F1.
 *
 * Матрицы дискретной задачи:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * @param throwOnDivergence поведение ИТЕРАЦИОННЫХ схем ([kulkarni] для
 *        квазиинтерполянтов, [combinedNystrom]) при недостижении сходимости:
 *        `true` (по умолчанию) — исключение, `false` — результат с
 *        `converged = false` и достигнутой невязкой в [SolutionFunc.residual].
 *        На прямые схемы не влияет. Параметр задан на уровне решателя,
 *        а не каждого метода: это политика обработки ошибок, а не свойство
 *        отдельной схемы.
 */
class FredholmSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    val op: FredholmOperator,
    cL: Double,
    rhs: RhsWithDerivatives,
    throwOnDivergence: Boolean = true,
    ctx: NumericsContext = NumericsContext.default(),
) : SecondKindSolverCore<DoubleArray>(
    basis, funcs, cL, rhs, throwOnDivergence, ctx,
) {
    // Числовые параметры итерационных схем (COMBINED_NYSTROM_*, KULKARNI_QUASI_*)
    // живут в [solvers.core.SecondKindDefaults]: у Фредгольма и Вольтерры они
    // совпадают, и расхождение при подборе нового значения было бы молчаливым.
    //
    // Поля `grid`, `n`, `dim` объявлены в [SecondKindSolverCore] и инициализируются РАНЬШЕ
    // полей этого класса — именно тот порядок, который нужен [omegaNodes] (он
    // эагерный и читает `dim`). Обратный порядок дал бы `dim = 2` МОЛЧА.

    private val ng = op.gNode.size

    /**
     * `L omega_i` в глобальных гауссовых узлах:
     * `LomegaNodes[i][k] = c_L (\mathcal K omega_{i-2})(gNode[k])`. ЛЕНИВОЕ поле.
     *
     * Почему ленивое: единственный потребитель — [matrixM2] (второе применение L),
     * а внутри класса `M2` нужна только схеме Кулкарни для ПРОЕКТОРОВ
     * ([kulkarni] → `kulkarniProjector`, где строится `I - M - M2 + M^2`); [matrixM2]
     * публична и вызывается также из тестов. Остальные схемы — [base], [sloan],
     * `kulkarniQuasi` для квазиинтерполянтов, всё семейство Nyström ([nystrom],
     * [combinedNystrom]) — к `M2` не обращаются вовсе.
     *
     * Стоимость сборки велика: `dim * ng` интегралов (по одному на каждую пару
     * `i, k`), и каждый из них — квадратура по `ng` узлам, то есть O(dim * ng^2)
     * обращений к ядру. При эагерном поле эту цену платили ВСЕ схемы, в том числе
     * те, которые её не используют.
     *
     * Режим ленивости — по умолчанию (`SYNCHRONIZED`), и это обязательно: поле
     * читается из потоков параллельной сборки ([ParallelAssembly] в [matrixM2]),
     * так что первое обращение возможно из рабочего потока, и `NONE` был бы гонкой.
     * Узким местом синхронизация не становится: чтение идёт один раз на столбец
     * (`dim` раз всего), а не во внутреннем цикле по узлам.
     */
    private val LomegaNodes: Array<DoubleArray> by lazy {
        Array(dim) { ki ->
            val i = ki - 2
            DoubleArray(ng) { k -> cL * op.apply(op.gNode[k]) { s -> basis.omega(i, s) } }
        }
    }

    /**
     * `omega_i` в глобальных узлах: `omegaNodes[i][k] = omega_{i-2}(gNode[k])` — аргумент
     * для `applyNodes` при сборке [matrixM].
     *
     * Поле оставлено ЭАГЕРНЫМ НЕ потому, что нужно всем схемам (его читает только
     * [matrixM], а семейство Nyström и `kulkarniQuasi` его не требуют), а потому, что
     * оно дешёвое: `dim * ng` вычислений сплайна без обращений к ядру, то есть в `ng`
     * раз дешевле [LomegaNodes]. Ленивость здесь дала бы только накладные расходы.
     */
    private val omegaNodes: Array<DoubleArray> = Array(dim) { ki ->
        val i = ki - 2
        DoubleArray(ng) { k -> basis.omega(i, op.gNode[k]) }
    }

    // --- Реализация точек расширения [SecondKindSolverCore] --------------------
    //
    // Подготовленный операнд у Фредгольма — массив значений функции в глобальных
    // гауссовых узлах: пределы интегрирования постоянны, поэтому одни и те же узлы
    // обслуживают любое `t`, и повторное применение L сводится к свёртке с ядром.

    override val equationName: String get() = "Фредгольм"

    override val kulkarniQuasiHint: String
        get() = "Для квазиинтерполянтов (mu, lambda) нет свойства P^2 = P, поэтому " +
            "редукция Кулкарни неприменима и используется простая итерация, требующая " +
            "сжатия (спектральный радиус оператора меньше единицы)"

    /**
     * Критерий останова мерится в гауссовых узлах оператора — тех же, по которым
     * идёт сама итерация. Синтетическая выборка, как у Вольтерры, здесь не нужна
     * и дала бы другое число итераций.
     */
    override val checkPoints: DoubleArray get() = op.gNode

    override fun prepare(u: (Double) -> Double): DoubleArray =
        DoubleArray(ng) { u(op.gNode[it]) }

    override fun image(o: DoubleArray): (Double) -> Double = { t -> cL * op.applyNodes(t, o) }

    override fun imageDeriv(o: DoubleArray): (Double) -> Double = { t -> cL * op.applyDerivNodes(t, o) }

    /** Пределы интегрирования постоянны, члена Лейбница нет, поэтому `uD` не читается. */
    @Suppress("UNUSED_PARAMETER")
    override fun imageDeriv2(o: DoubleArray, uD: (Double) -> Double): (Double) -> Double =
        { t -> cL * op.applyDeriv2Nodes(t, o) }

    /**
     * Контрольные точки СОВПАДАЮТ с узлами подготовки операнда, поэтому значения
     * уже вычислены — второй проход по `u` был бы чистым удвоением работы
     * на каждой итерации (`u` там — результат применения оператора, а не таблица).
     */
    override fun checkValues(u: (Double) -> Double, o: DoubleArray): DoubleArray = o

    override fun applyOperator(t: Double, u: (Double) -> Double): Double = op.apply(t, u)

    override fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double = op.applyDeriv(t, u)

    /** У Фредгольма вторая производная образа не содержит `u'`: пределы постоянны. */
    @Suppress("UNUSED_PARAMETER")
    override fun applyOperatorDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double =
        op.applyDeriv2(t, u)

    override fun omegaImages(i: Int): ImageTriple {
        val on = omegaNodes[i]
        return ImageTriple(
            { t -> cL * op.applyNodes(t, on) },
            { t -> cL * op.applyDerivNodes(t, on) },
            { t -> cL * op.applyDeriv2Nodes(t, on) },
        )
    }

    override fun doubleOmegaImages(i: Int): ImageTriple {
        val ln = LomegaNodes[i] // (L omega_i) в узлах
        return ImageTriple(
            { t -> cL * op.applyNodes(t, ln) },
            { t -> cL * op.applyDerivNodes(t, ln) },
            { t -> cL * op.applyDeriv2Nodes(t, ln) },
        )
    }

    // --- Nyström (сплайн-квадратура; см. docs/REFERENCES.md, раздел 3) -------

    /**
     * Опорные точки {eta_r} и агрегированные веса b_r базового Nyström:
     * b_r = sum_j sum_{q: s_{j,q}=eta_r} c_{j,q} W_j, W_j = int_a^b omega_j.
     * Точки упорядочены по возрастанию (для единообразия с Вольтерра; для F2 порядок
     * несуществен). Семейство xi (де Бура--Фикса) НЕ поддерживается: его функционалы
     * используют производную и не сводятся к линейной комбинации значений
     * (известное ограничение метода; обходится семейством xitilde).
     *
     * ИНДЕКСАЦИЯ ТОЧЕК — по паре (номер функционала, номер узла), см. [SupportPoints].
     * Раньше индекс искался по ЗНАЧЕНИЮ точки в `HashMap<Double, Int>`, то есть держался
     * на побитовом совпадении Double; работало это лишь потому, что и карта, и запрос
     * читали один и тот же массив `vf.nodes`. Слияние совпадающих точек разных
     * функционалов теперь идёт по ЯВНОМУ допуску `grid.breakpointInclusionEps`.
     */
    private fun nystromSupport(): Pair<DoubleArray, DoubleArray> {
        require(!funcs.usesDerivative) {
            "Nyström для семейства '${funcs.name}' не реализован: функционалы " +
                "де Бура--Фикса (xi) используют производную и не сводятся к значениям."
        }
        val vfs = Array(dim) { k ->
            funcs.chi(k - 2) as? ValueFunctional
                ?: error("Nyström: функционал '${funcs.name}' (j=${k - 2}) не является ValueFunctional.")
        }
        // W_j = int_a^b omega_j (высокоточная составная квадратура по узлам сетки).
        val wJ = DoubleArray(dim) { k -> op.quad.integrate(grid.breakpoints) { s -> basis.omega(k - 2, s) } }
        val support = SupportPoints.byAscendingValue(vfs, grid.breakpointInclusionEps)
        val pts = support.points
        val bAgg = DoubleArray(support.size)
        for (k in 0 until dim) {
            val vf = vfs[k]; val w = wJ[k]
            for (q in vf.nodes.indices) bAgg[support.indexOf(k, q)] += vf.coeffs[q] * w
        }
        return pts to bAgg
    }

    /** u^N_h(t) = f(t) + cL sum_r b_r K(t, eta_r) u_hat_r — восстановление решения. */
    private fun nystromEval(t: Double, pts: DoubleArray, bAgg: DoubleArray, uHat: DoubleArray): Double =
        fEff(t) + nystromQuadrature(t, pts, bAgg, uHat)

    /**
     * Квадратурный оператор (L^N_h u)(t) = cL sum_r b_r K(t, eta_r) u(eta_r).
     *
     * Зависит от u только через её значения в опорных точках [uAtPoints] — именно
     * это свойство делает оператор конечноранговым.
     */
    private fun nystromQuadrature(
        t: Double,
        pts: DoubleArray,
        bAgg: DoubleArray,
        uAtPoints: DoubleArray,
    ): Double {
        var acc = 0.0
        for (r in pts.indices) acc += bAgg[r] * op.kernel.k(t, pts[r]) * uAtPoints[r]
        return cL * acc
    }

    /** Матрица (I - A^N): A^N_{rho,r} = cL b_r K(eta_rho, eta_r). */
    private fun nystromMatrix(pts: DoubleArray, bAgg: DoubleArray): Array<DoubleArray> {
        val p = pts.size
        val a = LinearAlgebra.zeros(p, p)
        for (rho in 0 until p) {
            for (r in 0 until p) a[rho][r] = -cL * bAgg[r] * op.kernel.k(pts[rho], pts[r])
            a[rho][rho] += 1.0
        }
        return a
    }

    /**
     * КЛАССИЧЕСКИЙ сплайн-Nyström: подынтегральная функция g_t(s)=K(t,s)u(s)
     * заменяется своей сплайн-(квази)проекцией, интеграл — квадратурой
     * sum_j chi_j(g_t) W_j. Решается u = f + L^N_h u, что даёт линейную систему
     * (I - A^N) u_hat = f_hat по ЗНАЧЕНИЯМ решения в опорных точках {eta_r}.
     * Приближение u^N_h лежит ВНЕ сплайнового пространства. Не поддерживает семейство xi.
     *
     * ВАЖНО о порядке сходимости: это «голая» квадратура, которая сама по себе
     * НЕ повышает порядок. Опубликованные оценки суперсходимости O(h^7)/O(h^8)
     * относятся НЕ к ней, а к комбинированному оператору — см. [combinedNystrom].
     */
    fun nystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        val uHat = LinearAlgebra.solve(nystromMatrix(pts, bAgg), DoubleArray(pts.size) { fEff(pts[it]) }, ctx.backend)
        return SolutionFunc(eval = { t -> nystromEval(t, pts, bAgg, uHat) })
    }

    /**
     * Итерированный Nyström: u_hat^N_h(t)=f(t)+(L u^N_h)(t) с
     * ТОЧНЫМ оператором L (высокоточная квадратура op.applyNodes, как в sloan()). Одно
     * интегрирование найденного u^N_h, новой системы не требуется (аналог итерации Слоана).
     */
    fun iteratedNystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        val uHat = LinearAlgebra.solve(nystromMatrix(pts, bAgg), DoubleArray(pts.size) { fEff(pts[it]) }, ctx.backend)
        val uNodes = DoubleArray(ng) { nystromEval(op.gNode[it], pts, bAgg, uHat) }
        return SolutionFunc(eval = { t -> fEff(t) + cL * op.applyNodes(t, uNodes) })
    }

    /**
     * КОМБИНИРОВАННЫЙ оператор Nyström: u^N_h = f + L_n u^N_h, где
     *
     *     L_n = P_chi L + (I - P_chi) L^N_h,
     *
     * то есть на образе проектора действует ТОЧНЫЙ оператор, а на его дополнении —
     * квадратура. Отличие от [nystrom]: там решается u = f + L^N_h u («голая»
     * квадратура, классический Nyström).
     *
     * Зачем это нужно: в разности L - L_n = (I - P_chi)(L - L^C_h) остаток проектора
     * (I - P_chi) входит ДВАЖДЫ — явным множителем и внутри остатка квадратуры, —
     * что и даёт суперсходимость. Именно к этому оператору, а не к голой квадратуре,
     * относятся опубликованные оценки порядка O(h^7) и O(h^8) (см. список источников
     * в docs/REFERENCES.md: Allouch, Remogna, Sbibih, Tahrichi, AMC 404 (2021), Art. 126227;
     * Remogna, Sbibih, Tahrichi, Mathematics 11 (2023), Art. 3236).
     *
     * Способ решения: простая итерация u^{(m+1)} = f + L_n u^{(m)}. Оператор L_n
     * конечного ранга, поэтому задача равносильна конечномерной СЛАУ; итерация выбрана
     * как существенно более простая реализация (прямая сборка требует P×P интегралов
     * вида ∫K(t,s)K(s,eta_r)ds). Сходимость линейна со знаменателем ||L_n|| и требует
     * ||L_n|| < 1; при недостижении сходимости бросается исключение (а не возвращается
     * молча неверный результат).
     *
     * @throws IllegalStateException если итерация не сошлась и [throwOnDivergence] равно `true`.
     */
    fun combinedNystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        var uAtNodes = DoubleArray(ng) { uFun(op.gNode[it]) }
        val stop = IterationStopCriterion(COMBINED_NYSTROM_TOLERANCE)
        while (stop.performedIterations < COMBINED_NYSTROM_MAX_ITERATIONS) {
            val currentFun = uFun
            val currentNodes = uAtNodes
            val currentAtPoints = DoubleArray(pts.size) { currentFun(pts[it]) }
            // Точный оператор L на текущем итеранте и его проекция P_chi(L u).
            val exactImage = { t: Double -> cL * op.applyNodes(t, currentNodes) }
            val projectedExact = funcs.projectorCoeffs(exactImage)
            // Квадратурный оператор L^N_h и его проекция P_chi(L^N_h u).
            val quadratureImage = { t: Double -> nystromQuadrature(t, pts, bAgg, currentAtPoints) }
            val projectedQuadrature = funcs.projectorCoeffs(quadratureImage)
            // u^{(m+1)} = f + P_chi(L u) + L^N_h u - P_chi(L^N_h u).
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(projectedExact, t) +
                    quadratureImage(t) - basis.evalSpline(projectedQuadrature, t)
            }
            val nextNodes = DoubleArray(ng) { nextFun(op.gNode[it]) }
            var diff = 0.0
            for (k in 0 until ng) diff = maxOf(diff, abs(nextNodes[k] - currentNodes[k]))
            uFun = nextFun
            uAtNodes = nextNodes
            if (stop.accept(diff)) break
        }
        reportConvergence(
            converged = stop.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Комбинированный Nyström (Фредгольм)",
            iterations = stop.performedIterations,
            maxIterations = COMBINED_NYSTROM_MAX_ITERATIONS,
            residual = stop.residual,
            tolerance = COMBINED_NYSTROM_TOLERANCE,
            hint = "Для сходимости простой итерации требуется ||L_n|| < 1",
            diverged = stop.diverged,
        )
        val resultFun = uFun
        return SolutionFunc(
            eval = { t -> resultFun(t) },
            converged = stop.converged,
            iterations = stop.performedIterations,
            residual = stop.residual,
        )
    }

    /**
     * Итерированный комбинированный Nyström: \hat u^N_h = f + L u^N_h с ТОЧНЫМ
     * оператором L (аналог итерации Слоана; новой системы не требует).
     */
    fun iteratedCombinedNystrom(): SolutionFunc {
        val combined = combinedNystrom()
        val uNodes = DoubleArray(ng) { combined.eval(op.gNode[it]) }
        // Признак сходимости наследуется от исходного комбинированного оператора.
        return SolutionFunc(
            eval = { t -> fEff(t) + cL * op.applyNodes(t, uNodes) },
            converged = combined.converged,
            iterations = combined.iterations,
            residual = combined.residual,
        )
    }
}

/**
 * Решатель некорректного уравнения Фредгольма ПЕРВОГО рода `K u = f` методом
 * регуляризации.
 *
 * Математическая идея: уравнение заменяется возмущённым `(alpha I + K) u_alpha = f`,
 * которое алгебраически эквивалентно уравнению ВТОРОГО рода
 *
 *     u_alpha - K_eff u_alpha = f / alpha,   K_eff = -(1/alpha) K,
 *
 * после чего применяется обычная схема второго рода с `c_L = -1/alpha`.
 * Источник метода указан в `docs/REFERENCES.md` (раздел «Уравнения первого рода»).
 *
 * ОГРАНИЧЕНИЕ ПО ОБУСЛОВЛЕННОСТИ: элементы матрицы `M` растут как `alpha^{-1}`,
 * а `M2` — как `alpha^{-2}`, поэтому при малых `alpha` задача становится плохо
 * обусловленной. По этой причине публикуются только базовая схема и итерация Слоана:
 * схема Кулкарни, использующая `M2`, для уравнения первого рода признана неприменимой.
 *
 * ГРАНИЦА ПРИМЕНИМОСТИ ПО `alpha` — ИЗМЕРЕНО, А НЕ ПРЕДПОЛОЖЕНО. Формулировка
 * выше качественная; вот числа, установленные независимой численной верификацией
 * (вторым, независимо написанным кодом):
 *
 *  - ПРИ `alpha = 1e-10` ЗНАЧАЩИХ ЦИФР ОСТАЁТСЯ МАЛО. Обусловленность собранной
 *    системы составила `cond_inf ≈ 2.636149e+10`. Восемь ЗАКОННЫХ вариантов
 *    реализации (разные порядки вычисления ядра и правой части, разные решатели
 *    плотной СЛАУ) дали ЧЕТЫРЕ различных значения ОДНОЙ и той же величины:
 *    `3.822e-05 / 4.108e-05 / 4.176e-05 / 4.402e-05`, разброс `15.19 %`.
 *  - ЭТО НЕ БАГ, А ОГРАНИЧЕНИЕ ТОЧНОСТИ. Априорная оценка
 *    `eps · cond / значение ≈ 14.6 %` совпадает с наблюдаемым разбросом, то есть
 *    эффект полностью объясняется обусловленностью, а не ошибкой в коде.
 *  - ГРАНИЦА — ПРИМЕРНО `alpha = 1e-8`. При `alpha = 1e-6` и `alpha = 1e-8` те же
 *    восемь вариантов расхождения НЕ дают: `0.00 %` и `0.01 %` соответственно.
 *    Ниже примерно `1e-8` значащих цифр в результате остаётся мало, и `alpha`
 *    обязан выбираться осознанно.
 *  - НАСКОЛЬКО МАЛО НУЖНО, ЧТОБЫ СДВИНУТЬ ОТВЕТ. Запись точного решения
 *    делением `cos(wt)/(1+t)` против умножения `cos(wt) · p`, где `p = 1/(1+t)`,
 *    различается РОВНО на ОДНО округление — и меняет результат на `7.49 %`.
 *
 * КАК ПРОВЕРИТЬ СЕБЯ (самопроверка, не требующая внешнего эталона). Посчитайте
 * интересующую величину ДВУМЯ алгебраически ЭКВИВАЛЕНТНЫМИ способами записи
 * (например, деление против умножения на обратную величину) и сравните.
 * Расхождение НЕ есть ошибка одного из вариантов — это ОЦЕНКА СНИЗУ для реально
 * доступной точности: цифры, различающиеся между двумя законными записями,
 * не несут информации ни в одной из них. Сверху же точность ограничивает
 * `cond_inf · ε`; измерить оба множителя позволяют [baseCondition] и
 * [numerics.LinearAlgebra.solveDiagnosed]. Подробный разбор — `docs/ACCURACY.md`.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство аппроксимационных функционалов.
 * @param op оператор Фредгольма с исходным ядром.
 * @param rhs правая часть `f(t)` исходного уравнения первого рода.
 * @param rhsDeriv первая производная `f'(t)`.
 * @param rhsDeriv2 вторая производная `f''(t)`; нужна семейству `xi^<0>`.
 * @param alpha параметр регуляризации; должен быть строго положителен.
 * @param throwOnDivergence политика обработки недостижения сходимости итерационными
 *        схемами внутреннего решателя; см. [FredholmSecondKindSolver.throwOnDivergence].
 * @throws IllegalArgumentException если `alpha <= 0`.
 */
class FredholmFirstKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    val op: FredholmOperator,
    rhs: (Double) -> Double,
    rhsDeriv: (Double) -> Double,
    rhsDeriv2: (Double) -> Double = { 0.0 },
    val alpha: Double = DEFAULT_REGULARIZATION,
    val throwOnDivergence: Boolean = true,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    companion object {
        /**
         * Значение параметра регуляризации по умолчанию.
         *
         * Это ЭКСПЕРИМЕНТАЛЬНЫЙ выбор авторов цитируемой работы, а не рекомендация
         * из теории метода регуляризации: оптимальное `alpha` зависит от уровня шума
         * в данных и в общем случае должно подбираться (например, по принципу невязки).
         *
         * ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ ПОПАДАЕТ В ПРОБЛЕМНЫЙ ДИАПАЗОН. Именно при
         * `alpha = 1e-10` измерен разброс `15.19 %` между восемью алгебраически
         * эквивалентными вариантами записи одной величины, тогда как при `1e-6`
         * и `1e-8` расхождения нет (`0.00 %` и `0.01 %`); числа и их происхождение —
         * в KDoc [FredholmFirstKindSolver] и в `docs/ACCURACY.md`. Значение СОХРАНЕНО
         * (его менять нельзя — оно воспроизводит публикацию), но принимать его
         * молча не следует: если задача терпит большее `alpha`, точность выше.
         */
        const val DEFAULT_REGULARIZATION = 1e-10
    }

    init {
        require(alpha > 0.0) {
            "FredholmFirstKindSolver: параметр регуляризации alpha должен быть положительным, получено alpha=$alpha"
        }
    }

    private val cL = -1.0 / alpha
    private val fEff = { t: Double -> rhs(t) / alpha }
    private val fEffDeriv = { t: Double -> rhsDeriv(t) / alpha }
    // Порядок построения лямбд сохранён дословно; объединение в один объект
    // происходит ниже, на границе вызова внутреннего решателя.
    // Вторая производная правой части ОБЯЗАТЕЛЬНО пробрасывается во внутренний
    // решатель: без неё семейство xi^<0>, читающее f'', молча получало ноль вместо
    // истинного значения и строило неверную систему без какой-либо диагностики.
    private val fEffDeriv2 = { t: Double -> rhsDeriv2(t) / alpha }
    private val inner = FredholmSecondKindSolver(
        basis, funcs, op, cL,
        RhsWithDerivatives(fEff, fEffDeriv, fEffDeriv2),
        throwOnDivergence, ctx,
    )

    /** Базовая коллокационная схема для регуляризованного уравнения. */
    fun base(): SolutionFunc = inner.base()

    /**
     * Число обусловленности `cond_inf` СОБРАННОЙ матрицы базовой схемы `I - M`
     * — вместе с признаком достоверности самой оценки.
     *
     * ЗАЧЕМ. Граница применимости по `alpha` (см. KDoc класса) сформулирована через
     * `cond_inf`, но без этого метода пользователь не мог бы увидеть её для СВОЕЙ
     * задачи и СВОЕГО `alpha`: число из KDoc относится к конкретному прогону, а не к
     * методу вообще. Метод добавлен ИМЕННО для протокола воспроизводимости.
     *
     * НЕ ВЫЗЫВАЕТСЯ НИ ОДНОЙ СХЕМОЙ и ничего в них не меняет: цена O(n⁴)
     * (сборка `M` плюс обращение) платится только тем, кто явно спросил.
     *
     * ОЦЕНКА МОЖЕТ ОКАЗАТЬСЯ НЕДОСТОВЕРНОЙ, и это штатный исход, а не ошибка:
     * при малом `alpha` матрица почти вырождена, а оценка через обращение в этом
     * режиме теряет смысл (измерено: при `alpha <= 1e-12` невязка обращения
     * достигала 0.08…760). Именно поэтому возвращается [ConditionEstimate],
     * а не `Double`: печатать его можно только после проверки
     * [ConditionEstimate.isReliable] (либо через [ConditionEstimate.valueOrNull]).
     */
    fun baseCondition(
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = Conditioning.conditionInf(inner.baseMatrix(), tolerance)

    /** Итерация Слоана, применённая к регуляризованному уравнению. */
    fun sloan(): SolutionFunc = inner.sloan()

    /**
     * Схема Кулкарни для регуляризованного уравнения.
     *
     * НЕ РЕКОМЕНДУЕТСЯ к применению: схема использует матрицу `M2`, элементы которой
     * растут как `alpha^{-2}`, что при типичных `alpha ~ 1e-10` делает систему
     * численно неразрешимой. Метод сохранён для полноты API и экспериментов.
     */
    fun kulkarni(): SolutionFunc = inner.kulkarni()

    /** Итерированная схема Кулкарни; те же ограничения, что и у [kulkarni]. */
    fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}
