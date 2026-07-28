package solvers.volterra

import kotlin.math.abs
import numerics.*
import numerics.functionals.*

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

    companion object {
        /** Абсолютный допуск: узел сетки считается строго внутри (a, t), если x < t - EPS. */
        const val BREAKPOINT_INCLUSION_EPS = 1e-15
    }

    /** Составное разбиение [a, t]: внутренние узлы сетки < t, затем сам t. */
    private fun subBreakpoints(t: Double): DoubleArray {
        val bp = grid.breakpoints
        val list = ArrayList<Double>()
        for (x in bp) { if (x < t - BREAKPOINT_INCLUSION_EPS) list.add(x) else break }
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
     * Интеграл int_lo^hi g(s) ds по составному разбиению [lo,hi], делённому узлами сетки.
     * Используется для весов Nyström с ограничением на компактный носитель omega_j
     * ([x_j,x_{j+3}]) -> не более трёх подынтервалов, что снимает O(n)-стоимость на точку.
     */
    fun integrateRange(lo: Double, hi: Double, g: (Double) -> Double): Double {
        if (hi <= lo) return 0.0
        val list = ArrayList<Double>()
        list.add(lo)
        for (x in grid.breakpoints) if (x > lo + BREAKPOINT_INCLUSION_EPS && x < hi - BREAKPOINT_INCLUSION_EPS) list.add(x)
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

/** Результат решения: вычислитель приближённого решения `u_h(t)` в произвольной точке. */
class SolutionFunc(val eval: (Double) -> Double)

/**
 * Линейный решатель уравнения Вольтерры II рода u - L u = f, L = c_L * \mathcal V,
 * где (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds (ПЕРЕМЕННЫЙ верхний предел).
 *
 * c_L = 1 для уравнения II рода; при редукции I->II рода (см. [FirstKindSolver])
 * тоже c_L = 1, но с другим (редуцированным) ядром и правой частью.
 * Правая часть f и её производные задаются явно (fEff/fEffDeriv/fEffDeriv2),
 * чтобы решатель переиспользовался и для задач I рода.
 *
 * Матрицы дискретной задачи:
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
    val fEffDeriv2: (Double) -> Double = { 0.0 },
) {
    private companion object {
        /**
         * Предел числа итераций в схеме Кулкарни для квазиинтерполянтов (mu, lambda).
         * Выбор реализации: теоретической гарантии сходимости нет (нет свойства P^2=P),
         * поэтому нужен жёсткий предел; на модельных задачах сходимость наступает за единицы итераций.
         */
        const val KULKARNI_QUASI_MAX_ITERATIONS = 200

        /**
         * Критерий останова итерации Кулкарни для квазиинтерполянтов: равномерная норма
         * разности соседних итерантов на контрольных точках. Значение вблизи машинной
         * точности: дальнейшее уточнение не имеет смысла из-за шума квадратуры.
         */
        const val KULKARNI_QUASI_TOLERANCE = 1e-13

        /** Предел числа итераций для комбинированного оператора Nyström. */
        const val COMBINED_NYSTROM_MAX_ITERATIONS = 200

        /** Критерий останова итерации комбинированного Nyström. */
        const val COMBINED_NYSTROM_TOLERANCE = 1e-13
    }

    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    // ВНИМАНИЕ (отличие от Фредгольма): у оператора Вольтерра область интегрирования
    // [a,t] зависит от t, поэтому предвычисление на фиксированных узлах невозможно.
    // Все применения L = c_L \mathcal V выражаются через замыкания op.apply / op.applyDeriv.

    /** L g(t) = c_L (\mathcal V g)(t) и её производная (Лейбниц). */
    private fun applyL(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.apply(t, g) }
    private fun applyLDeriv(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.applyDeriv(t, g) }

    /**
     * (L g)''(t) = c_L (\mathcal V g)''(t) по (V2''): требует g И g' (член K(t,t) g'(t)).
     * gD — первая производная самого операнда g.
     */
    private fun applyLDeriv2(g: (Double) -> Double, gD: (Double) -> Double): (Double) -> Double =
        { t -> cL * op.applyDeriv2(t, g, gD) }

    /** chi_j(g) по значениям g, g' и g'' (обёртка). */
    private fun chiOf(
        g: (Double) -> Double,
        gD: (Double) -> Double,
        gDD: (Double) -> Double = { 0.0 },
    ): DoubleArray = DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD, gDD) }

    /** Матрица M_{j,i} = chi_j(L omega_i). Для xi — (L omega_i)', для xi^<0> — и (L omega_i)''. */
    fun matrixM(): Array<DoubleArray> {
        // Столбцы M независимы по i; cols[i] = столбец i.
        val cols = ParallelAssembly.assembleRows(dim, dim) { i ->
            val idx = i - 2
            val omega = { s: Double -> basis.omega(idx, s) }
            val omegaD = { s: Double -> basis.omegaDeriv(idx, s) }
            val image = applyL(omega)
            val imageDeriv = applyLDeriv(omega)
            // Вторая производная образа требует и omega_i, и omega_i' (член K(t,t) omega_i').
            val imageDeriv2 = applyLDeriv2(omega, omegaD)
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(image, imageDeriv, imageDeriv2) }
        }
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) for (j in 0 until dim) m[j][i] = cols[i][j]
        return m
    }

    /** Матрица M2_{j,i} = chi_j(L(L omega_i)) (двойное применение L). */
    fun matrixM2(): Array<DoubleArray> {
        val cols = ParallelAssembly.assembleRows(dim, dim) { i ->
            val idx = i - 2
            val omega = { s: Double -> basis.omega(idx, s) }
            val image = applyL(omega)
            val imageDeriv = applyLDeriv(omega)
            val doubleImage = applyL(image)
            val doubleImageDeriv = applyLDeriv(image)
            // Вторая производная требует сам образ L omega_i и его производную.
            val doubleImageDeriv2 = applyLDeriv2(image, imageDeriv)
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(doubleImage, doubleImageDeriv, doubleImageDeriv2) }
        }
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) for (j in 0 until dim) m[j][i] = cols[i][j]
        return m
    }

    /** g_j = chi_j(f). */
    fun vectorG(): DoubleArray = chiOf(fEff, fEffDeriv, fEffDeriv2)

    /** d_j = chi_j(L f). */
    fun vectorD(): DoubleArray {
        val rhsImage = { t: Double -> cL * op.apply(t) { s -> fEff(s) } }
        val rhsImageDeriv = { t: Double -> cL * op.applyDeriv(t) { s -> fEff(s) } }
        // Вторая производная образа правой части требует и f, и f'.
        val rhsImageDeriv2 = applyLDeriv2(fEff, fEffDeriv)
        return chiOf(rhsImage, rhsImageDeriv, rhsImageDeriv2)
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
        val splineImage = applyL { s -> basis.evalSpline(c, s) }
        return SolutionFunc { t -> fEff(t) + splineImage(t) }
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
        val yhD = { s: Double -> basis.evalSplineDeriv(c, s) }
        val yhImage = applyL(yh)
        val yhImageDeriv = applyLDeriv(yh)
        val yhImageDeriv2 = applyLDeriv2(yh, yhD)
        val wFun = { t: Double -> fEff(t) + yhImage(t) }
        val wDFun = { t: Double -> fEffDeriv(t) + yhImageDeriv(t) }
        val wDDFun = { t: Double -> fEffDeriv2(t) + yhImageDeriv2(t) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun, wDDFun)
        return SolutionFunc { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) }
    }

    /**
     * Кулкарни для mu, lambda: итерация u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)).
     * [численное наблюдение]: разрешимость/сходимость не гарантированы (нет P^2=P).
     *
     * Итерант хранится как НЕПРЕРЫВНАЯ функция: реконструкция того же порядка,
     * что и базис (через basis.evalSpline и точную квадратуру op.apply), без понижающей
     * кусочно-линейной интерполяции узловых значений.
     *
     * Ранее здесь использовалось хранение итеранта в виде значений на равномерной
     * выборке с кусочно-линейным восстановлением, что ограничивало точность величиной
     * O(h_sample^2) НЕЗАВИСИМО от порядка базиса. В решателе Фредгольма это было
     * исправлено ранее; здесь применён тот же подход для согласованности двух решателей.
     */
    private fun kulkarniQuasi(): SolutionFunc {
        var uFun: (Double) -> Double = { t -> fEff(t) }
        // Контрольные точки только для критерия останова (в самой итерации не участвуют).
        val checkPoints = DoubleArray(4 * n + 1) { grid.a + (grid.b - grid.a) * it / (4 * n) }
        var uAtCheck = DoubleArray(checkPoints.size) { uFun(checkPoints[it]) }
        for (iter in 0 until KULKARNI_QUASI_MAX_ITERATIONS) {
            val curFun = uFun
            val pc = funcs.projectorCoeffs(curFun)
            val pcFun = { s: Double -> basis.evalSpline(pc, s) }
            val luFun = applyL(curFun)                    // L u
            val pLu = funcs.projectorCoeffs(luFun)        // P_chi(L u)
            val lpu = applyL(pcFun)                       // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(lpu)         // P_chi(L(P_chi u))
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(pLu, t) + lpu(t) - basis.evalSpline(pLPu, t)
            }
            val nextAtCheck = DoubleArray(checkPoints.size) { nextFun(checkPoints[it]) }
            var diff = 0.0
            for (k in nextAtCheck.indices) diff = maxOf(diff, abs(nextAtCheck[k] - uAtCheck[k]))
            uFun = nextFun
            uAtCheck = nextAtCheck
            if (diff < KULKARNI_QUASI_TOLERANCE) break
        }
        val finalFun = uFun
        return SolutionFunc { t -> finalFun(t) }
    }

    /** Итерированный Кулкарни: ^u_h^K = f + L u_h^K. */
    fun iteratedKulkarni(): SolutionFunc {
        val kulkarniSolution = kulkarni()
        val kulkarniImage = applyL { s -> kulkarniSolution.eval(s) }
        return SolutionFunc { t -> fEff(t) + kulkarniImage(t) }
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
        return SolutionFunc { t -> nystromEval(sup, uHat, t) }
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
        return SolutionFunc { t -> fEff(t) + uN(t) }
    }

    /**
     * КОМБИНИРОВАННЫЙ оператор Nyström для уравнения Вольтерры:
     * u^N_h = f + L_n u^N_h, где L_n = P_chi L + (I - P_chi) L^N_h.
     *
     * На образе проектора действует ТОЧНЫЙ оператор, на дополнении — квадратура с
     * t-зависимыми весами. Отличие от [nystrom]: там решается u = f + L^N_h u.
     *
     * СТАТУС ИСТОЧНИКА: конструкция L_n взята из теории для уравнения Фредгольма
     * (см. [solvers.fredholm.SecondKindSolver.combinedNystrom]); для уравнения Вольтерры это
     * АДАПТАЦИЯ: доказательства суперсходимости в известной литературе НЕТ. Поведение
     * следует трактовать как численное наблюдение.
     *
     * @throws IllegalStateException если простая итерация не сошлась.
     */
    fun combinedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        val checkPoints = DoubleArray(4 * n + 1) { grid.a + (grid.b - grid.a) * it / (4 * n) }
        var uAtCheck = DoubleArray(checkPoints.size) { uFun(checkPoints[it]) }
        var converged = false
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
            if (diff < COMBINED_NYSTROM_TOLERANCE) { converged = true; break }
        }
        check(converged) {
            "Комбинированный Nyström (Вольтерра) не сошёлся за " +
                "$COMBINED_NYSTROM_MAX_ITERATIONS итераций"
        }
        val resultFun = uFun
        return SolutionFunc { t -> resultFun(t) }
    }

    /** Итерированный комбинированный Nyström: \hat u^N_h = f + L u^N_h с точным L. */
    fun iteratedCombinedNystrom(): SolutionFunc {
        val combined = combinedNystrom()
        val image = applyL { s -> combined.eval(s) }
        return SolutionFunc { t -> fEff(t) + image(t) }
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
 * Случай `m = 1` (однократное дифференцирование) применим только при `K(t,t) != 0`;
 * это проверяется при создании решателя.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство аппроксимационных функционалов.
 * @param kernel ядро ИСХОДНОГО уравнения первого рода.
 * @param rhsDeriv производная правой части `f'(t)` исходного уравнения.
 * @param smoothPart гладкая часть решения, известная аналитически; из-под конечной
 *        разности она выносится, чтобы не усиливать шум (см. пояснение к [gEffDeriv]).
 * @param smoothPartDeriv производная гладкой части.
 * @throws IllegalArgumentException если `K(t,t)` близко к нулю в контрольных точках
 *         либо выбрано семейство функционалов, требующее второй производной.
 */
class FirstKindSolver(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    kernel: KernelV,
    rhsDeriv: (Double) -> Double,
    smoothPart: (Double) -> Double,
    smoothPartDeriv: (Double) -> Double,
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
         * ОГРАНИЧЕНИЕ: шаг абсолютный, поэтому на очень коротких отрезках
         * (`b - a` порядка `1e-3` и меньше) шаблон `t ± 4h` выйдет за пределы области.
         */
        const val FINITE_DIFFERENCE_STEP = 1e-3

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

    /** Диагональ ядра `K(t,t)` — знаменатель редукции первого рода ко второму. */
    private val kernelDiagonal = { t: Double -> kernel.k(t, t) }

    init {
        // Редукция делит на K(t,t), поэтому обращение диагонали в ноль недопустимо.
        // Проверяем узлы сетки и середины интервалов — точки, где деление реально
        // выполняется. Без этой проверки ядро со слабой особенностью (например K = t - s,
        // где K(t,t) = 0) дало бы NaN/Inf без какой-либо диагностики.
        val breakpoints = grid.breakpoints
        for (i in breakpoints.indices) {
            val samples = if (i < breakpoints.size - 1) {
                doubleArrayOf(breakpoints[i], 0.5 * (breakpoints[i] + breakpoints[i + 1]))
            } else {
                doubleArrayOf(breakpoints[i])
            }
            for (t in samples) {
                val diagonal = kernelDiagonal(t)
                require(abs(diagonal) >= KERNEL_DIAGONAL_TOLERANCE) {
                    "Решатель уравнения Вольтерры I рода требует K(t,t) != 0 (случай m=1); " +
                        "|K(t,t)|=${abs(diagonal)} при t=$t слишком мало"
                }
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

    private val inner = SecondKindSolver(
        basis, funcs, reducedOperator, cL = 1.0, fEff = gEff, fEffDeriv = gEffDeriv,
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
