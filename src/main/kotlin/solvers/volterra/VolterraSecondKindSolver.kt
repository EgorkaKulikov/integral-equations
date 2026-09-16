package solvers.volterra

import kotlin.math.abs
import numerics.*
import splines.*
import solvers.core.SolutionFunc
import solvers.core.SupportPoints
import solvers.core.reportConvergence
import splines.functionals.*
import splines.metrics.*
import solvers.core.ImageTriple
import solvers.core.IterationStopCriterion
import solvers.core.RhsWithDerivatives
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_TOLERANCE
import solvers.core.SecondKindSolverCore

/**
 * Линейный решатель уравнения Вольтерры II рода u - L u = f, L = c_L * \mathcal V,
 * где (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds (ПЕРЕМЕННЫЙ верхний предел).
 *
 * c_L = 1 для уравнения II рода; при редукции I->II рода (см. [VolterraFirstKindSolver])
 * тоже c_L = 1, но с другим (редуцированным) ядром и правой частью.
 * Правая часть f и её производные задаются явно ([RhsWithDerivatives]),
 * чтобы решатель переиспользовался и для задач I рода.
 *
 * Матрицы дискретной задачи:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * @param throwOnDivergence поведение ИТЕРАЦИОННЫХ схем ([kulkarni] для
 *        квазиинтерполянтов, [combinedNystrom]) при недостижении сходимости:
 *        `true` (по умолчанию) — исключение, `false` — результат с
 *        `converged = false` и достигнутой невязкой в [solvers.core.SolutionFunc.residual].
 *        На прямые схемы не влияет.
 */
public class VolterraSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    public val op: VolterraOperator,
    cL: Double,
    rhs: RhsWithDerivatives,
    throwOnDivergence: Boolean = true,
    ctx: NumericsContext = NumericsContext.default(),
) : SecondKindSolverCore<(Double) -> Double>(
    basis, funcs, cL, rhs, throwOnDivergence, ctx,
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
     * ValueFunctional-ы семейства и индексация точек. В отличие от Фредгольма
     * веса W_j(t)=int_a^t omega_j зависят от t, поэтому агрегированные веса b_r(t)
     * вычисляются на лету (nystromB). Семейство xi (де Бура--Фикса) НЕ поддерживается:
     * его функционалы используют производную и не сводятся к линейной комбинации значений.
     *
     * ИНДЕКСАЦИЯ ТОЧЕК — по паре (номер функционала, номер узла), см. [SupportPoints].
     * Раньше здесь жила `HashMap<Double, Int>` с поиском по ЗНАЧЕНИЮ точки, то есть
     * требовавшая побитового совпадения Double и работавшая лишь потому, что и карта,
     * и запрос читали один и тот же массив `vf.nodes`.
     */
    private class NystromSupport(
        val pts: DoubleArray,
        val vfs: Array<ValueFunctional>,
        val support: SupportPoints,
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
        val support = SupportPoints.byAscendingValue(vfs, grid.breakpointInclusionEps)
        return NystromSupport(support.points, vfs, support)
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
            for (q in vf.nodes.indices) b[sup.support.indexOf(k, q)] += vf.coeffs[q] * w
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
        val a = DenseMatrix.zeros(p, p)
        for (rho in 0 until p) {
            val b = nystromB(sup, sup.pts[rho]) // t-зависимые веса при t=eta_rho
            for (r in 0 until p) a[rho, r] = -cL * b[r] * op.kernel.k(sup.pts[rho], sup.pts[r])
            a[rho, rho] += 1.0
        }
        return LinearAlgebra.solve(a, DoubleArray(p) { fEff(sup.pts[it]) }, ctx.backend)
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
    public fun nystrom(): SolutionFunc {
        val sup = nystromSupport()
        val uHat = nystromSolve(sup)
        return SolutionFunc(eval = { t -> nystromEval(sup, uHat, t) })
    }

    /**
     * Итерированный Nyström: u_hat^N_h(t)=f(t)+(L u^N_h)(t) с ТОЧНЫМ оператором
     * Вольтерра L (замыкание applyL, как в sloan()). Одно интегрирование найденного
     * u^N_h, новой системы не требуется (аналог итерации Слоана).
     */
    public fun iteratedNystrom(): SolutionFunc {
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
    public fun combinedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        // Критерий останова мерится на ТОМ ЖЕ множестве [checkPoints], что и в `kulkarniQuasi`.
        // Локальный пересчёт той же формулой был бы вторым критерием в одном классе:
        // при правке одного из них две схемы молча разъехались бы.
        val checkPoints = this.checkPoints
        var uAtCheck = DoubleArray(checkPoints.size) { uFun(checkPoints[it]) }
        val stop = IterationStopCriterion(COMBINED_NYSTROM_TOLERANCE)
        while (stop.performedIterations < COMBINED_NYSTROM_MAX_ITERATIONS) {
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
            if (stop.accept(diff)) break
        }
        reportConvergence(
            converged = stop.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Комбинированный Nyström (Вольтерра)",
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
     * Итерированный комбинированный Nyström: \hat u^N_h = f + L u^N_h с точным L.
     * Признак сходимости наследуется от [combinedNystrom].
     */
    public fun iteratedCombinedNystrom(): SolutionFunc {
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
