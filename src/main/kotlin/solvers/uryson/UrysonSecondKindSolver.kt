package solvers.uryson

import kotlin.math.abs
import numerics.DenseMatrix
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext
import numerics.ParallelAssembly
import solvers.core.SolutionFunc
import splines.functionals.ProjFunctionals
import solvers.core.SupportPoints
import solvers.core.reportConvergence
import solvers.core.NewtonResult

/**
 * Решатели нелинейного уравнения Урысона ВТОРОГО рода
 * `x(t) - cL \int_a^b K(t,s,x(s)) ds = f(t)`.
 *
 * Реализованы схемы: базовая коллокация, итерация Слоана, модификация Кулкарни и её
 * итерированный вариант, простой сплайн-метод Nyström, а также комбинированный метод
 * Nyström `P_theta L + (I - P_theta) L^N_h` и его итерированный вариант.
 * Источники перечислены в `docs/REFERENCES.md`.
 *
 * Решатель не знает о модельных задачах: правая часть передаётся функцией [rhs],
 * а множитель — параметром [cL]. Готовые фабрики для модельных задач находятся
 * в пакете `problems.uryson`.
 *
 * ПОЧЕМУ ЗДЕСЬ НЕТ ОБЩЕГО ЯДРА `solvers.core.SecondKindSolverCore`. Ядро строит
 * матрицу `M_{j,i} = chi_j(L omega_i)`, которая существует лишь для ЛИНЕЙНОГО `L`:
 * образ базисной функции не зависит от решения. У Урысона оператор нелинеен, и
 * аналогом `M` служит якобиан [CollocationCore.bMatrix], ЗАВИСЯЩИЙ от текущего
 * приближения `c`; базовая схема — цикл Ньютона, а не одно `LinearAlgebra.solve`.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство проекционных функционалов `theta_j`. Тип сужен до
 *        [ProjFunctionals] НЕ исторически: схемы обходят `th.nodes`/`th.coeffs`
 *        напрямую через `valueFunctional`, который на любом другом семействе
 *        падает, а в [UrysonFirstKindSolver.solveMorozov] от `usesDerivative ==
 *        false` зависит корректность `cChi()` как оценки усиления шума.
 * @param space сплайн-пространство (нужны веса Nyström `W_j`).
 * @param op оператор Урысона.
 * @param cL множитель перед интегральным оператором.
 * @param rhs правая часть `f(t)` — сырая функция, а НЕ `RhsWithDerivatives`.
 *        Производные `f` читают только `chi_j` семейств `xi`, а здесь [funcs] —
 *        всегда [ProjFunctionals] с `usesDerivative == false`: все обращения к
 *        правой части берут лишь значение. Тройка добавила бы два поля, которые
 *        никогда не читаются, но выглядели бы влияющими на результат.
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
public class UrysonSecondKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val space: SplineSpace,
    public val op: UrysohnOperator,
    public val cL: Double,
    public val rhs: (Double) -> Double,
    public val tol: Double = DEFAULT_TOLERANCE,
    public val maxIter: Int = DEFAULT_MAX_ITERATIONS,
    public val kulkarniMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    public val nystromMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    public val throwOnDivergence: Boolean = true,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // Семейство и сплайн-пространство считают ЧАСТИ ТОЙ ЖЕ задачи — тем же бэкендом.
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "space", space.ctx)
    }

    public companion object {
        /** Критерий останова по умолчанию: близко к машинной точности. */
        public const val DEFAULT_TOLERANCE: Double = 1e-12

        /**
         * Предел итераций Ньютона в базовой схеме. Взят с большим запасом: метод
         * квадратично сходится за единицы итераций, а предел защищает лишь от
         * зацикливания на вырожденных данных.
         */
        public const val DEFAULT_MAX_ITERATIONS: Int = 10_000

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
        public const val DEFAULT_FIXED_POINT_MAX_ITERATIONS: Int = 60

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

    public val grid: Grid = basis.grid
    public val n: Int = grid.n

    /** Предвычисленные значения `theta_j(f)` для точной правой части. */
    private val thetaF: DoubleArray = DoubleArray(n + 2) { k -> funcs.valueFunctional(k - 2).applyTo(rhs) }

    private val collocation = CollocationCore(basis, funcs, op, ctx)

    /**
     * Базовая схема: `c = theta(f) + cL Xi(c)`.
     *
     * Решается методом Ньютона для `F(c) = c - theta(f) - cL Xi(c) = 0` с
     * аналитическим якобианом `J = I - cL B(c)`. Ньютон выбран вместо простой
     * итерации, поскольку сходится и при отсутствии сжатия (например, при `cL = 1`
     * и кубическом ядре).
     *
     * @return коэффициенты сплайна вместе со сведениями о сходимости Ньютона.
     * @throws IllegalStateException при недостижении сходимости, если [throwOnDivergence].
     */
    public fun solveBase(): NewtonResult {
        val c = thetaF.copyOf()
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        val run = runNewtonIterations(
            x = c,
            maxSteps = maxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val xi = collocation.xiVector(current)
                DoubleArray(n + 2) { current[it] - thetaF[it] - cL * xi[it] }
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
     * Шаг Ньютона с АНАЛИТИЧЕСКИМ якобианом `I - cL B(c)`, ОБЩИЙ для базовой
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

    /** Шаг Ньютона с якобианом `J = I - cL B(c)` (строки собираются независимо). */
    private fun newtonStep(c: DoubleArray, negativeResidual: DoubleArray): DoubleArray {
        val b = collocation.bMatrix(c)
        val jacobian = ParallelAssembly.assembleDense(n + 2, n + 2, ctx.parallel) { r, col ->
            val value = -cL * b[r, col]
            if (r == col) value + 1.0 else value
        }
        return LinearAlgebra.solve(jacobian, negativeResidual, ctx.backend)
    }

    /**
     * Реконструкция правой части схемы Кулкарни по коэффициентам `c` сплайна `y_h`.
     *
     * Возвращает тройку:
     *  - `yhNodes` — значения `y_h` в узлах квадратуры [UrysohnOperator.gNode];
     *  - `gAtSupport` — функция `g(t) = f(t) + cL (U y_h)(t)` (вычисляется в любой
     *    точке, в частности в опорных точках функционалов);
     *  - `gCoeffs` — коэффициенты проекции `P_theta g`.
     *
     * Блок нужен дважды: на каждой итерации квази-Ньютона (внутри `G_K`) и после
     * выхода из цикла — при восстановлении `x_h^K = y_h + (I - P_theta) g`.
     */
    private fun projectedRhs(c: DoubleArray): Triple<DoubleArray, (Double) -> Double, DoubleArray> {
        val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
        val gAtSupport = { t: Double -> rhs(t) + cL * op.applyNodes(t, yhNodes) }
        val gCoeffs = funcs.projectorCoeffs(gAtSupport)
        return Triple(yhNodes, gAtSupport, gCoeffs)
    }

    /** Базовое приближение `x_h` как сплайн. */
    public fun base(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        return SolutionFunc(
            eval = { t -> basis.evalSpline(c, t) },
            converged = newton.converged,
            iterations = newton.iterations,
            residual = newton.residual,
        )
    }

    /** Итерация Слоана: `\tilde x_h(t) = f(t) + cL (U x_h)(t)`. */
    public fun sloan(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        val splineSolution = { t: Double -> basis.evalSpline(c, t) }
        val eval = { t: Double -> rhs(t) + cL * op.apply(t) { s -> splineSolution(s) } }
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
     * Система для `y_h = P_theta x_h^K`: `c = theta(f) + cL Theta(U(arg(c)))`, где
     * `arg = y_h + (I - P_theta)[f + cL U(y_h)]`. Решается квази-Ньютоном, у которого
     * в роли предобуславливателя выступает якобиан БАЗОВОЙ схемы `I - cL B(c)`.
     * Такой выбор предписан источником и обеспечивает сходимость при отсутствии сжатия,
     * где простая итерация расходится.
     *
     * Итоговое приближение восстанавливается как `x_h^K = y_h + (I - P_theta)[f + cL U(y_h)]`.
     */
    public fun kulkarni(): SolutionFunc {
        val fNodes = DoubleArray(op.gNode.size) { rhs(op.gNode[it]) }

        /** Правая часть системы Кулкарни `G_K(c)`. */
        fun gK(c: DoubleArray): DoubleArray {
            val (yhNodes, _, gCoeffs) = projectedRhs(c)
            val uyhNodes = DoubleArray(op.gNode.size) { op.applyNodes(op.gNode[it], yhNodes) }
            val gNodes = DoubleArray(op.gNode.size) { fNodes[it] + cL * uyhNodes[it] }
            // Остаток проектора в узлах квадратуры: arg = y_h + (I - P_theta) g.
            val argNodes = DoubleArray(op.gNode.size) {
                yhNodes[it] + gNodes[it] - basis.evalSpline(gCoeffs, op.gNode[it])
            }
            return DoubleArray(n + 2) { k ->
                val th = funcs.valueFunctional(k - 2)
                var acc = 0.0
                for (q in th.nodes.indices) acc += th.coeffs[q] * op.applyNodes(th.nodes[q], argNodes)
                thetaF[k] + cL * acc
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
     * Сплайн-метод Nyström: `x_h^N(t) = f(t) + cL sum_j theta_j(g_t) W_j`,
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
    public fun nystrom(): SolutionFunc {
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
            return rhs(t) + cL * acc
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
                val jacobian = DenseMatrix.zeros(p, p)
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
                        jacobian[row, col] = (identity * step - (perturbed[row] - gx[row])) / step
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

    /**
     * КОМБИНИРОВАННЫЙ метод Nyström: `u = f + cL L_n u`,
     * `L_n = P_theta L + (I - P_theta) L^N_h` — точный оператор на образе проектора,
     * квадратура на его дополнении. Именно к этому оператору (а не к простому [nystrom])
     * относятся оценки суперсходимости для полиномиальных квазиинтерполянтов
     * (Remogna–Sbibih–Tahrichi, Mathematics 11 (2023), Art. 3236; см. `docs/REFERENCES.md`).
     *
     * Система решается методом Ньютона с аналитическим якобианом по значениям решения
     * в опорных точках функционалов и в узлах квадратуры; подробности — [CombinedNystromSolver].
     */
    public fun combinedNystrom(): SolutionFunc = CombinedNystromSolver(this).combined()

    /**
     * Итерированный комбинированный Nyström: `\hat u^N_h = f + cL L u^N_h`, где `u^N_h` —
     * решение [combinedNystrom]; однократное применение точного оператора без новой системы.
     */
    public fun iteratedCombinedNystrom(): SolutionFunc = CombinedNystromSolver(this).iterated()

    /**
     * Итерированный метод Кулкарни: `\hat u^K_h = f + cL L u^K_h`, где `u^K_h` — решение
     * [kulkarni]; аналог итерации Слоана, применённой к приближению Кулкарни.
     */
    public fun iteratedKulkarni(): SolutionFunc {
        val k = kulkarni()
        val uNodes = DoubleArray(op.gNode.size) { k.eval(op.gNode[it]) }
        return SolutionFunc(
            eval = { t -> rhs(t) + cL * op.applyNodes(t, uNodes) },
            converged = k.converged,
            iterations = k.iterations,
            residual = k.residual,
        )
    }
}
