package solvers.uryson

import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext
import splines.functionals.ProjFunctionals
import solvers.core.reportConvergence
import solvers.core.FirstKindSolution

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
public class UrysonFirstKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val space: SplineSpace,
    public val op: UrysohnOperator,
    public val tau: Double = DEFAULT_TAU,
    public val gnTol: Double = DEFAULT_GN_TOLERANCE,
    public val gnMaxIter: Int = DEFAULT_GN_MAX_ITERATIONS,
    public val throwOnDivergence: Boolean = false,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // КРИТИЧНО именно здесь: [solveMorozov] считает стабилизатор `Omega` через
        // `space.omegaReg` (то есть через `space.ctx.backend`), а систему Гаусса–Ньютона —
        // через собственный `ctx.backend`. При расхождении две части ОДНОГО критерия
        // Морозова считались бы разными реализациями LU — молча.
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "space", space.ctx)
    }

    public companion object {
        /**
         * Коэффициент запаса в принципе невязки Морозова.
         *
         * Теория требует только `tau > 1`; конкретное значение — выбор реализации:
         * чем оно ближе к единице, тем меньше сглаживание, но тем выше чувствительность
         * к неточности оценки уровня шума.
         */
        public const val DEFAULT_TAU: Double = 1.1

        /** Критерий останова Гаусса–Ньютона по равномерной норме шага. */
        public const val DEFAULT_GN_TOLERANCE: Double = 1e-10

        /**
         * Предел итераций Гаусса–Ньютона при фиксированном `alpha`. Метод применяется
         * внутри гомотопии по параметру регуляризации, где каждый следующий запуск
         * стартует с предыдущего решения, поэтому большого числа итераций не требуется.
         */
        public const val DEFAULT_GN_MAX_ITERATIONS: Int = 50

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

    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    private val core = CollocationCore(basis, funcs, op, ctx)
    private val weights = space.weights
    private val gramR = space.gramR

    /** Вектор значений функционалов `theta_j(f)` для произвольной правой части. */
    public fun thetaOf(f: (Double) -> Double): DoubleArray =
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
    public fun solveFixedAlpha(thetaFDelta: DoubleArray, alpha: Double, c0: DoubleArray): DoubleArray {
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
    public fun residual(c: DoubleArray, thetaFDelta: DoubleArray): Double {
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
     * НЕЯВНАЯ ЗАВИСИМОСТЬ ОЦЕНКИ ШУМА ОТ ТИПА [funcs]. Множитель
     * `funcs.cChi()` в `barDelta` — коэффициент усиления возмущения входных
     * данных функционалами `theta_j`. Он корректен ИМЕННО потому, что тип
     * параметра [funcs] ограничен [ProjFunctionals] — семейством функционалов-
     * ЗНАЧЕНИЙ (`usesDerivative` там есть константа `false`). Для них
     * `cChi()` действительно равен норме (квази)проектора на возмущениях значений.
     *
     * Расширение типа [funcs] до общего `FunctionalFamily` ПОТРЕБУЕТ пересмотра
     * этой оценки: для семейств с производными (xi) `cChi()` оценкой усиления
     * шума не является и ведёт себя по `h` качественно противоположно (см. KDoc
     * `splines.functionals.FunctionalFamily.cChi` и `numerics.functionals.DerivFunctional`).
     *
     * @param thetaFDelta вектор `theta_j(f^delta)` зашумлённых данных.
     * @param delta уровень шума в норме `L^2`; при `delta = 0` путь проходится целиком.
     */
    public fun solveMorozov(thetaFDelta: DoubleArray, delta: Double): FirstKindSolution {
        // Корректность cChi() как коэффициента усиления шума держится на том, что
        // funcs — ProjFunctionals (функционалы-значения). См. KDoc метода.
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
