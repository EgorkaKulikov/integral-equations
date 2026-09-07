package problems.uryson

import numerics.GaussLegendre
import splines.Grid
import splines.MinimalSplineBasis
import numerics.NumericsContext
import splines.functionals.ProjFunctionals
import solvers.uryson.Kernel
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import solvers.uryson.UrysonFirstKindSolver
import solvers.uryson.UrysonSecondKindSolver

/**
 * Модельная задача для нелинейного уравнения Урысона: ядро, множитель и точное решение.
 *
 * Правая часть НЕ задаётся явно, а строится из точного решения численно (квадратурой),
 * поэтому тестовые данные всегда согласованы с оператором и квадратурной формулой.
 *
 * Соотношения между точным решением `x*` и правой частью `f`:
 *  - уравнение второго рода: `f(t) = x*(t) - lambda \int K(t,s,x*(s)) ds`;
 *  - уравнение первого рода: `f(t) = \int K(t,s,x*(s)) ds`.
 *
 * Отрезок задаётся не здесь, а сеткой [Grid], передаваемой в оператор.
 *
 * @param name краткое имя задачи для таблиц и сообщений тестов.
 * @param kernel ядро `K(t,s,u)` вместе с производной по `u`.
 * @param lambda множитель перед интегральным оператором (для задач первого рода не используется).
 * @param exact точное решение — эталон для вычисления погрешности.
 * @param secondKind `true` — уравнение второго рода, `false` — первого.
 */
class UrysonProblem(
    val name: String,
    val kernel: Kernel,
    val lambda: Double,
    val exact: (Double) -> Double,
    val secondKind: Boolean,
) {
    /** Точная правая часть `f(t)`, вычисленная через оператор [op]. */
    fun rhsExact(t: Double, op: UrysohnOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - lambda * integral else integral
    }

    companion object {
        /**
         * Задача A (второго рода): `K = 1/(t+s+u)`, `lambda = -1`, `x* = 1/(t+1)`.
         *
         * Ядро гладкое и убывающее, оператор сжимающий — базовый сценарий сходимости.
         */
        val A = UrysonProblem(
            name = "A",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = -1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = true,
        )

        /**
         * Задача B (второго рода): `K = e^{t-2s} u^3`, `lambda = 1`, `x* = e^t`.
         *
         * Кубическая нелинейность при `lambda = 1` делает оператор НЕсжимающим:
         * простая итерация расходится, поэтому задача проверяет именно ньютоновский путь.
         * Решение `e^t` принадлежит гиперболической порождающей системе `phi^H`.
         */
        val B = UrysonProblem(
            name = "B",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = Math.exp(t - 2.0 * s) * u * u * u
                override fun dkdu(t: Double, s: Double, u: Double) = 3.0 * Math.exp(t - 2.0 * s) * u * u
            },
            lambda = 1.0,
            exact = { t -> Math.exp(t) },
            secondKind = true,
        )

        /** Задача C (первого рода, некорректная): `K = 1/(t+s+u)`, `x* = 1/(t+1)`. */
        val C = UrysonProblem(
            name = "C",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = 1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = false,
        )

        /**
         * Задача D (первого рода, некорректная): `K = e^{-(t-s)^2} u^3`, `x* = e^t`.
         *
         * Особенность: `dK/du(t,s,0) = 0`, поэтому из НУЛЕВОГО начального приближения
         * якобиан вырождается и метод Гаусса–Ньютона не сдвигается с места. Именно
         * поэтому решатели стартуют с проекции постоянной функции.
         */
        val D = UrysonProblem(
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

/**
 * Создаёт решатель уравнения второго рода для модельной задачи.
 *
 * @param ctx контекст вычислений; обязан совпадать с контекстами [funcs] и [space]
 *        (проверяется конструктором решателя).
 */
fun secondKindSolver(
    problem: UrysonProblem,
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
    ctx: NumericsContext = NumericsContext.default(),
): UrysonSecondKindSolver = UrysonSecondKindSolver(
    basis = basis,
    funcs = funcs,
    space = space,
    op = op,
    lambda = problem.lambda,
    rhs = { t -> problem.rhsExact(t, op) },
    ctx = ctx,
)

/**
 * Создаёт регуляризованный решатель уравнения первого рода.
 *
 * @param ctx контекст вычислений; обязан совпадать с контекстами [funcs] и [space].
 */
fun firstKindSolver(
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
    ctx: NumericsContext = NumericsContext.default(),
): UrysonFirstKindSolver = UrysonFirstKindSolver(basis, funcs, space, op, ctx = ctx)

/**
 * Число контрольных узлов профиля шума на один интервал сетки.
 *
 * Профиль должен быть заметно мельче сетки, иначе шум окажется «видимым» для базиса
 * и будет частично воспроизведён вместо того, чтобы играть роль возмущения данных.
 */
private const val NOISE_NODES_PER_INTERVAL = 4

/**
 * Норма, в которой задаётся уровень шума `delta` для [noisyRightHandSide].
 *
 *  * [L2] — `||xi||_{L^2(a,b)} = delta` (прежнее поведение, используется golden-тестами);
 *  * [SUP] — `||xi||_infty = max_t |xi(t)| = delta` — согласована с условием (VII)
 *    статьи (шум задан в `C[a,b]`). Для кусочно-линейного профиля максимум модуля
 *    достигается в контрольном узле, поэтому вычисляется точно, без квадратуры.
 */
enum class NoiseNorm { L2, SUP }

/**
 * Строит зашумлённую правую часть `f^delta = f + xi` с заданной нормой шума
 * `||xi|| = delta` в норме [norm] (по умолчанию `L^2`).
 *
 * Шум моделируется кусочно-линейным профилем со случайными значениями в контрольных
 * узлах, отмасштабированным точно под требуемый уровень `delta`.
 *
 * ВОСПРОИЗВОДИМОСТЬ: генератор инициализируется ЯВНО передаваемым [seed], поэтому
 * результат полностью детерминирован. Скрытого источника случайности здесь нет.
 * Профиль (узлы и случайные значения) при данном `seed` ОДИНАКОВ для обеих норм —
 * различается лишь масштабный множитель.
 *
 * @param exactRhs точная правая часть `f`.
 * @param grid сетка, задающая отрезок.
 * @param quad квадратура для вычисления `L^2`-нормы шума (при [NoiseNorm.SUP] не используется).
 * @param delta требуемая норма возмущения; при нуле возвращается исходная функция.
 * @param seed зерно генератора псевдослучайных чисел.
 * @param norm норма, в которой задан `delta`.
 */
fun noisyRightHandSide(
    exactRhs: (Double) -> Double,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
    norm: NoiseNorm = NoiseNorm.L2,
): (Double) -> Double {
    if (delta == 0.0) return exactRhs
    val random = kotlin.random.Random(seed)
    val nodeCount = NOISE_NODES_PER_INTERVAL * grid.n
    val noiseNodes = DoubleArray(nodeCount + 1) { grid.a + (grid.b - grid.a) * it / nodeCount }
    val noiseValues = DoubleArray(nodeCount + 1) { random.nextDouble(-1.0, 1.0) }
    val noiseProfile = { t: Double ->
        var k = 0
        while (k < nodeCount - 1 && t >= noiseNodes[k + 1]) k++
        val left = noiseNodes[k]
        val right = noiseNodes[k + 1]
        val w = ((t - left) / (right - left)).coerceIn(0.0, 1.0)
        noiseValues[k] * (1 - w) + noiseValues[k + 1] * w
    }
    val profileNorm = when (norm) {
        NoiseNorm.L2 -> Math.sqrt(quad.integrate(noiseNodes) { t -> noiseProfile(t) * noiseProfile(t) })
        // Кусочно-линейная функция достигает максимума модуля в узле.
        NoiseNorm.SUP -> noiseValues.maxOf { Math.abs(it) }
    }
    val scale = if (profileNorm > 0) delta / profileNorm else 0.0
    return { t -> exactRhs(t) + scale * noiseProfile(t) }
}

/**
 * Возвращает вектор `theta_j(f^delta)` зашумлённых данных для задачи первого рода —
 * входные данные метода [UrysonFirstKindSolver.solveMorozov].
 *
 * @param delta уровень шума в норме [norm] (по умолчанию `L^2`).
 * @param seed зерно генератора; фиксируется явно ради воспроизводимости.
 * @param norm норма, в которой задан `delta` (см. [NoiseNorm]).
 */
fun noisyThetaCoefficients(
    problem: UrysonProblem,
    solver: UrysonFirstKindSolver,
    op: UrysohnOperator,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
    norm: NoiseNorm = NoiseNorm.L2,
): DoubleArray {
    val noisy = noisyRightHandSide({ t -> problem.rhsExact(t, op) }, grid, quad, delta, seed, norm)
    return solver.thetaOf(noisy)
}
