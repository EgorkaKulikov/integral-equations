package problems.uryson

import numerics.GaussLegendre
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
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

/** Создаёт решатель уравнения второго рода для модельной задачи. */
fun secondKindSolver(
    problem: UrysonProblem,
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
): UrysonSecondKindSolver = UrysonSecondKindSolver(
    basis = basis,
    funcs = funcs,
    space = space,
    op = op,
    lambda = problem.lambda,
    rhs = { t -> problem.rhsExact(t, op) },
)

/** Создаёт регуляризованный решатель уравнения первого рода. */
fun firstKindSolver(
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
): UrysonFirstKindSolver = UrysonFirstKindSolver(basis, funcs, space, op)

/**
 * Число контрольных узлов профиля шума на один интервал сетки.
 *
 * Профиль должен быть заметно мельче сетки, иначе шум окажется «видимым» для базиса
 * и будет частично воспроизведён вместо того, чтобы играть роль возмущения данных.
 */
private const val NOISE_NODES_PER_INTERVAL = 4

/**
 * Строит зашумлённую правую часть `f^delta = f + xi` с заданной нормой шума
 * `||xi||_{L^2} = delta`.
 *
 * Шум моделируется кусочно-линейным профилем со случайными значениями в контрольных
 * узлах, отмасштабированным точно под требуемый уровень `delta`.
 *
 * ВОСПРОИЗВОДИМОСТЬ: генератор инициализируется ЯВНО передаваемым [seed], поэтому
 * результат полностью детерминирован. Скрытого источника случайности здесь нет.
 *
 * @param exactRhs точная правая часть `f`.
 * @param grid сетка, задающая отрезок.
 * @param quad квадратура для вычисления нормы шума.
 * @param delta требуемая норма возмущения в `L^2`; при нуле возвращается исходная функция.
 * @param seed зерно генератора псевдослучайных чисел.
 */
fun noisyRightHandSide(
    exactRhs: (Double) -> Double,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
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
    val l2Norm = Math.sqrt(quad.integrate(noiseNodes) { t -> noiseProfile(t) * noiseProfile(t) })
    val scale = if (l2Norm > 0) delta / l2Norm else 0.0
    return { t -> exactRhs(t) + scale * noiseProfile(t) }
}

/**
 * Возвращает вектор `theta_j(f^delta)` зашумлённых данных для задачи первого рода —
 * входные данные метода [UrysonFirstKindSolver.solveMorozov].
 *
 * @param delta уровень шума в норме `L^2`.
 * @param seed зерно генератора; фиксируется явно ради воспроизводимости.
 */
fun noisyThetaCoefficients(
    problem: UrysonProblem,
    solver: UrysonFirstKindSolver,
    op: UrysohnOperator,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
): DoubleArray {
    val noisy = noisyRightHandSide({ t -> problem.rhsExact(t, op) }, grid, quad, delta, seed)
    return solver.thetaOf(noisy)
}
