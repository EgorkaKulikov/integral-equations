package problems.fredholm

import numerics.MinimalSplineBasis
import numerics.functionals.FunctionalFamily
import solvers.fredholm.FirstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.KernelF
import solvers.fredholm.SecondKindSolver

/**
 * Модельная задача для линейного уравнения Фредгольма: ядро, точное решение и род
 * уравнения. Правая часть НЕ задаётся явно, а строится из точного решения численно
 * (квадратурой) — так тестовые данные остаются согласованными с оператором и
 * квадратурной формулой, а не «зашитыми» константами.
 *
 * Соотношения между точным решением `u*` и правой частью `f`:
 *  - уравнение II рода: `f = u* - K u*`;
 *  - уравнение I рода:  `f = K u*`.
 *
 * Область интегрирования задаётся не здесь, а сеткой [numerics.Grid], передаваемой
 * в оператор: задача описывает только ядро и решение.
 *
 * @param name краткое имя задачи, используемое в таблицах и сообщениях тестов.
 * @param kernel ядро `K(t,s)` вместе с аналитическими частными производными.
 * @param exact точное решение `u*(t)` — эталон для вычисления погрешности.
 * @param exactDeriv первая производная точного решения `u*'(t)`; требуется
 *        семействам функционалов де Бура–Фикса `xi^<1>`, `xi^<2>`.
 * @param secondKind `true` — уравнение второго рода, `false` — первого.
 * @param exactDeriv2 вторая производная `u*''(t)`; требуется семейству `xi^<0>`.
 *        Значение по умолчанию (ноль) допустимо ТОЛЬКО когда вторая производная
 *        действительно равна нулю, иначе семейство `xi^<0>` молча получит неверную
 *        правую часть.
 */
class FredholmProblem(
    val name: String,
    val kernel: KernelF,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val exactDeriv2: (Double) -> Double = { 0.0 },
) {
    /**
     * Точная правая часть `f(t)`, вычисленная через оператор [op].
     *
     * @param t точка вычисления.
     * @param op оператор Фредгольма, построенный на том же ядре и нужной сетке.
     */
    fun rhsExact(t: Double, op: FredholmOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /**
     * Первая производная правой части `f'(t)`; нужна семействам функционалов,
     * использующим производную. Для уравнения II рода равна `u*' - d/dt (K u*)`.
     */
    fun rhsExactDeriv(t: Double, op: FredholmOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    /**
     * Вторая производная правой части `f''(t)`; нужна семейству `xi^<0>`.
     * Для уравнения II рода равна `u*'' - d^2/dt^2 (K u*)`.
     */
    fun rhsExactDeriv2(t: Double, op: FredholmOperator): Double {
        val integralDD = op.applyDeriv2(t) { s -> exact(s) }
        return if (secondKind) exactDeriv2(t) - integralDD else integralDD
    }

    companion object {
        /**
         * Задача с решением из порождающего пространства: `K = e^{t-2s}`, `u* = t^2`.
         *
         * Поскольку `u*` принадлежит `span{1, t, t^2}`, совпадающему с полиномиальной
         * порождающей системой `phi^B`, метод обязан воспроизводить решение с машинной
         * точностью. Это делает задачу удобным индикатором ошибок реализации.
         *
         * Производные ядра выписаны полностью: `K_t = K`, `K_s = -2K`, `K_tt = K`,
         * а также `u*'' = 2`.
         */
        val F2span = FredholmProblem(
            name = "F2span",
            kernel = KernelF(
                k = { t, s -> Math.exp(t - 2.0 * s) },
                kT = { t, s -> Math.exp(t - 2.0 * s) },
                kTT = { t, s -> Math.exp(t - 2.0 * s) },
            ),
            exact = { t -> t * t },
            exactDeriv = { t -> 2.0 * t },
            secondKind = true,
            exactDeriv2 = { 2.0 },
        )

        /**
         * Задача с рациональным решением: `K = 1/(1+t+s)`, `u* = 1/(t+1)`.
         * Решение не лежит ни в одной из порождающих систем, поэтому задача служит
         * основным инструментом измерения порядка сходимости.
         */
        val F2 = FredholmProblem(
            name = "F2",
            kernel = KernelF(
                k = { t, s -> 1.0 / (1.0 + t + s) },
                kT = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
                kTT = { t, s -> 2.0 / ((1.0 + t + s) * (1.0 + t + s) * (1.0 + t + s)) },
            ),
            exact = { t -> 1.0 / (t + 1.0) },
            exactDeriv = { t -> -1.0 / ((t + 1.0) * (t + 1.0)) },
            secondKind = true,
            exactDeriv2 = { t -> 2.0 / ((t + 1.0) * (t + 1.0) * (t + 1.0)) },
        )

        /**
         * Задача с экспоненциальным решением: `K = e^{-(t-s)^2}`, `u* = e^t`.
         * Решение согласовано с гиперболической порождающей системой `phi^H`.
         */
        val F2exp = FredholmProblem(
            name = "F2exp",
            kernel = KernelF(
                k = { t, s -> Math.exp(-(t - s) * (t - s)) },
                kT = { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) },
                kTT = { t, s -> (4.0 * (t - s) * (t - s) - 2.0) * Math.exp(-(t - s) * (t - s)) },
            ),
            exact = { t -> Math.exp(t) },
            exactDeriv = { t -> Math.exp(t) },
            secondKind = true,
            exactDeriv2 = { t -> Math.exp(t) },
        )

        /**
         * Некорректная задача ПЕРВОГО рода: `K = e^{-(t-s)^2}`, `u* = e^t`.
         * Решается методом регуляризации (см. `solvers.fredholm.FirstKindSolver`).
         *
         * Производные ядра и решения выписаны полностью: `K_s = 2(t-s)K`,
         * `K_tt = (4(t-s)^2 - 2)K`, `u*'' = e^t`.
         */
        val F1 = FredholmProblem(
            name = "F1",
            kernel = KernelF(
                k = { t, s -> Math.exp(-(t - s) * (t - s)) },
                kT = { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) },
                kTT = { t, s -> (4.0 * (t - s) * (t - s) - 2.0) * Math.exp(-(t - s) * (t - s)) },
            ),
            exact = { t -> Math.exp(t) },
            exactDeriv = { t -> Math.exp(t) },
            secondKind = false,
            exactDeriv2 = { t -> Math.exp(t) },
        )
    }
}

/**
 * Создаёт решатель уравнения второго рода для модельной задачи.
 *
 * Удобство в том, что правая часть и её производные берутся из самой задачи
 * (вычисляются через точное решение), а множитель перед оператором равен единице.
 */
fun secondKindSolver(
    problem: FredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
): SecondKindSolver = SecondKindSolver(
    basis, funcs, op, cL = 1.0,
    fEff = { t -> problem.rhsExact(t, op) },
    fEffDeriv = { t -> problem.rhsExactDeriv(t, op) },
    fEffDeriv2 = { t -> problem.rhsExactDeriv2(t, op) },
)

/**
 * Создаёт решатель уравнения ПЕРВОГО рода для модельной задачи.
 *
 * @param alpha параметр регуляризации (см. [FirstKindSolver.DEFAULT_REGULARIZATION]).
 */
fun firstKindSolver(
    problem: FredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
    alpha: Double = FirstKindSolver.DEFAULT_REGULARIZATION,
): FirstKindSolver = FirstKindSolver(
    basis, funcs, op,
    rhs = { t -> problem.rhsExact(t, op) },
    rhsDeriv = { t -> problem.rhsExactDeriv(t, op) },
    rhsDeriv2 = { t -> problem.rhsExactDeriv2(t, op) },
    alpha = alpha,
)
