package problems.volterra

import numerics.MinimalSplineBasis
import numerics.functionals.FunctionalFamily
import solvers.core.RhsWithDerivatives
import solvers.volterra.KernelV
import solvers.volterra.VolterraFirstKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * Модельная задача для линейного уравнения Вольтерры: ядро, точное решение и род
 * уравнения. Правая часть НЕ задаётся явно, а строится из точного решения численно
 * (квадратурой) — так тестовые данные остаются согласованными с оператором и
 * квадратурной формулой, а не «зашитыми» константами.
 *
 * Соотношения между точным решением `u*` и правой частью `f`:
 *  - уравнение II рода: `f = u* - V u*`;
 *  - уравнение I рода:  `f = V u*`,
 *
 * где `(V u)(t) = \int_a^t K(t,s) u(s) ds` — оператор с ПЕРЕМЕННЫМ верхним пределом.
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
class VolterraProblem(
    val name: String,
    val kernel: KernelV,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val exactDeriv2: (Double) -> Double = { 0.0 },
) {
    /**
     * Точная правая часть `f(t)`, вычисленная через оператор [op].
     *
     * @param t точка вычисления.
     * @param op оператор Вольтерры, построенный на том же ядре и нужной сетке.
     */
    fun rhsExact(t: Double, op: VolterraOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /**
     * Первая производная правой части `f'(t)`; нужна семействам функционалов,
     * использующим производную. Вычисляется по правилу Лейбница, включая граничный
     * член `K(t,t) u*(t)`, специфичный для оператора Вольтерры.
     */
    fun rhsExactDeriv(t: Double, op: VolterraOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    /**
     * Вторая производная правой части `f''(t)`; нужна семейству `xi^<0>`.
     * Вычисление `(V u*)''` требует и самого решения, и его первой производной
     * (из-за члена `K(t,t) u*'(t)`), поэтому в оператор передаются обе функции.
     */
    fun rhsExactDeriv2(t: Double, op: VolterraOperator): Double {
        val integralDD = op.applyDeriv2(t, { s -> exact(s) }, { s -> exactDeriv(s) })
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
         * Производные ядра выписаны ПОЛНОСТЬЮ (`K_t = K`, `K_s = -2K`, `K_tt = K`), как и
         * `u*'' = 2`. Ранее `kS`/`kTT`/`exactDeriv2` оставались нулевыми по умолчанию при
         * ненулевых истинных значениях, и вторая производная `(Vu)''` вычислялась неверно
         * для семейства `xi^<0>` — молча, без какой-либо диагностики.
         */
        val V2span = VolterraProblem(
            name = "V2span",
            kernel = KernelV(
                k = { t, s -> Math.exp(t - 2.0 * s) },
                kT = { t, s -> Math.exp(t - 2.0 * s) },
                kS = { t, s -> -2.0 * Math.exp(t - 2.0 * s) },
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
        val V2 = VolterraProblem(
            name = "V2",
            kernel = KernelV(
                k = { t, s -> 1.0 / (1.0 + t + s) },
                kT = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
                kS = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
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
         * Диагональ ядра `K(t,t) = 1` отлична от нуля.
         */
        val V2exp = VolterraProblem(
            name = "V2exp",
            kernel = KernelV(
                k = { t, s -> Math.exp(-(t - s) * (t - s)) },
                kT = { t, s -> -2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) },
                kS = { t, s -> 2.0 * (t - s) * Math.exp(-(t - s) * (t - s)) },
                kTT = { t, s -> (4.0 * (t - s) * (t - s) - 2.0) * Math.exp(-(t - s) * (t - s)) },
            ),
            exact = { t -> Math.exp(t) },
            exactDeriv = { t -> Math.exp(t) },
            secondKind = true,
            exactDeriv2 = { t -> Math.exp(t) },
        )

        /**
         * Задача со сглаживающим ядром: `K = t - s`, `u* = cos t`.
         *
         * Диагональ ядра обращается в ноль (`K(t,t) = 0`), что усиливает сглаживающее
         * действие оператора. На этой задаче одношаговая схема Кулкарни численно
         * оказывается заметно точнее итерации Слоана — наблюдаемые порядки составляют
         * примерно 3 (базовая схема), 4 (Слоан), 5 (Кулкарни) и 6 (итерированный
         * Кулкарни). Это ЧИСЛЕННОЕ НАБЛЮДЕНИЕ, а не доказанный результат.
         *
         * Внимание: из-за `K(t,t) = 0` задача НЕ пригодна для решателя уравнения
         * первого рода — редукция дифференцированием требует `K(t,t) != 0`.
         */
        val V2win = VolterraProblem(
            name = "V2win",
            kernel = KernelV(
                k = { t, s -> t - s },
                kT = { _, _ -> 1.0 },
                kS = { _, _ -> -1.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cos(t) },
            exactDeriv = { t -> -Math.sin(t) },
            secondKind = true,
            exactDeriv2 = { t -> -Math.cos(t) },
        )

        /**
         * Задача ПЕРВОГО рода: `K = 1 + t - s`, `u* = cos t`.
         *
         * В отличие от уравнения Фредгольма первого рода (некорректного), уравнение
         * Вольтерры первого рода при `K(t,t) != 0` корректно и сводится к уравнению
         * второго рода однократным дифференцированием (случай `m = 1`).
         * Здесь `K(t,t) = 1`, поэтому редукция применима.
         */
        val V1 = VolterraProblem(
            name = "V1",
            kernel = KernelV(
                k = { t, s -> 1.0 + t - s },
                kT = { _, _ -> 1.0 },
                kS = { _, _ -> -1.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cos(t) },
            exactDeriv = { t -> -Math.sin(t) },
            secondKind = false,
            exactDeriv2 = { t -> -Math.cos(t) },
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
    problem: VolterraProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: VolterraOperator,
): VolterraSecondKindSolver = VolterraSecondKindSolver(
    basis, funcs, op, cL = 1.0,
    rhs = RhsWithDerivatives(
        value = { t -> problem.rhsExact(t, op) },
        deriv = { t -> problem.rhsExactDeriv(t, op) },
        deriv2 = { t -> problem.rhsExactDeriv2(t, op) },
    ),
)

/**
 * Создаёт решатель уравнения ПЕРВОГО рода для модельной задачи (сведение к уравнению
 * второго рода дифференцированием).
 *
 * Ядро и правая часть редуцированного уравнения строятся из исходной задачи:
 * решателю передаются диагональ ядра `K(t,t)`, производная ядра `K_t(t,s)`, а также
 * правая часть `f'(t)` и точное решение с производной — последние нужны, чтобы
 * численно дифференцировать только малый остаток, а не всю правую часть
 * (подробности в KDoc [VolterraFirstKindSolver]).
 *
 * @throws IllegalArgumentException если `K(t,t)` обращается в ноль либо выбрано
 *         семейство функционалов, требующее второй производной.
 */
fun firstKindSolver(
    problem: VolterraProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: VolterraOperator,
): VolterraFirstKindSolver = VolterraFirstKindSolver(
    basis, funcs,
    kernel = problem.kernel,
    rhsDeriv = { t -> problem.rhsExactDeriv(t, op) },
    smoothPart = { t -> problem.exact(t) },
    smoothPartDeriv = { t -> problem.exactDeriv(t) },
)
