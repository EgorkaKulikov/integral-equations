package problems.analytic

import solvers.volterra.KernelV

/**
 * Задача Вольтерры II рода `u(t) - \int_a^t K(t,s) u(s) ds = f(t)` с АНАЛИТИЧЕСКИ
 * известными точным решением и правой частью.
 *
 * Смысл полей совпадает с [AnalyticFredholmProblem]; отличается только оператор
 * (переменный верхний предел интегрирования).
 *
 * ПОЧЕМУ ЗДЕСЬ НЕТ ПОЛЯ `spectralRadius` (в отличие от [AnalyticFredholmProblem]).
 * Оператор Вольтерры с ограниченным ядром КВАЗИНИЛЬПОТЕНТЕН: его спектральный
 * радиус равен нулю при ЛЮБОМ ядре (оценка `|V^m u| <= (M(b-a))^m/m! * |u|`).
 * Поэтому простая итерация сходится всегда, и ограничение, существенное для
 * задач Фредгольма, здесь отсутствует по построению.
 */
class AnalyticVolterraProblem(
    val name: String,
    val kernel: KernelV,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val exactDeriv2: (Double) -> Double,
    val rhs: (Double) -> Double,
    val rhsDeriv: (Double) -> Double,
    val rhsDeriv2: (Double) -> Double,
    val derivation: String,
) {
    companion object {
        /**
         * ЯДРО СВЁРТКИ, пример из постановки задания: `K(t,s) = 1`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Ядро зависит только от разности,
         * `k(tau) = 1`, поэтому уравнение `u - k * u = f` (звёздочка — свёртка)
         * переходит в алгебраическое:
         *
         *     U(p) - \hat k(p) U(p) = F(p),   U(p) = F(p) / (1 - \hat k(p)).
         *
         * Здесь `\hat k(p) = 1/p` и `F(p) = 1/p`, значит
         *
         *     U(p) = (1/p) / (1 - 1/p) = (1/p) * p/(p-1) = 1/(p-1),
         *
         * и обратное преобразование даёт `u*(t) = e^t`.
         *
         * Прямая проверка: `\int_0^t e^s ds = e^t - 1`, поэтому
         * `u* - V u* = e^t - (e^t - 1) = 1 = f`.
         */
        val CONVOLUTION_CONST = AnalyticVolterraProblem(
            name = "A-conv-1",
            kernel = KernelV(
                k = { _, _ -> 1.0 },
                kT = { _, _ -> 0.0 },
                kS = { _, _ -> 0.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) },
            exactDeriv = { t -> Math.exp(t) },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=1, f=1; U=(1/p)/(1-1/p)=1/(p-1); u*=e^t",
        )

        /**
         * ЯДРО СВЁРТКИ `K(t,s) = t - s`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Здесь `k(tau) = tau`, поэтому
         * `\hat k(p) = 1/p^2`, а `F(p) = 1/p`. Тогда
         *
         *     U(p) = (1/p) / (1 - 1/p^2) = (1/p) * p^2/(p^2 - 1) = p/(p^2 - 1),
         *
         * что является образом гиперболического косинуса: `u*(t) = cosh t`.
         *
         * Прямая проверка (интегрирование по частям):
         *
         *     \int_0^t (t-s) cosh s ds = t sinh t - [s sinh s - cosh s]_0^t
         *                              = t sinh t - t sinh t + cosh t - 1
         *                              = cosh t - 1,
         *
         * поэтому `u* - V u* = cosh t - (cosh t - 1) = 1 = f`.
         */
        val CONVOLUTION_LINEAR = AnalyticVolterraProblem(
            name = "A-conv-lin",
            kernel = KernelV(
                k = { t, s -> t - s },
                kT = { _, _ -> 1.0 },
                kS = { _, _ -> -1.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cosh(t) },
            exactDeriv = { t -> Math.sinh(t) },
            exactDeriv2 = { t -> Math.cosh(t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=t-s, f=1; U=(1/p)/(1-1/p^2)=p/(p^2-1); u*=cosh t",
        )

        /**
         * ЯДРО СВЁРТКИ `K(t,s) = e^{t-s}`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Здесь `k(tau) = e^{tau}`, поэтому
         * `\hat k(p) = 1/(p-1)`, а `F(p) = 1/p`. Тогда
         *
         *     U(p) = (1/p) / (1 - 1/(p-1)) = (1/p) * (p-1)/(p-2) = (p-1)/(p(p-2)).
         *
         * Разложение на простейшие дроби: `(p-1)/(p(p-2)) = A/p + B/(p-2)` с
         * `A(p-2) + Bp = p - 1`. При `p = 0` получаем `-2A = -1`, то есть `A = 1/2`;
         * при `p = 2` получаем `2B = 1`, то есть `B = 1/2`. Значит
         *
         *     U(p) = (1/2)/p + (1/2)/(p-2)   =>   u*(t) = (1 + e^{2t}) / 2.
         *
         * Прямая проверка:
         *
         *     \int_0^t e^{t-s} (1+e^{2s})/2 ds = (e^t/2) \int_0^t (e^{-s} + e^{s}) ds
         *                                      = (e^t/2)(e^t - e^{-t}) = (e^{2t} - 1)/2,
         *
         * поэтому `u* - V u* = (1 + e^{2t})/2 - (e^{2t} - 1)/2 = 1 = f`.
         */
        val CONVOLUTION_EXP = AnalyticVolterraProblem(
            name = "A-conv-exp",
            kernel = KernelV(
                k = { t, s -> Math.exp(t - s) },
                kT = { t, s -> Math.exp(t - s) },
                kS = { t, s -> -Math.exp(t - s) },
                kTT = { t, s -> Math.exp(t - s) },
            ),
            exact = { t -> 0.5 * (1.0 + Math.exp(2.0 * t)) },
            exactDeriv = { t -> Math.exp(2.0 * t) },
            exactDeriv2 = { t -> 2.0 * Math.exp(2.0 * t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=e^{t-s}, f=1; U=(p-1)/(p(p-2)); u*=(1+e^{2t})/2",
        )

        /**
         * МЕТОД ПРОИЗВОДНЫХ РЕШЕНИЙ (MMS) для уравнения Вольтерры:
         * ядро `K(t,s) = t*s`, ЗАДАННОЕ решение `u*(t) = cos t`.
         *
         * Ядро НЕ является ядром свёртки, поэтому путь вывода полностью независим
         * от лапласовых задач выше.
         *
         * Вычисление образа (интеграл взят по частям):
         *
         *     (V u*)(t) = t \int_0^t s cos s ds = t [s sin s + cos s]_0^t
         *               = t (t sin t + cos t - 1)
         *               = t^2 sin t + t cos t - t.
         *
         * Отсюда правая часть
         *
         *     f(t) = cos t - t^2 sin t - t cos t + t.
         *
         * Производные (прямое дифференцирование):
         *
         *     f'(t)  = -sin t - t sin t - t^2 cos t - cos t + 1,
         *     f''(t) = -cos t - 3 t cos t + t^2 sin t.
         *
         * Решение `cos t` не лежит в `span{1, t, t^2}`, поэтому задача пригодна
         * для измерения порядка сходимости.
         */
        val MANUFACTURED = AnalyticVolterraProblem(
            name = "A-mms-V",
            kernel = KernelV(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kS = { t, _ -> t },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cos(t) },
            exactDeriv = { t -> -Math.sin(t) },
            exactDeriv2 = { t -> -Math.cos(t) },
            rhs = { t -> Math.cos(t) - t * t * Math.sin(t) - t * Math.cos(t) + t },
            rhsDeriv = { t ->
                -Math.sin(t) - t * Math.sin(t) - t * t * Math.cos(t) - Math.cos(t) + 1.0
            },
            rhsDeriv2 = { t -> -Math.cos(t) - 3.0 * t * Math.cos(t) + t * t * Math.sin(t) },
            derivation = "MMS: K=t*s, u*=cos t; f=cos t - t^2 sin t - t cos t + t",
        )

        /** Все аналитические задачи Вольтерры. */
        val ALL = listOf(CONVOLUTION_CONST, CONVOLUTION_LINEAR, CONVOLUTION_EXP, MANUFACTURED)
    }
}
