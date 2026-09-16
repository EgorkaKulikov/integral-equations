package problems.analytic

import solvers.volterra.KernelV

/**
 * Volterra problem of the second kind `u(t) - \int_a^t K(t,s) u(s) ds = f(t)` with ANALYTICALLY
 * known exact solution and right-hand side.
 *
 * The meaning of the fields is the same as in [AnalyticFredholmProblem]; only the operator
 * differs (variable upper limit of integration).
 *
 * WHY THERE IS NO `spectralRadius` FIELD HERE (unlike [AnalyticFredholmProblem]).
 * A Volterra operator with a bounded kernel is QUASINILPOTENT: its spectral
 * radius is zero for ANY kernel (the estimate `|V^m u| <= (M(b-a))^m/m! * |u|`).
 * Hence simple iteration always converges, and the restriction essential for
 * Fredholm problems is absent here by construction.
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
         * CONVOLUTION KERNEL, the example from the assignment statement: `K(t,s) = 1`, `f(t) = 1`.
         *
         * Derivation by the Laplace transform. The kernel depends only on the difference,
         * `k(tau) = 1`, so the equation `u - k * u = f` (the asterisk is convolution)
         * turns into an algebraic one:
         *
         *     U(p) - \hat k(p) U(p) = F(p),   U(p) = F(p) / (1 - \hat k(p)).
         *
         * Here `\hat k(p) = 1/p` and `F(p) = 1/p`, hence
         *
         *     U(p) = (1/p) / (1 - 1/p) = (1/p) * p/(p-1) = 1/(p-1),
         *
         * and the inverse transform gives `u*(t) = e^t`.
         *
         * Direct check: `\int_0^t e^s ds = e^t - 1`, hence
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
         * CONVOLUTION KERNEL `K(t,s) = t - s`, `f(t) = 1`.
         *
         * Derivation by the Laplace transform. Here `k(tau) = tau`, so
         * `\hat k(p) = 1/p^2` and `F(p) = 1/p`. Then
         *
         *     U(p) = (1/p) / (1 - 1/p^2) = (1/p) * p^2/(p^2 - 1) = p/(p^2 - 1),
         *
         * which is the transform of the hyperbolic cosine: `u*(t) = cosh t`.
         *
         * Direct check (integration by parts):
         *
         *     \int_0^t (t-s) cosh s ds = t sinh t - [s sinh s - cosh s]_0^t
         *                              = t sinh t - t sinh t + cosh t - 1
         *                              = cosh t - 1,
         *
         * hence `u* - V u* = cosh t - (cosh t - 1) = 1 = f`.
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
         * CONVOLUTION KERNEL `K(t,s) = e^{t-s}`, `f(t) = 1`.
         *
         * Derivation by the Laplace transform. Here `k(tau) = e^{tau}`, so
         * `\hat k(p) = 1/(p-1)` and `F(p) = 1/p`. Then
         *
         *     U(p) = (1/p) / (1 - 1/(p-1)) = (1/p) * (p-1)/(p-2) = (p-1)/(p(p-2)).
         *
         * Partial fraction decomposition: `(p-1)/(p(p-2)) = A/p + B/(p-2)` with
         * `A(p-2) + Bp = p - 1`. At `p = 0` we get `-2A = -1`, i.e. `A = 1/2`;
         * at `p = 2` we get `2B = 1`, i.e. `B = 1/2`. Hence
         *
         *     U(p) = (1/2)/p + (1/2)/(p-2)   =>   u*(t) = (1 + e^{2t}) / 2.
         *
         * Direct check:
         *
         *     \int_0^t e^{t-s} (1+e^{2s})/2 ds = (e^t/2) \int_0^t (e^{-s} + e^{s}) ds
         *                                      = (e^t/2)(e^t - e^{-t}) = (e^{2t} - 1)/2,
         *
         * hence `u* - V u* = (1 + e^{2t})/2 - (e^{2t} - 1)/2 = 1 = f`.
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
         * METHOD OF MANUFACTURED SOLUTIONS (MMS) for the Volterra equation:
         * kernel `K(t,s) = t*s`, PRESCRIBED solution `u*(t) = cos t`.
         *
         * The kernel is NOT a convolution kernel, so the derivation path is fully independent
         * of the Laplace problems above.
         *
         * Computation of the image (the integral is taken by parts):
         *
         *     (V u*)(t) = t \int_0^t s cos s ds = t [s sin s + cos s]_0^t
         *               = t (t sin t + cos t - 1)
         *               = t^2 sin t + t cos t - t.
         *
         * Hence the right-hand side
         *
         *     f(t) = cos t - t^2 sin t - t cos t + t.
         *
         * Derivatives (direct differentiation):
         *
         *     f'(t)  = -sin t - t sin t - t^2 cos t - cos t + 1,
         *     f''(t) = -cos t - 3 t cos t + t^2 sin t.
         *
         * The solution `cos t` does not lie in `span{1, t, t^2}`, so the problem is suitable
         * for measuring the convergence order.
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

        /** All analytic Volterra problems. */
        val ALL = listOf(CONVOLUTION_CONST, CONVOLUTION_LINEAR, CONVOLUTION_EXP, MANUFACTURED)
    }
}
