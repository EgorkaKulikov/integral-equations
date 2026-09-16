package problems.analytic

import solvers.fredholm.KernelF

// ============================================================================
// ANALYTICALLY EXACT PROBLEMS (the strongest verification baseline)
//
// The essential difference from `problems.fredholm.FredholmProblem` and
// `problems.volterra.VolterraProblem`: there the right-hand side `f` is built NUMERICALLY,
// by the quadrature of the project itself (`rhsExact` calls `op.apply`). Because of that the
// quadrature error enters both the baseline and the solution, partly cancelling out —
// the check ends up closed on the implementation being checked.
//
// Here both the exact solution `u*` and the right-hand side `f` are written out ANALYTICALLY:
// the integrals were taken by hand, the coefficients obtained by solving finite systems on paper.
// Not a single quantity is computed by the project quadrature, so the link "baseline error —
// solution error" is broken completely.
//
// Three independent sources of exact solutions:
//  1. DEGENERATE (separable) Fredholm kernels: `K = sum a_i(t) b_i(s)` reduces
//     the equation to a FINITE system of linear equations solvable in closed form
//     without any discretization whatsoever.
//  2. VOLTERRA CONVOLUTION KERNELS: `K(t,s) = k(t-s)` allows the Laplace transform
//     to be applied and the solution to be obtained in closed form.
//  3. METHOD OF MANUFACTURED SOLUTIONS: the solution is
//     prescribed arbitrarily, the right-hand side is computed analytically by substitution.
// ============================================================================

/**
 * Fredholm problem of the second kind `u(t) - \int_a^b K(t,s) u(s) ds = f(t)` with
 * ANALYTICALLY known exact solution and right-hand side.
 *
 * @param name short problem name for test messages.
 * @param kernel kernel `K(t,s)` together with its partial derivatives.
 * @param exact exact solution `u*(t)`, written in closed form.
 * @param exactDeriv first derivative `u*'(t)`.
 * @param exactDeriv2 second derivative `u*''(t)`.
 * @param rhs right-hand side `f(t)` — computed ANALYTICALLY, not by quadrature.
 * @param rhsDeriv first derivative `f'(t)`, analytic as well.
 * @param rhsDeriv2 second derivative `f''(t)`, analytic as well.
 * @param derivation verbal description of the derivation — it goes into test messages,
 *        so that on a mismatch one sees at once which computation to check.
 * @param spectralRadius spectral radius `rho(K)` of the integral operator.
 *
 *        Why it is needed: the equation itself is uniquely solvable for any `rho != 1`
 *        (which the closed-form solution confirms), but the schemes based
 *        on SIMPLE ITERATION (`kulkarniQuasi` for the quasi-interpolants `mu`, `lambda`;
 *        `combinedNystrom`) require `rho < 1` and diverge when `rho > 1`. This is a
 *        limitation OF THE SOLUTION METHOD, not of the problem, and tests must account for it
 *        explicitly rather than hide it behind a relaxed tolerance.
 *
 *        The values are obtained ANALYTICALLY: for a degenerate kernel of rank `m`
 *        the operator is finite-dimensional, and its nonzero eigenvalues coincide with
 *        the eigenvalues of the moment matrix `G_{ij} = \int_0^1 b_i(s) a_j(s) ds`.
 */
class AnalyticFredholmProblem(
    val name: String,
    val kernel: KernelF,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val exactDeriv2: (Double) -> Double,
    val rhs: (Double) -> Double,
    val rhsDeriv: (Double) -> Double,
    val rhsDeriv2: (Double) -> Double,
    val derivation: String,
    val spectralRadius: Double,
) {
    /**
     * Whether the problem is suitable for schemes solved by simple iteration.
     *
     * The threshold is taken with a margin (0.9, not 1.0): near one the iteration formally
     * converges, but so slowly that it runs into the step limit.
     */
    val supportsFixedPointSchemes: Boolean get() = spectralRadius < 0.9

    companion object {
        /**
         * RANK 1, the example from the assignment statement: `K(t,s) = t*s`, `f(t) = t`.
         *
         * Derivation (entirely on paper). The kernel is degenerate with a single term
         * `a(t) = t`, `b(s) = s`, hence
         *
         *     u(t) = f(t) + t * c,   c = \int_0^1 s u(s) ds.
         *
         * Substituting `u(s) = s + c s = s(1 + c)` into the definition of `c`:
         *
         *     c = \int_0^1 s * s(1+c) ds = (1+c) \int_0^1 s^2 ds = (1+c)/3,
         *     3c = 1 + c  =>  c = 1/2.
         *
         * Result: `u*(t) = t (1 + 1/2) = (3/2) t`.
         *
         * Check by substitution: `u* - K u* = (3/2)t - t \int_0^1 s (3/2)s ds
         * = (3/2)t - t (3/2)(1/3) = (3/2)t - (1/2)t = t = f`.
         *
         * SPECIAL FEATURE: the solution is linear, i.e. it lies in `span{1, t, t^2}` —
         * the polynomial generating system `phi^B`. Hence on the basis `B` the method
         * must reproduce it to MACHINE precision; the convergence order is not defined
         * here. The problem is used as an accuracy check, not an order check.
         */
        val SEPARABLE_RANK1_LINEAR = AnalyticFredholmProblem(
            name = "A-sep1-lin",
            kernel = KernelF(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> 1.5 * t },
            exactDeriv = { 1.5 },
            exactDeriv2 = { 0.0 },
            rhs = { t -> t },
            rhsDeriv = { 1.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=t*s, f=t; c=int_0^1 s u ds=1/2; u*=(3/2)t",
            // Rank 1 kernel: the single eigenvalue = int_0^1 s*s ds = 1/3.
            spectralRadius = 1.0 / 3.0,
        )

        /**
         * RANK 1 with a transcendental solution: `K(t,s) = t*s`, `f(t) = e^t`.
         *
         * Derivation. As above, `u(t) = e^t + c t`, where `c = \int_0^1 s u(s) ds`.
         * Substitution gives
         *
         *     c = \int_0^1 s e^s ds + c \int_0^1 s^2 ds = 1 + c/3,
         *
         * since `\int_0^1 s e^s ds = [s e^s - e^s]_0^1 = (e - e) - (0 - 1) = 1`.
         * Hence `(2/3) c = 1`, i.e. `c = 3/2` and
         *
         *     u*(t) = e^t + (3/2) t.
         *
         * Check: `K u* = t (\int_0^1 s e^s ds + (3/2)\int_0^1 s^2 ds)
         * = t (1 + 1/2) = (3/2) t`, so `u* - K u* = e^t = f`.
         *
         * Unlike [SEPARABLE_RANK1_LINEAR], the solution does NOT lie in `span{1,t,t^2}`,
         * so the problem is suitable for measuring the convergence order.
         */
        val SEPARABLE_RANK1 = AnalyticFredholmProblem(
            name = "A-sep1",
            kernel = KernelF(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) + 1.5 * t },
            exactDeriv = { t -> Math.exp(t) + 1.5 },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { t -> Math.exp(t) },
            rhsDeriv = { t -> Math.exp(t) },
            rhsDeriv2 = { t -> Math.exp(t) },
            derivation = "K=t*s, f=e^t; c=1+c/3 => c=3/2; u*=e^t+(3/2)t",
            // The same kernel as in SEPARABLE_RANK1_LINEAR: rho = 1/3.
            spectralRadius = 1.0 / 3.0,
        )

        /**
         * RANK 2: `K(t,s) = 1 + t*s`, `f(t) = e^t`.
         *
         * Derivation. The kernel is degenerate with two terms (`a_1 = 1, b_1 = 1` and
         * `a_2 = t, b_2 = s`), hence
         *
         *     u(t) = e^t + c_0 + c_1 t,
         *     c_0 = \int_0^1 u(s) ds,   c_1 = \int_0^1 s u(s) ds.
         *
         * Substituting `u` into the definitions gives the system
         *
         *     c_0 = (e-1) + c_0 + c_1/2,          (1)
         *     c_1 = 1     + c_0/2 + c_1/3.        (2)
         *
         * From (1) at once `0 = (e-1) + c_1/2`, i.e. `c_1 = -2(e-1)`.
         * Substituting into (2): `-2(e-1) = 1 + c_0/2 - (2/3)(e-1)`, whence
         * `c_0/2 = -(4/3)(e-1) - 1` and `c_0 = -(8/3)(e-1) - 2`.
         *
         * Result: `u*(t) = e^t - (8/3)(e-1) - 2 - 2(e-1) t`.
         *
         * Check: `K u* = c_0 + c_1 t` by construction of the coefficients, so
         * `u* - K u* = e^t = f` identically.
         *
         * Table integrals used: `\int_0^1 e^s ds = e - 1`,
         * `\int_0^1 s e^s ds = 1`.
         */
        val SEPARABLE_RANK2 = AnalyticFredholmProblem(
            name = "A-sep2",
            kernel = KernelF(
                k = { t, s -> 1.0 + t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) + RANK2_C0 + RANK2_C1 * t },
            exactDeriv = { t -> Math.exp(t) + RANK2_C1 },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { t -> Math.exp(t) },
            rhsDeriv = { t -> Math.exp(t) },
            rhsDeriv2 = { t -> Math.exp(t) },
            derivation = "K=1+t*s, f=e^t; c_1=-2(e-1), c_0=-(8/3)(e-1)-2",
            // Moment matrix [[1, 1/2], [1/2, 1/3]]; the eigenvalues are the roots of
            // lambda^2 - (4/3)lambda + 1/12 = 0, i.e. lambda = (2/3) +- sqrt(1/3 - 1/12 + 1/9).
            // The larger root is rho = 2/3 + sqrt(13)/6 ~ 1.2676 > 1.
            spectralRadius = 2.0 / 3.0 + Math.sqrt(13.0) / 6.0,
        )

        /**
         * RANK 3: `K(t,s) = 1 + t*s + t^2 s^2`, `f(t) = t^3`.
         *
         * Derivation. The kernel is degenerate with three terms, hence
         *
         *     u(t) = t^3 + c_0 + c_1 t + c_2 t^2,
         *     c_k = \int_0^1 s^k u(s) ds,  k = 0,1,2.
         *
         * Denoting the moments of the right-hand side by `m_k = \int_0^1 s^k * s^3 ds`, i.e.
         * `m_0 = 1/4`, `m_1 = 1/5`, `m_2 = 1/6`, and substituting `u` into the definitions of `c_k`,
         * we obtain the system (the terms being `\int_0^1 s^k * s^j ds = 1/(k+j+1)`):
         *
         *     c_0 = m_0 + c_0     + c_1/2 + c_2/3,
         *     c_1 = m_1 + c_0/2   + c_1/3 + c_2/4,
         *     c_2 = m_2 + c_0/3   + c_1/4 + c_2/5.
         *
         * The first equation degenerates into `0 = 1/4 + c_1/2 + c_2/3`, whence
         * `c_1 = -1/2 - (2/3) c_2`. Substituting into the second gives
         * `-c_0/2 - (25/36) c_2 = 8/15`, i.e. `c_0 = -16/15 - (25/18) c_2`.
         * Substituting both into the third leads to `(193/135) c_2 = -113/360`, whence
         *
         *     c_2 = -339/1544,
         *     c_1 = -273/772,
         *     c_0 = -11761/15440.
         *
         * Result: `u*(t) = t^3 - (339/1544) t^2 - (273/772) t - 11761/15440`.
         *
         * Control substitutions (done by hand):
         *  - at `t = 0`: `u*(0) = c_0`, `K u*(0) = c_0`, so `f(0) = 0 = 0^3`;
         *  - at `t = 1`: `u*(1) - K u*(1) = 1 = 1^3`.
         *
         * The solution is a polynomial of degree THREE, so it does not lie in
         * `span{1, t, t^2}` and the problem is suitable for measuring the convergence order.
         */
        val SEPARABLE_RANK3 = AnalyticFredholmProblem(
            name = "A-sep3",
            kernel = KernelF(
                k = { t, s -> 1.0 + t * s + t * t * s * s },
                kT = { t, s -> s + 2.0 * t * s * s },
                kTT = { _, s -> 2.0 * s * s },
            ),
            exact = { t -> t * t * t + RANK3_C2 * t * t + RANK3_C1 * t + RANK3_C0 },
            exactDeriv = { t -> 3.0 * t * t + 2.0 * RANK3_C2 * t + RANK3_C1 },
            exactDeriv2 = { t -> 6.0 * t + 2.0 * RANK3_C2 },
            rhs = { t -> t * t * t },
            rhsDeriv = { t -> 3.0 * t * t },
            rhsDeriv2 = { t -> 6.0 * t },
            derivation = "K=1+ts+t^2s^2, f=t^3; c_2=-339/1544, c_1=-273/772, c_0=-11761/15440",
            // The moment matrix is the 3x3 Hilbert matrix; its largest eigenvalue
            // is the well-known tabulated value ~1.40832 > 1.
            spectralRadius = 1.4083189271236535,
        )

        /**
         * METHOD OF MANUFACTURED SOLUTIONS (MMS) for the Fredholm equation:
         * kernel `K(t,s) = 1/(1+t+s)`, PRESCRIBED solution `u*(t) = t^3`.
         *
         * Here the order is reversed: the solution is chosen arbitrarily, and the right-hand side
         * is computed by substitution. The key requirement of the assignment is to take the integral
         * ANALYTICALLY, not by quadrature.
         *
         * Computation of `\int_0^1 s^3/(a+s) ds` at `a = 1+t`. Polynomial division
         * (the identity `s^3 = (s+a)(s^2 - a s + a^2) - a^3`) gives
         *
         *     s^3/(a+s) = s^2 - a s + a^2 - a^3/(a+s),
         *
         * whence
         *
         *     \int_0^1 s^3/(a+s) ds = 1/3 - a/2 + a^2 - a^3 ln((a+1)/a).
         *
         * Consequently
         *
         *     f(t) = t^3 - 1/3 + a/2 - a^2 + a^3 ln((a+1)/a),   a = 1 + t.
         *
         * Derivatives (obtained by differentiating in `t`, with `da/dt = 1` and
         * `d/da ln((a+1)/a) = -1/(a(a+1))`):
         *
         *     f'(t)  = 3t^2 + 1/2 - 2a + 3a^2 ln((a+1)/a) - a^2/(a+1),
         *     f''(t) = 6t - 2 + 6a ln((a+1)/a) - 3a/(a+1) - (a^2+2a)/(a+1)^2.
         *
         * Control at `t = 0` (`a = 1`): `f(0) = -1/3 + 1/2 - 1 + ln 2 ~ -0.14019`,
         * which agrees with `-\int_0^1 s^3/(1+s) ds = -(1/3 - 1/2 + 1 - ln 2)`.
         *
         * Unlike the separable problems, the kernel here is NOT degenerate, so the check
         * exercises a different path in the matrix assembly.
         */
        val MANUFACTURED = AnalyticFredholmProblem(
            name = "A-mms-F",
            kernel = KernelF(
                k = { t, s -> 1.0 / (1.0 + t + s) },
                kT = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
                kTT = { t, s -> 2.0 / ((1.0 + t + s) * (1.0 + t + s) * (1.0 + t + s)) },
            ),
            exact = { t -> t * t * t },
            exactDeriv = { t -> 3.0 * t * t },
            exactDeriv2 = { t -> 6.0 * t },
            rhs = { t ->
                val a = 1.0 + t
                t * t * t - 1.0 / 3.0 + a / 2.0 - a * a + a * a * a * Math.log((a + 1.0) / a)
            },
            rhsDeriv = { t ->
                val a = 1.0 + t
                3.0 * t * t + 0.5 - 2.0 * a + 3.0 * a * a * Math.log((a + 1.0) / a) - a * a / (a + 1.0)
            },
            rhsDeriv2 = { t ->
                val a = 1.0 + t
                6.0 * t - 2.0 + 6.0 * a * Math.log((a + 1.0) / a) -
                    3.0 * a / (a + 1.0) - (a * a + 2.0 * a) / ((a + 1.0) * (a + 1.0))
            },
            derivation = "MMS: K=1/(1+t+s), u*=t^3; f=t^3-1/3+a/2-a^2+a^3 ln((a+1)/a), a=1+t",
            // Upper estimate by the C[0,1] norm: rho <= max_t int_0^1 ds/(1+t+s) = ln 2 ~ 0.693 < 1.
            spectralRadius = Math.log(2.0),
        )

        /** All analytic Fredholm problems. */
        val ALL = listOf(
            SEPARABLE_RANK1_LINEAR,
            SEPARABLE_RANK1,
            SEPARABLE_RANK2,
            SEPARABLE_RANK3,
            MANUFACTURED,
        )

        /**
         * Problems suitable for measuring the convergence ORDER: their solutions do not lie
         * in the `span{1, t, t^2}` of the polynomial generating system.
         */
        val CONVERGENT = listOf(SEPARABLE_RANK1, SEPARABLE_RANK2, SEPARABLE_RANK3, MANUFACTURED)

        /**
         * Problems with `rho(K) < 1` — the only ones on which the schemes implemented
         * by SIMPLE ITERATION are applicable (Kulkarni scheme for the quasi-interpolants
         * `mu` and `lambda`, combined Nyström).
         *
         * The problems [SEPARABLE_RANK2] and [SEPARABLE_RANK3] are NOT included: they have `rho > 1`.
         * To be clear — the problems themselves are well-posed and uniquely solvable (which the
         * closed-form solution proves), and the schemes based on DIRECTLY solving the linear system (`base`,
         * `sloan`, `kulkarni` for projectors) solve them without any trouble.
         */
        val FIXED_POINT_SAFE = ALL.filter { it.supportsFixedPointSchemes }

        // --- Exact constants computed by hand (see the KDoc of the corresponding problems) ---

        /** `c_1 = -2(e-1)` for the rank 2 problem. */
        private val RANK2_C1 = -2.0 * (Math.E - 1.0)

        /** `c_0 = -(8/3)(e-1) - 2` for the rank 2 problem. */
        private val RANK2_C0 = -(8.0 / 3.0) * (Math.E - 1.0) - 2.0

        /** `c_2 = -339/1544` for the rank 3 problem. */
        private const val RANK3_C2 = -339.0 / 1544.0

        /** `c_1 = -273/772` for the rank 3 problem. */
        private const val RANK3_C1 = -273.0 / 772.0

        /** `c_0 = -11761/15440` for the rank 3 problem. */
        private const val RANK3_C0 = -11761.0 / 15440.0
    }
}
