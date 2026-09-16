package problems.volterra

import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily
import solvers.core.RhsWithDerivatives
import solvers.volterra.KernelV
import solvers.volterra.VolterraFirstKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * Model problem for the linear Volterra equation: kernel, exact solution and the kind of
 * the equation. The right-hand side is NOT given explicitly but built from the exact solution
 * numerically (by quadrature) — this keeps the test data consistent with the operator and
 * the quadrature rule instead of hard-wiring constants.
 *
 * Relations between the exact solution `u*` and the right-hand side `f`:
 *  - equation of the second kind: `f = u* - V u*`;
 *  - equation of the first kind:  `f = V u*`,
 *
 * where `(V u)(t) = \int_a^t K(t,s) u(s) ds` is the operator with a VARIABLE upper limit.
 *
 * The integration domain is defined not here but by the grid [splines.Grid] passed
 * to the operator: the problem describes only the kernel and the solution.
 *
 * @param name short problem name used in tables and test messages.
 * @param kernel kernel `K(t,s)` together with its analytic partial derivatives.
 * @param exact exact solution `u*(t)` — the baseline for computing the error.
 * @param exactDeriv first derivative of the exact solution `u*'(t)`; required
 *        by the de Boor–Fix functional families `xi^<1>`, `xi^<2>`.
 * @param secondKind `true` — equation of the second kind, `false` — of the first.
 * @param exactDeriv2 second derivative `u*''(t)`; required by the family `xi^<0>`.
 *        The default value (zero) is admissible ONLY when the second derivative
 *        is indeed zero, otherwise the family `xi^<0>` silently gets a wrong
 *        right-hand side.
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
     * Exact right-hand side `f(t)`, computed through the operator [op].
     *
     * @param t evaluation point.
     * @param op Volterra operator built on the same kernel and the required grid.
     */
    fun rhsExact(t: Double, op: VolterraOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /**
     * First derivative of the right-hand side `f'(t)`; needed by the functional families
     * that use the derivative. Computed by the Leibniz rule, including the boundary
     * term `K(t,t) u*(t)` specific to the Volterra operator.
     */
    fun rhsExactDeriv(t: Double, op: VolterraOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    /**
     * Second derivative of the right-hand side `f''(t)`; needed by the family `xi^<0>`.
     * Computing `(V u*)''` requires both the solution itself and its first derivative
     * (because of the term `K(t,t) u*'(t)`), so both functions are passed to the operator.
     */
    fun rhsExactDeriv2(t: Double, op: VolterraOperator): Double {
        val integralDD = op.applyDeriv2(t, { s -> exact(s) }, { s -> exactDeriv(s) })
        return if (secondKind) exactDeriv2(t) - integralDD else integralDD
    }

    companion object {
        /**
         * Problem whose solution lies in the generating space: `K = e^{t-2s}`, `u* = t^2`.
         *
         * Since `u*` belongs to `span{1, t, t^2}`, which coincides with the polynomial
         * generating system `phi^B`, the method must reproduce the solution to machine
         * precision. That makes the problem a convenient indicator of implementation errors.
         *
         * The kernel derivatives are written out IN FULL (`K_t = K`, `K_s = -2K`, `K_tt = K`), as is
         * `u*'' = 2`. Formerly `kS`/`kTT`/`exactDeriv2` were left at their zero defaults while
         * the true values were nonzero, and the second derivative `(Vu)''` was computed incorrectly
         * for the family `xi^<0>` — silently, without any diagnostics.
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
         * Problem with a rational solution: `K = 1/(1+t+s)`, `u* = 1/(t+1)`.
         * The solution lies in none of the generating systems, so the problem serves
         * as the main tool for measuring the convergence order.
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
         * Problem with an exponential solution: `K = e^{-(t-s)^2}`, `u* = e^t`.
         * The solution matches the hyperbolic generating system `phi^H`.
         * The kernel diagonal `K(t,t) = 1` is nonzero.
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
         * Problem with a smoothing kernel: `K = t - s`, `u* = cos t`.
         *
         * The kernel diagonal vanishes (`K(t,t) = 0`), which strengthens the smoothing
         * action of the operator. On this problem the one-step Kulkarni scheme turns out
         * numerically markedly more accurate than the Sloan iteration — the observed orders are
         * about 3 (base scheme), 4 (Sloan), 5 (Kulkarni) and 6 (iterated
         * Kulkarni). This is a NUMERICAL OBSERVATION, not a proven result.
         *
         * Note: because of `K(t,t) = 0` the problem is NOT suitable for the solver of the
         * equation of the first kind — reduction by differentiation requires `K(t,t) != 0`.
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
         * Problem of the FIRST kind: `K = 1 + t - s`, `u* = cos t`.
         *
         * Unlike the Fredholm equation of the first kind (which is ill-posed), the Volterra
         * equation of the first kind with `K(t,t) != 0` is well-posed and reduces to an equation
         * of the second kind by a single differentiation (the case `m = 1`).
         * Here `K(t,t) = 1`, so the reduction applies.
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
 * Creates a solver of the equation of the second kind for a model problem.
 *
 * The convenience is that the right-hand side and its derivatives are taken from the problem
 * itself (computed through the exact solution), and the multiplier in front of the operator is one.
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
 * Creates a solver of the equation of the FIRST kind for a model problem (reduction to an
 * equation of the second kind by differentiation).
 *
 * The kernel and right-hand side of the reduced equation are built from the original problem:
 * the solver receives the kernel diagonal `K(t,t)`, the kernel derivative `K_t(t,s)`, as well as
 * the right-hand side `f'(t)` and the exact solution with its derivative — the latter are needed so
 * that only a small remainder is differentiated numerically rather than the whole right-hand side
 * (details in the KDoc of [VolterraFirstKindSolver]).
 *
 * @throws IllegalArgumentException if `K(t,t)` vanishes or a functional family requiring
 *         the second derivative has been chosen.
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
