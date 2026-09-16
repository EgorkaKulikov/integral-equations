package problems.fredholm

import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmFirstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF

/**
 * Model problem for the linear Fredholm equation: kernel, exact solution and the kind of
 * the equation. The right-hand side is NOT given explicitly but built from the exact solution
 * numerically (by quadrature) — this keeps the test data consistent with the operator and
 * the quadrature rule instead of hard-wiring constants.
 *
 * Relations between the exact solution `u*` and the right-hand side `f`:
 *  - equation of the second kind: `f = u* - K u*`;
 *  - equation of the first kind:  `f = K u*`.
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
class FredholmProblem(
    val name: String,
    val kernel: KernelF,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val secondKind: Boolean,
    val exactDeriv2: (Double) -> Double = { 0.0 },
) {
    /**
     * Exact right-hand side `f(t)`, computed through the operator [op].
     *
     * @param t evaluation point.
     * @param op Fredholm operator built on the same kernel and the required grid.
     */
    fun rhsExact(t: Double, op: FredholmOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - integral else integral
    }

    /**
     * First derivative of the right-hand side `f'(t)`; needed by the functional families
     * that use the derivative. For an equation of the second kind it equals `u*' - d/dt (K u*)`.
     */
    fun rhsExactDeriv(t: Double, op: FredholmOperator): Double {
        val integralD = op.applyDeriv(t) { s -> exact(s) }
        return if (secondKind) exactDeriv(t) - integralD else integralD
    }

    /**
     * Second derivative of the right-hand side `f''(t)`; needed by the family `xi^<0>`.
     * For an equation of the second kind it equals `u*'' - d^2/dt^2 (K u*)`.
     */
    fun rhsExactDeriv2(t: Double, op: FredholmOperator): Double {
        val integralDD = op.applyDeriv2(t) { s -> exact(s) }
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
         * The kernel derivatives are written out in full: `K_t = K`, `K_s = -2K`, `K_tt = K`,
         * as well as `u*'' = 2`.
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
         * Problem with a rational solution: `K = 1/(1+t+s)`, `u* = 1/(t+1)`.
         * The solution lies in none of the generating systems, so the problem serves
         * as the main tool for measuring the convergence order.
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
         * Problem with an exponential solution: `K = e^{-(t-s)^2}`, `u* = e^t`.
         * The solution matches the hyperbolic generating system `phi^H`.
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
         * Ill-posed problem of the FIRST kind: `K = e^{-(t-s)^2}`, `u* = e^t`.
         * Solved by the regularization method (see [FredholmFirstKindSolver]).
         *
         * The kernel and solution derivatives are written out in full: `K_s = 2(t-s)K`,
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
 * Creates a solver of the equation of the second kind for a model problem.
 *
 * The convenience is that the right-hand side and its derivatives are taken from the problem
 * itself (computed through the exact solution), and the multiplier in front of the operator is one.
 */
fun secondKindSolver(
    problem: FredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
): FredholmSecondKindSolver = FredholmSecondKindSolver(
    basis, funcs, op, cL = 1.0,
    rhs = RhsWithDerivatives(
        value = { t -> problem.rhsExact(t, op) },
        deriv = { t -> problem.rhsExactDeriv(t, op) },
        deriv2 = { t -> problem.rhsExactDeriv2(t, op) },
    ),
)

/**
 * Creates a solver of the equation of the FIRST kind for a model problem.
 *
 * @param alpha regularization parameter (see [FredholmFirstKindSolver.DEFAULT_REGULARIZATION]).
 */
fun firstKindSolver(
    problem: FredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
    alpha: Double = FredholmFirstKindSolver.DEFAULT_REGULARIZATION,
): FredholmFirstKindSolver = FredholmFirstKindSolver(
    basis, funcs, op,
    rhs = { t -> problem.rhsExact(t, op) },
    rhsDeriv = { t -> problem.rhsExactDeriv(t, op) },
    rhsDeriv2 = { t -> problem.rhsExactDeriv2(t, op) },
    alpha = alpha,
)
