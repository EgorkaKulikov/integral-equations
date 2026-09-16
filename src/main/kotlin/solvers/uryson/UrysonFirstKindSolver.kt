package solvers.uryson

import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext
import splines.functionals.ProjFunctionals
import solvers.core.reportConvergence
import solvers.core.FirstKindSolution

/**
 * Regularized spline collocation for the nonlinear Uryson equation of the FIRST kind
 * `\int_a^b K(t,s,x(s)) ds = f(t)`.
 *
 * The problem is ill-posed, so the Tikhonov functional
 * `||Theta_h(U x_h) - Theta_h(f^delta)||^2 + alpha c^T R_h c` is minimized, the stabilizer
 * `R_h` being given by the norm of the space `W^{1,2}`. The minimization is carried out by
 * Gauss–Newton iterations, and the parameter `alpha` is chosen by Morozov's discrepancy principle.
 * The sources are in `docs/REFERENCES.md`.
 *
 * The solver knows nothing about model problems: the noisy data is supplied as a ready
 * vector `theta_j(f^delta)`, which is built by the `problems.uryson` package.
 *
 * @param tau safety factor in Morozov's principle; the theory only requires `tau > 1`.
 * @param gnTol stopping criterion of Gauss–Newton on the step norm.
 * @param gnMaxIter limit on the number of Gauss–Newton iterations at a fixed `alpha`.
 * @param throwOnDivergence behaviour when Gauss–Newton fails to converge:
 *        `true` (the default — as in all the other solvers) — an exception,
 *        `false` — the last approximation is returned. The policy applies to the PUBLIC
 *        call [solveFixedAlpha]. The homotopy [solveMorozov] opts out of it
 *        EXPLICITLY by passing `false` at its own call: it walks a path of decreasing
 *        `alpha` in dozens of steps with a warm start, and failing the step criterion
 *        at an INDIVIDUAL `alpha` is a routine part of that path (the problem is ill-posed,
 *        intermediate `alpha` are bound to be ill-conditioned), not an error: the final
 *        `alpha` is then chosen by Morozov's discrepancy principle among those visited.
 *        A warning is written to the log in any case.
 */
public class UrysonFirstKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val space: SplineSpace,
    public val op: UrysohnOperator,
    public val tau: Double = DEFAULT_TAU,
    public val gnTol: Double = DEFAULT_GN_TOLERANCE,
    public val gnMaxIter: Int = DEFAULT_GN_MAX_ITERATIONS,
    public val throwOnDivergence: Boolean = true,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // CRITICAL precisely here: [solveMorozov] computes the stabilizer `Omega` through
        // `space.omegaReg` (i.e. through `space.ctx.backend`), and the Gauss–Newton system
        // through its own `ctx.backend`. Should they differ, two parts of ONE Morozov
        // criterion would be computed by different LU implementations — silently.
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonFirstKindSolver", ctx, "space", space.ctx)
    }

    public companion object {
        /**
         * Safety factor in Morozov's discrepancy principle.
         *
         * The theory only requires `tau > 1`; the particular value is an implementation choice:
         * the closer it is to one, the less smoothing, but the higher the sensitivity
         * to an inaccurate noise level estimate.
         */
        public const val DEFAULT_TAU: Double = 1.1

        /** Stopping criterion of Gauss–Newton on the uniform step norm. */
        public const val DEFAULT_GN_TOLERANCE: Double = 1e-10

        /**
         * Limit on Gauss–Newton iterations at a fixed `alpha`. The method is applied
         * inside a homotopy in the regularization parameter, where each next run
         * starts from the previous solution, so a large number of iterations is not needed.
         */
        public const val DEFAULT_GN_MAX_ITERATIONS: Int = 50

        /** Upper bound of the exponent in the logarithmic grid of the parameter `alpha`. */
        private const val ALPHA_MAX_EXPONENT = 2.0

        /** Lower bound of the exponent in the logarithmic grid of the parameter `alpha`. */
        private const val ALPHA_MIN_EXPONENT = -12.0

        /**
         * Number of homotopy steps in `alpha`. Together with the exponent bounds it fixes the
         * grid step `10^{-0.25}`: fine enough for the Morozov point to be determined
         * robustly, and coarse enough for the whole path to be computed in reasonable time.
         */
        private const val ALPHA_PATH_STEPS = 56

        /**
         * Threshold below which the warm start is deemed degenerate and replaced by the
         * projection of a constant function. Needed for kernels with `dK/du(t,s,0) = 0`, where
         * the Jacobian vanishes at a zero initial guess.
         */
        private const val DEGENERATE_START_THRESHOLD = 1e-8
    }

    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    private val core = CollocationCore(basis, funcs, op, ctx)
    private val weights = space.weights
    private val gramR = space.gramR

    /** Vector of functional values `theta_j(f)` for an arbitrary right-hand side. */
    public fun thetaOf(f: (Double) -> Double): DoubleArray =
        DoubleArray(n + 2) { funcs.valueFunctional(it - 2).applyTo(f) }

    /**
     * Solves the regularized problem at a FIXED `alpha` by the Gauss–Newton method.
     *
     * The step is defined by the system
     * `(B^T W_h B + alpha R_h) delta = -B^T W_h (Xi - theta(f^delta)) - alpha R_h c`.
     *
     * The policy for failure to converge is taken from [throwOnDivergence];
     * the homotopy [solveMorozov] calls the internal overload and opts out of it explicitly.
     *
     * @param thetaFDelta vector `theta_j(f^delta)` of the noisy data.
     * @param alpha regularization parameter, strictly positive.
     * @param c0 initial guess for the coefficients.
     */
    public fun solveFixedAlpha(thetaFDelta: DoubleArray, alpha: Double, c0: DoubleArray): DoubleArray =
        solveFixedAlpha(thetaFDelta, alpha, c0, throwOnDivergence)

    /**
     * Implementation of the step at a fixed `alpha` with an EXPLICIT divergence policy.
     *
     * A separate overload is needed exactly so that [solveMorozov] can walk the path in the
     * regularization parameter with `throwOnDivergence = false` without changing the policy
     * requested by the caller for the public [solveFixedAlpha].
     */
    private fun solveFixedAlpha(
        thetaFDelta: DoubleArray,
        alpha: Double,
        c0: DoubleArray,
        throwOnDivergence: Boolean,
    ): DoubleArray {
        require(alpha > 0.0) { "regularization parameter alpha must be positive, got alpha=$alpha" }
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
            methodName = "Gauss-Newton (Uryson, first kind, alpha=$alpha)",
            iterations = gnMaxIter,
            maxIterations = gnMaxIter,
            residual = lastStep,
            tolerance = gnTol,
            hint = "at an individual alpha this is expected inside the homotopy in the regularization " +
                "parameter; the final alpha is chosen by Morozov's discrepancy principle",
        )
        return c
    }

    /** Discrete residual `res_h = ||Theta_h(U x_h) - Theta_h(f^delta)||` in the weighted norm. */
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
     * Chooses `alpha` by Morozov's discrepancy principle: the largest value at which
     * `res_h(alpha) <= tau * C_theta * sqrt(b - a) * delta`.
     *
     * Implemented as a homotopy in DECREASING `alpha` with a warm start: the solution at
     * one value serves as the initial guess for the next. For an ill-posed
     * problem this markedly stabilizes Gauss–Newton and makes the residual monotone.
     *
     * If the target is unreachable along the whole path (on too coarse a grid, say),
     * the solution with the smallest residual attained is returned — without "rocking" the solution.
     *
     * IMPLICIT DEPENDENCE OF THE NOISE ESTIMATE ON THE TYPE OF [funcs]. The factor
     * `funcs.cChi()` in `barDelta` is the amplification factor of a perturbation of the input
     * data by the functionals `theta_j`. It is correct PRECISELY because the type of the
     * parameter [funcs] is restricted to [ProjFunctionals] — a family of VALUE
     * functionals (`usesDerivative` there is the constant `false`). For them
     * `cChi()` is indeed the norm of the (quasi-)projector on perturbations of values.
     *
     * Widening the type of [funcs] to the generic `FunctionalFamily` WILL REQUIRE revisiting
     * this estimate: for families with derivatives (xi) `cChi()` is not an amplification
     * estimate and behaves in `h` in a qualitatively opposite way (see the KDoc of
     * `splines.functionals.FunctionalFamily.cChi` and `numerics.functionals.DerivFunctional`).
     *
     * @param thetaFDelta vector `theta_j(f^delta)` of the noisy data.
     * @param delta noise level in the `L^2` norm; at `delta = 0` the whole path is walked.
     */
    public fun solveMorozov(thetaFDelta: DoubleArray, delta: Double): FirstKindSolution {
        // The correctness of cChi() as a noise amplification factor rests on funcs being
        // a ProjFunctionals (value functionals). See the KDoc of the method.
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
            // Divergence at an INDIVIDUAL alpha is a routine part of the path, not an error:
            // the solver policy is disabled here explicitly (see the KDoc of [throwOnDivergence]).
            c = solveFixedAlpha(thetaFDelta, alpha, start, throwOnDivergence = false)
            val res = residual(c, thetaFDelta)
            if (delta == 0.0) {
                // No noise: the Morozov criterion degenerates, we go down to the smallest alpha.
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
            ?: error("the path in the regularization parameter is empty: check ALPHA_PATH_STEPS.")
    }
}
