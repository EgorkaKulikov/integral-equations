package solvers.volterra

import kotlin.math.abs
import numerics.*
import splines.*
import solvers.core.SolutionFunc
import splines.functionals.*
import splines.metrics.*
import solvers.core.RhsWithDerivatives

/**
 * Solver for the FIRST-kind Volterra equation `(V u)(t) = \int_a^t K(t,s) u(s) ds = f(t)`
 * by reduction to a second-kind equation through differentiation.
 *
 * The mathematical idea. Unlike the first-kind Fredholm equation (ill-posed and
 * requiring regularization), the Volterra problem is WELL-POSED when `K(t,t) != 0`. Differentiating
 * the original equation with respect to `t` by the Leibniz rule gives
 *
 *     K(t,t) u(t) + \int_a^t K_t(t,s) u(s) ds = f'(t),
 *
 * and after dividing by `K(t,t)` we arrive at the second-kind equation
 *
 *     u(t) - (W u)(t) = g(t),   (W u)(t) = \int_a^t [-K_t(t,s)/K(t,t)] u(s) ds,
 *     g(t) = f'(t)/K(t,t),
 *
 * which is solved by the usual second-kind scheme with `c_L = 1`. The source of the method is given
 * in `docs/REFERENCES.md` (section "First-kind equations").
 *
 * The case `m = 1` (a single differentiation) applies only when `K(t,t) != 0`.
 *
 * DIAGNOSTICS OF A DEGENERATE DIAGONAL — two independent lines of defence:
 *
 *  1. *A preliminary check at construction time* — over all points where the division is actually
 *     performed on a regular basis: the grid breakpoints, the interval midpoints, the Gauss nodes
 *     of the composite quadrature of the reduced operator, and the finite-difference stencil points
 *     around each of them. It gives early and cheap diagnostics before the computation starts.
 *  2. *A guard at the division site itself* ([safeDiagonal]) — it fires on ANY evaluation,
 *     including one at an arbitrary point `t` requested by the user from a ready
 *     solution. It is what provides the guarantee: a silent `NaN`/`Inf` cannot arise anywhere.
 *
 * The second line of defence is necessary because the set of division points is neither finite nor
 * precomputable: the Volterra operator integrates over `[a,t]` with a TRUNCATED last
 * cell, so the Gauss nodes depend on `t`, while the Sloan iteration evaluates `g(t)`
 * at any requested point. Previously such a situation returned `NaN` without any
 * signal: for instance, with a diagonal positive at the breakpoints and midpoints but zero
 * at an intermediate point, `base()` produced a plausible `E_h ~ 1.3e-5`, while `sloan()`
 * silently returned `NaN`.
 *
 * @param basis minimal spline basis.
 * @param funcs family of approximation functionals.
 * @param kernel kernel of the ORIGINAL first-kind equation — the kernel itself, not a ready
 *        [VolterraOperator]. The reduction divides by the diagonal `K(t,t)` (an operation on the
 *        KERNEL) and assembles its OWN operator [reducedOperator] with its own
 *        quadrature of order `REDUCED_OPERATOR_QUADRATURE_ORDER`; the quadrature of a passed
 *        operator would be ignored while its kernel would be extracted anyway,
 *        i.e. the parameter would end up half dead.
 * @param rhsDeriv derivative of the right-hand side `f'(t)` of the original equation.
 * @param smoothPart smooth part of the solution, known analytically; it is taken out from under the
 *        finite difference so as not to amplify noise (see the note on [gEffDeriv]).
 * @param smoothPartDeriv derivative of the smooth part.
 * @param throwOnDivergence policy for handling a failure to converge by the iterative
 *        schemes of the inner solver; see [VolterraSecondKindSolver.throwOnDivergence].
 * @throws IllegalArgumentException if `K(t,t)` is close to zero at the check points;
 *         if the interval is too short for the finite-difference stencil; or if
 *         a functional family requiring the second derivative was chosen.
 * @throws IllegalStateException if `K(t,t)` vanishes at a division point
 *         discovered during the computation (see [safeDiagonal]).
 */
public class VolterraFirstKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: FunctionalFamily,
    kernel: KernelV,
    rhsDeriv: (Double) -> Double,
    smoothPart: (Double) -> Double,
    smoothPartDeriv: (Double) -> Double,
    public val throwOnDivergence: Boolean = true,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    private companion object {
        /**
         * Threshold below which the kernel diagonal `|K(t,t)|` is considered zero and the reduction
         * of the first kind to the second is declared inapplicable. The value is chosen much
         * larger than the machine epsilon but much smaller than typical kernel values: dividing
         * by a smaller quantity gives an uncontrolled amplification of the error.
         */
        const val KERNEL_DIAGONAL_TOLERANCE = 1e-12

        /**
         * Finite-difference step for numerical differentiation.
         *
         * For a FOURTH-order formula the optimum of the sum of the approximation error `O(h^4)`
         * and the round-off error `O(eps/h)` is attained at `h ~ eps^{1/5} ~ 1e-3`.
         * With a smaller step (say `1e-6`, optimal for second order) the round-off error
         * and the quadrature noise start to dominate.
         *
         * THE STEP IS LEFT ABSOLUTE DELIBERATELY. A relative step (a fraction of `b - a`) would lift
         * the restriction on the interval length, but it would at the same time change the numerical results
         * on ALL existing problems, V1 included: the step size enters the approximation
         * error `O(h^4)` and the round-off error `O(eps/h)`, so changing the step
         * shifts `E_h` in the last significant digits. Such a replacement is a substantive
         * change of the algorithm requiring a deliberate re-shooting of the baseline, and it is not part
         * of the task "add the missing diagnostics". Instead, intervals on which
         * the stencil does not fit are EXPLICITLY FORBIDDEN (see [MIN_INTERVAL_STENCIL_STEPS]).
         */
        const val FINITE_DIFFERENCE_STEP = 1e-3

        /**
         * Minimal interval length `b - a`, expressed in steps of [FINITE_DIFFERENCE_STEP].
         *
         * A worst-case estimate over the branches of [deriv4]. The one-sided branch is chosen for
         * points closer than `2h` to an end, and its stencil extends `4h` in
         * the opposite direction: the total span reaches `2h + 4h = 6h`. Hence
         * for `b - a >= 6h` the stencil is guaranteed to stay inside `[a,b]` for any `t`,
         * whereas for a shorter interval it leaves the domain, where the kernel and the operator are
         * extended by zero, which would silently distort the derivative.
         */
        const val MIN_INTERVAL_STENCIL_STEPS = 6

        /**
         * Relative tolerance when comparing a point with the interval ends: it protects the choice of the
         * finite-difference branch from round-off in the coordinates.
         */
        const val BOUNDARY_RELATIVE_TOLERANCE = 1e-9

        /** Quadrature order for the reduced operator. */
        const val REDUCED_OPERATOR_QUADRATURE_ORDER = 8
    }

    private val grid = basis.grid
    private val quad = GaussLegendre(REDUCED_OPERATOR_QUADRATURE_ORDER)

    /** Kernel of the original first-kind equation (needed in methods too, not only in the initializers). */
    private val sourceKernel = kernel

    /**
     * The kernel diagonal `K(t,t)` WITH A CHECK — the denominator of the first-to-second-kind reduction.
     *
     * ALL divisions by the diagonal go through this function, so a degeneracy cannot
     * pass unnoticed at any point — including those that cannot be enumerated
     * in advance (the Gauss nodes of the truncated cell `[x_k, t]` and arbitrary points `t`
     * requested from a ready solution).
     *
     * @throws IllegalStateException if `|K(t,t)|` is below [KERNEL_DIAGONAL_TOLERANCE].
     */
    private fun safeDiagonal(t: Double): Double {
        val diagonal = sourceKernel.k(t, t)
        check(abs(diagonal) >= KERNEL_DIAGONAL_TOLERANCE) {
            "the first-kind Volterra solver requires K(t,t) != 0 (the case m=1): " +
                "at the division point t=$t we got K(t,t)=$diagonal " +
                "(|K(t,t)|=${abs(diagonal)} < the threshold $KERNEL_DIAGONAL_TOLERANCE). " +
                "This point coincides neither with a grid breakpoint nor with an interval midpoint, " +
                "hence the preliminary check did not cover it."
        }
        return diagonal
    }

    /** The kernel diagonal `K(t,t)` — the denominator of the first-to-second-kind reduction. */
    private val kernelDiagonal = { t: Double -> safeDiagonal(t) }

    init {
        // The interval must accommodate the finite-difference stencil: the step is absolute, hence
        // on a short interval the points t ± k*h would leave [a,b], where the kernel and the operator
        // are extended by zero — the derivative would be distorted silently.
        val intervalLength = grid.b - grid.a
        val requiredLength = MIN_INTERVAL_STENCIL_STEPS * FINITE_DIFFERENCE_STEP
        require(intervalLength >= requiredLength) {
            "the first-kind Volterra solver is inapplicable on a too short interval: " +
                "b - a = $intervalLength, while the fourth-order finite-difference stencil requires " +
                "at least $MIN_INTERVAL_STENCIL_STEPS steps of $FINITE_DIFFERENCE_STEP, that is " +
                "b - a >= $requiredLength. The difference step is absolute and does not scale with the interval " +
                "length, otherwise the stencil points would leave the domain of definition."
        }
        // The reduction divides by K(t,t), so a vanishing diagonal is inadmissible.
        // We check ALL points where the division is performed on a regular basis. Previously
        // only the breakpoints and the midpoints were checked, although the main consumer of the division is
        // the quadrature of the reduced operator and the finite-difference stencil.
        for (t in diagonalCheckPoints()) {
            val diagonal = kernel.k(t, t)
            require(abs(diagonal) >= KERNEL_DIAGONAL_TOLERANCE) {
                "the first-kind Volterra solver requires K(t,t) != 0 (the case m=1); " +
                    "K(t,t)=$diagonal at t=$t (|K(t,t)|=${abs(diagonal)}) is too small"
            }
        }
        // The family xi^<0> requires the SECOND derivative of the image (Wu)'' and of the right-hand side g''.
        // After the reduction the kernel K_W is itself defined through numerical differentiation, and its
        // derivatives K_W_s and K_W_tt are analytically unavailable: obtaining them would require
        // a threefold numerical differentiation with uncontrolled noise. Previously such a
        // call SILENTLY returned a wrong result (both derivatives were taken to be zero) —
        // now it is an explicit error instead of a silent distortion.
        require(!funcs.usesSecondDerivative) {
            "the first-kind Volterra solver does not support the family '${funcs.name}': " +
                "after the I->II kind reduction the second derivative of the kernel is analytically unavailable. " +
                "Use theta, xi^<1>, xi^<2>, mu or lambda."
        }
    }

    /**
     * Points at which the reduction is guaranteed to divide by `K(t,t)` in any scenario.
     *
     * Three groups are collected:
     *
     *  1. the grid breakpoints and the interval midpoints — the support points of the `theta` functionals;
     *  2. the Gauss nodes of the composite quadrature of the reduced operator over the full grid
     *     cells — this is where `gEff` and the reduced kernel are evaluated most often;
     *  3. the whole finite-difference stencil `t ± k*h`, `k = 1..4`, around every point
     *     of groups 1 and 2 — these points coincide neither with breakpoints nor with midpoints.
     *
     * The set is NOT exhaustive and cannot be: the Volterra operator integrates over
     * `[a,t]` with a truncated last cell, so its Gauss nodes depend on `t`.
     * The ultimate guarantee is given by [safeDiagonal], while this list provides early
     * diagnostics before the computation even starts.
     */
    private fun diagonalCheckPoints(): DoubleArray {
        val breakpoints = grid.breakpoints
        val (referenceNodes, _) = quad.refNodesWeights()
        val base = ArrayList<Double>()
        for (i in breakpoints.indices) {
            base.add(breakpoints[i])
            if (i < breakpoints.size - 1) {
                val lo = breakpoints[i]
                val hi = breakpoints[i + 1]
                base.add(0.5 * (lo + hi))
                // Gauss nodes of the cell [x_i, x_{i+1}] of the composite quadrature.
                val half = 0.5 * (hi - lo)
                val mid = 0.5 * (hi + lo)
                for (node in referenceNodes) base.add(mid + half * node)
            }
        }
        // The finite-difference stencil around each base point; points outside [a,b]
        // are discarded — there the one-sided branch of deriv4 takes over.
        val all = ArrayList<Double>(base)
        for (t in base) {
            for (k in 1..4) {
                val left = t - k * FINITE_DIFFERENCE_STEP
                val right = t + k * FINITE_DIFFERENCE_STEP
                if (left >= grid.a) all.add(left)
                if (right <= grid.b) all.add(right)
            }
        }
        return all.toDoubleArray()
    }

    /**
     * First derivative by a fourth-order formula.
     *
     * Inside the interval the five-point central difference
     * `(-f(t+2h) + 8f(t+h) - 8f(t-h) + f(t-2h)) / (12h)` is used. Near the ends its stencil would leave
     * `[a,b]`, where the kernel and the operator are extended by zero, which would distort the
     * result; one-sided five-point formulas of the same fourth order are therefore used there —
     * "forward" at the left end and "backward" at the right one.
     */
    private fun deriv4(t: Double, f: (Double) -> Double): Double {
        val leftEnd = grid.a
        val rightEnd = grid.b
        val step = FINITE_DIFFERENCE_STEP
        val boundaryTolerance = BOUNDARY_RELATIVE_TOLERANCE * (rightEnd - leftEnd)
        return when {
            // Left end: the central stencil (t - 2h) would leave a.
            t - 2 * step < leftEnd - boundaryTolerance ->
                (-25 * f(t) + 48 * f(t + step) - 36 * f(t + 2 * step) +
                    16 * f(t + 3 * step) - 3 * f(t + 4 * step)) / (12 * step)
            // Right end: the central stencil (t + 2h) would leave b.
            t + 2 * step > rightEnd + boundaryTolerance ->
                (25 * f(t) - 48 * f(t - step) + 36 * f(t - 2 * step) -
                    16 * f(t - 3 * step) + 3 * f(t - 4 * step)) / (12 * step)
            // Interior region: the central five-point difference.
            else ->
                (-f(t + 2 * step) + 8 * f(t + step) - 8 * f(t - step) + f(t - 2 * step)) / (12 * step)
        }
    }

    /**
     * Reduced kernel `K_W(t,s) = -K_t(t,s)/K(t,t)`.
     * Its derivative with respect to `t` is analytically unavailable and is computed by a finite difference.
     */
    private val reducedKernel = KernelV(
        k = { t, s -> -kernel.kT(t, s) / kernelDiagonal(t) },
        kT = { t, s -> deriv4(t) { argument -> -kernel.kT(argument, s) / kernelDiagonal(argument) } },
    )

    private val reducedOperator = VolterraOperator(reducedKernel, grid, quad)

    /** Right-hand side of the reduced equation: `g(t) = f'(t)/K(t,t)`. */
    private val gEff = { t: Double -> rhsDeriv(t) / kernelDiagonal(t) }

    /**
     * Derivative of the right-hand side `g'(t)`, needed by the functional families that use a derivative.
     *
     * Differentiating the whole `g` numerically would mean a SECOND differentiation
     * on top of `f'(t)`, which is itself obtained analytically by Leibniz and contains a quadrature:
     * the quadrature noise and the round-off error would then be amplified sharply.
     *
     * The decomposition `g(t) = s(t) + r(t)` is used instead, where `s` is the known smooth
     * part of the solution and `r = g - s` is a small remainder carrying the quadrature contribution. Then
     * `g'(t) = s'(t) + r'(t)`: the smooth part is differentiated ANALYTICALLY, and the finite
     * difference is applied only to the remainder. This removes the large smooth term from under the
     * subtraction, and the catastrophic loss of accuracy affects only the small quantity `|r|`.
     */
    private val gEffResidual = { t: Double -> gEff(t) - smoothPart(t) }
    private val gEffDeriv = { t: Double -> smoothPartDeriv(t) + deriv4(t, gEffResidual) }

    private val inner = VolterraSecondKindSolver(
        basis, funcs, reducedOperator, cL = 1.0,
        // The second derivative is not supplied: the previous default `{ 0.0 }` is kept.
        rhs = RhsWithDerivatives(value = gEff, deriv = gEffDeriv),
        throwOnDivergence = throwOnDivergence,
        ctx = ctx,
    )

    /** Base collocation scheme for the reduced equation. */
    public fun base(): SolutionFunc = inner.base()

    /** Sloan iteration applied to the reduced equation. */
    public fun sloan(): SolutionFunc = inner.sloan()

    /** Kulkarni scheme for the reduced equation. */
    public fun kulkarni(): SolutionFunc = inner.kulkarni()

    /** Iterated Kulkarni scheme for the reduced equation. */
    public fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}
