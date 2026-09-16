package solvers.core

import solvers.core.SecondKindDefaults.DIVERGENCE_GROWTH_FACTOR

/**
 * THE SINGLE STOPPING CRITERION of the simple iterations for second-kind equations.
 *
 * All three iterative schemes of the project (`kulkarniQuasi` in [SecondKindSolverCore],
 * `combinedNystrom` for Fredholm and for Volterra) build a sequence of iterates
 * and measure the uniform norm of the difference of consecutive ones on their own set of check points.
 * They differ ONLY in the point set; the logic of "what to do with the resulting number"
 * is common to them, and it used to be written out three times. Duplication of the same sort as the one
 * already removed in [SecondKindDefaults]: a fix made to one copy is easily not carried over
 * to the other two, and schemes that are compared against each other in the characterization
 * tests silently drift apart.
 *
 * The class holds the state of a run and answers a single question: should the loop be interrupted.
 *
 * TWO exit conditions, the order of checks matters:
 *  1. convergence `diff < tolerance` — checked FIRST, so that at the boundary the behaviour
 *     matches the former (single-condition) criterion exactly;
 *  2. divergence — the criterion ceased to be finite OR grew relative to
 *     the smallest attained value by more than a factor of [DIVERGENCE_GROWTH_FACTOR].
 *
 * @param tolerance the required value of the stopping criterion.
 */
internal class IterationStopCriterion(private val tolerance: Double) {

    /** Convergence reached: the stopping criterion dropped below `tolerance`. */
    var converged = false
        private set

    /**
     * Divergence detected: the computation was stopped EARLY, the iteration limit is not exhausted.
     * Telling this case apart from "iterations ran out" is needed in the diagnostics: otherwise
     * the message would claim that all admissible steps had been tried, which is false.
     */
    var diverged = false
        private set

    /** Number of iterations actually performed. */
    var performedIterations = 0
        private set

    /**
     * The last FINITE value of the stopping criterion — the one that goes into
     * [solvers.core.SolutionFunc.residual].
     *
     * Finite precisely, not merely the last one: on divergence with overflow the
     * last value turns out to be `Inf` or `NaN`, and such a "residual" says nothing
     * about how bad the result is.
     *
     * The initial value `NaN` is replaced on the VERY FIRST call of [accept] with a finite
     * `diff`. Formally it could survive if `diff` turned out to be non-numeric already on the
     * first iteration — and then the contract `residual > 0 && !isNaN` checked by
     * `DivergenceContractTest` would be violated. That case is UNREACHABLE: the first iterate
     * is `f` with finite values, the second is its image under a finite-rank operator
     * with a bounded kernel, so the difference at the first step is finite for any
     * `cL`. Overflow is the result of ACCUMULATED geometric growth and takes
     * dozens of steps, by which time the field is already filled in (and by which time the computation is usually
     * already stopped by the growth detection).
     */
    var residual = Double.NaN
        private set

    /**
     * The smallest attained value of the criterion — the baseline for detecting growth.
     * The comparison is made against the minimum rather than against the previous step: under geometric
     * divergence the ratio of consecutive values equals the ratio of the progression and is
     * bounded, so a step-wise comparison would never fire, whereas the
     * accumulated growth relative to the minimum grows without bound (the detailed
     * rationale is in the KDoc of [DIVERGENCE_GROWTH_FACTOR]).
     */
    private var minDiff = Double.POSITIVE_INFINITY

    /**
     * Accept the next value of the stopping criterion and decide whether to interrupt the loop.
     *
     * @param diff uniform norm of the difference of consecutive iterates at the check points.
     * @return `true` if the loop should be interrupted (either by convergence or by divergence).
     */
    fun accept(diff: Double): Boolean {
        performedIterations++
        if (diff.isFinite()) residual = diff
        // Convergence is checked FIRST (see the class KDoc). The comparison `NaN < tolerance`
        // yields `false`, so a non-number cannot slip through here.
        if (diff < tolerance) {
            converged = true
            return true
        }
        if (!diff.isFinite() || (minDiff.isFinite() && diff > DIVERGENCE_GROWTH_FACTOR * minDiff)) {
            diverged = true
            return true
        }
        // At the first step `minDiff` equals `+Inf` (not finite), so divergence
        // cannot be declared before a baseline for comparison has been accumulated.
        if (diff < minDiff) minDiff = diff
        return false
    }
}
