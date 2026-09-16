package solvers.core

/**
 * SHARED NUMERICAL PARAMETERS OF THE ITERATIVE SCHEMES FOR SECOND-KIND EQUATIONS.
 *
 * The single source of truth for the constants that used to be duplicated
 * in the `private companion object` of the Fredholm and Volterra solvers with IDENTICAL
 * values. Duplication is dangerous because, when a new value is tuned, it is easy
 * to fix it in one file and forget the other; then two schemes that are compared
 * against each other in the characterization tests silently drift apart.
 *
 * The values were NOT changed when they were extracted — the stage is behaviourally neutral.
 *
 * The visibility is `internal`: this is an implementation detail of the solvers, not part of the
 * library's public API. The iteration limits and the tolerances are deliberately NOT
 * parameterized by the user: they are tuned for specific schemes and carry no
 * meaning apart from them.
 */
internal object SecondKindDefaults {
    /**
     * Limit on the number of iterations in the Kulkarni scheme for QUASI-interpolants (`mu`, `lambda`).
     *
     * Implementation choice: there is no theoretical convergence guarantee (the quasi-interpolation
     * operator lacks the projector property `P^2 = P`), so a hard limit is mandatory.
     * On the model problems convergence occurs within a few iterations; the margin is needed
     * for problems whose operator norm is close to one.
     */
    const val KULKARNI_QUASI_MAX_ITERATIONS = 200

    /**
     * Stopping criterion of the Kulkarni iteration for quasi-interpolants: the uniform norm of the
     * difference of consecutive iterates at the check points. The value is close to machine
     * precision: refining further makes no sense because of the quadrature noise.
     */
    const val KULKARNI_QUASI_TOLERANCE = 1e-13

    /**
     * Limit on the number of iterations for the combined Nyström operator.
     *
     * Implementation choice: on the model problems convergence is reached within a few steps;
     * the margin is needed for problems whose operator norm is close to one.
     */
    const val COMBINED_NYSTROM_MAX_ITERATIONS = 200

    /**
     * Stopping criterion of the combined Nyström iteration: the uniform norm of the difference of
     * consecutive iterates at the check points (Gauss nodes for Fredholm,
     * an equidistant check grid for Volterra).
     */
    const val COMBINED_NYSTROM_TOLERANCE = 1e-13

    /**
     * Threshold for DIVERGENCE DETECTION: by what factor the stopping criterion must
     * exceed the smallest value attained during the run for further iterations
     * to be declared pointless.
     *
     * Why it is needed. The only exit condition used to be `diff < TOLERANCE`. On
     * a problem whose operator norm exceeds one, the simple iteration diverges
     * geometrically, and the scheme honestly ground on to the limit of 200 steps, inflating
     * the iterate to `1e120` along the way and risking overflow (`Inf`, then `NaN`) with
     * a slightly larger norm or a slightly larger limit. All those steps are a pure waste
     * of time: the fate of the iteration is decided long before the last step.
     *
     * Why the comparison is with the SMALLEST attained value rather than with the previous one.
     * Under geometric divergence with ratio `q` the ratio of consecutive values
     * equals `q` and stays bounded (on the model problem with `||L|| = 4` it is exactly 4),
     * so the step-wise comparison `diff > FACTOR * prevDiff` would NEVER fire —
     * the criterion would be dead code. Accumulated growth relative to the minimum,
     * on the contrary, behaves like `q^m` and is certain to break the threshold. At the same time the comparison
     * with the minimum is STRICTER than the step-wise one: `min <= prev`, so any firing of the
     * step-wise condition happens no earlier than the firing of the accumulated one.
     *
     * Why a false firing is excluded STRUCTURALLY. The minimum is updated AFTER
     * the convergence check (see [IterationStopCriterion.accept]), so any value
     * below the tolerance leaves the loop before it can enter the minimum. Consequently,
     * `minDiff >= 1e-13` always holds, and the false-firing threshold never drops below
     * `1e6 * 1e-13 = 1e-7`. For the threshold to fire on a CONVERGING iteration, its
     * criterion would first have to fall to `1e-13`...`1e-7` and then jump six
     * orders of magnitude above its own noise plateau — which a decreasing geometric
     * majorant with `||L|| < 1` does not allow.
     *
     * Why exactly `1e6` (the numbers are measured, not guessed). The margin before a false
     * firing was taken from instrumented runs with the threshold disabled:
     * the peak of the ratio `diff / min(diff)` over the whole run was measured.
     *  * On the regular problems of the project the peak is `1.5e-1`...`1.3e1` (a margin of
     *    at least `7.7e4` up to the threshold).
     *  * The worst spike on a CONVERGING run at all is `1.79e+02` (a margin of ≈ `5.6e3`).
     *    It is attained on a SPECIALLY crafted stress problem: kernel `K = 1`,
     *    Volterra operator (quasi-nilpotent — the Neumann series converges for ANY
     *    `cL`, i.e. the transient growth `cL^m/m!` is maximal), `cL = 11.5`, 182 iterations.
     *  * The first actual firing of the threshold happens at `cL = 16` (the Kulkarni scheme) and
     *    `cL = 18` (the combined Nyström), and both of these configurations fail to converge
     *    within 200 steps even WITHOUT the threshold and blow up to `1e10`...`1e16`: that is, the threshold
     *    interrupts only those runs that are doomed anyway.
     * The full measurements are in `.tasks/code-review-remediation/stage8/REVIEW-8.1.md`.
     *
     * At the same time `1e6` is a negligible fraction of the margin up to `Double` overflow (`~1e308`),
     * so the detection fires long before significant digits are lost.
     * The threshold is deliberately NOT parameterized: like the tolerances, it is tied to specific
     * schemes and has no meaning outside them.
     */
    const val DIVERGENCE_GROWTH_FACTOR = 1e6
}
