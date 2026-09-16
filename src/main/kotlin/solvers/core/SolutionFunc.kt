package solvers.core

import java.util.logging.Logger

// ============================================================================
// 12. SOLUTION RESULT AND THE SINGLE CONVERGENCE CONTRACT
// ============================================================================

/** Logger for the convergence diagnostics of the iterative schemes of all solvers. */
private val convergenceLogger: Logger = Logger.getLogger("numerics.Convergence")

/**
 * Result of solving an integral equation: an evaluator of the approximate solution
 * `u_h(t)` together with the information on HOW that result was obtained.
 *
 * The type is shared by all three solvers (Fredholm, Volterra, Uryson). Previously each
 * of the `solvers.*` packages had its own class with the same name, and they drifted apart:
 * the Uryson solver had a field `iterations`, the others did not. The very notion of
 * "a solution plus the information about how it was obtained" is operator-neutral (it contains nothing
 * specific to the kind of equation), so its place is in `numerics`, next to [Grid] and
 * [LinearAlgebra]. A single type lets the convergence contract be described ONCE instead of
 * three times, and rules out a repeated divergence of the implementations.
 *
 * The reason for stopping ("iterations ran out" versus "divergence detected") is deliberately NOT
 * stored in this type. What matters for the caller is the fact `converged == false`
 * and the magnitude of the residual; distinguishing the two reasons is needed exclusively for DIAGNOSTICS
 * and is fully expressed by the message text of [reportConvergence]. A fifth field would be
 * dead weight in the public API: it would have to be filled in by all the direct schemes,
 * where it is meaningless, and no consumer in the project reads it.
 *
 * @param eval evaluator of `u_h(t)` at an arbitrary point.
 * @param converged convergence flag. For DIRECT schemes (base
 *        collocation, the Sloan iteration, Kulkarni on a projector, the classical Nyström)
 *        the value `true` is trivial: they solve a linear system and perform no iterations.
 *        For the iterative schemes the value reflects the actual outcome.
 * @param iterations number of steps of the iterative process ACTUALLY PERFORMED,
 *        and NOT the number of stopping-criterion checks. The value `0` is possible in two
 *        cases: for DIRECT schemes (there are no iterations at all) and on IMMEDIATE convergence,
 *        when the initial guess already satisfies the criterion and the vector of unknowns
 *        does not change. Telling them apart by this field is IMPOSSIBLE and UNNECESSARY: in both
 *        cases the number of steps really is zero. Previously the Uryson schemes counted checks
 *        and reported `1` for zero performed steps.
 * @param residual the attained FINAL value of the stopping criterion. It is needed precisely
 *        in the `throwOnDivergence = false` mode: without it the caller gets a non-converged
 *        result and cannot judge how bad it is. For direct schemes it is `0.0`.
 *
 *        THE MEANING OF THE QUANTITY DEPENDS ON THE SCHEME, and IT HAS no single interpretation —
 *        that is the price of a single result type shared by three solvers:
 *         - The Newton schemes of Uryson — the norm of the RESIDUAL `F(x)`. When
 *           `converged == true` it is always measured AT THE RETURNED point. When the
 *           step limit is exhausted, it is the residual BEFORE THE LAST step, not at the returned
 *           point (deliberately, see `solvers.uryson.runNewtonIterations`).
 *         - The simple iterations of Fredholm and Volterra (`kulkarniQuasi`,
 *           `combinedNystrom`) — the norm of the DIFFERENCE OF CONSECUTIVE ITERATES, not a residual.
 *           When divergence is detected EARLY this may be a value from several
 *           steps back: the last FINITE value is kept, not the literally last one
 *           (see [solvers.core.IterationStopCriterion.residual]).
 *        Comparing `residual` ACROSS SCHEMES is therefore meaningless; it is meaningful
 *        only against the `tol` the scheme was run with.
 *
 *        FINITE precisely: on divergence with overflow the literally last
 *        value is `Inf` or `NaN`, and such a number carries no information about
 *        how bad the result is and is unfit for comparisons.
 */
public class SolutionFunc(
    public val eval: (Double) -> Double,
    public val converged: Boolean = true,
    public val iterations: Int = 0,
    public val residual: Double = 0.0,
)

/**
 * Single diagnostics of a failure to converge for ALL iterative schemes of the project.
 *
 * The default behaviour is an exception: silently returning a wrong result
 * is inadmissible, since a user of the library cannot tell a converged
 * result from a diverging one. A warning is ALWAYS written to the log (including when an
 * exception follows) — the log remains an additional channel rather than the
 * only signal, as it used to be in the Uryson solver.
 *
 * The mode `throwOnDivergence = false` is provided for research scenarios and
 * tests: it allows one DELIBERATELY to obtain a non-converged result, which in that
 * case is marked `converged = false` and carries the attained residual.
 *
 * @param converged the actual convergence flag.
 * @param throwOnDivergence throw an exception when `converged == false`.
 * @param methodName name of the scheme for the message (for example, `"kulkarniQuasi (Fredholm)"`).
 * @param iterations number of iterations performed.
 * @param maxIterations limit on the number of iterations.
 * @param residual the attained value of the stopping criterion.
 * @param tolerance the required value of the stopping criterion.
 * @param hint optional explanation of the cause (for example, a condition on the operator norm).
 * @param diverged the computation was stopped EARLY because divergence was detected, and not
 *        because the iteration limit was exhausted. A parameter with a default value: the schemes
 *        that do not detect divergence (the Newton iterations in the Uryson solver)
 *        call the function unchanged.
 * @throws IllegalStateException if convergence was not reached and `throwOnDivergence`.
 */
internal fun reportConvergence(
    converged: Boolean,
    throwOnDivergence: Boolean,
    methodName: String,
    iterations: Int,
    maxIterations: Int,
    residual: Double,
    tolerance: Double,
    hint: String? = null,
    diverged: Boolean = false,
) {
    if (converged) return
    // Two DIFFERENT events that used to be described by a single text.
    //  * Iterations ran out: the process might still converge, raising the limit makes sense.
    //  * Divergence detected: the computation was stopped EARLY, raising the limit is
    //    pointless — the problem or the scheme has to change.
    // In the second case the old text WOULD BE MISLEADING: it implied
    // that all admissible steps had been tried. The common part of the wording ("convergence not
    // reached", the iteration count, the attained and the required values) is kept in BOTH
    // variants: it is the substantive core of the diagnostics, not decoration.
    val message = buildString {
        if (diverged) {
            append("$methodName: convergence not reached — DIVERGENCE detected ")
            append("at step $iterations (the limit $maxIterations is not exhausted): the stopping criterion ")
            append("grows instead of decreasing; last finite value $residual, ")
            append("required $tolerance. Continuing is pointless: raising the iteration limit ")
            append("will not help")
        } else {
            append("$methodName: convergence not reached in $iterations iterations ")
            append("(limit $maxIterations): attained $residual, required $tolerance")
        }
        if (hint != null) append(". $hint")
    }
    convergenceLogger.warning(message)
    if (throwOnDivergence) {
        throw IllegalStateException(
            "$message. To obtain a non-converged result deliberately, " +
                "build the solver with throwOnDivergence = false and check the field " +
                "SolutionFunc.converged.",
        )
    }
}
