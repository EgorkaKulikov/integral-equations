package solvers.uryson

import numerics.LinearAlgebra

/**
 * Outcome of a Newton iteration run: how many steps were ACTUALLY performed, what the
 * residual is, and why the run stopped.
 *
 * @param converged convergence flag. Set ONLY when the residual norm at the returned
 *        point is actually below the tolerance — regardless of which criterion
 *        stopped the loop (see [runNewtonIterations]).
 * @param performedSteps number of Newton steps ACTUALLY performed. Zero means the
 *        initial guess already satisfied the residual criterion and the vector of
 *        unknowns was left untouched.
 * @param stalled the run stopped by STALLING: the step norm became negligible while the
 *        residual stayed above the tolerance. The step limit is NOT exhausted in this case,
 *        so raising it is pointless — the process is no longer moving. This case must be
 *        told apart in diagnostics: a "ran out of iterations" message would be
 *        misleading here. Possible only when `converged == false`.
 * @param residual residual norm. WHEN `converged == true` it is always taken at THE SAME
 *        point that is returned. WHEN `stalled == true` it is also taken at the returned
 *        point (the recomputation is already done — it is what showed the residual to be
 *        insufficient). WHEN the step limit is exhausted (`converged == false && !stalled`)
 *        it is the residual BEFORE the LAST step, NOT at the returned point; this is
 *        DELIBERATE, the rationale is in [runNewtonIterations].
 */
internal class NewtonRun(
    val converged: Boolean,
    val performedSteps: Int,
    val stalled: Boolean,
    val residual: Double,
)

/**
 * SINGLE NEWTON LOOP for all iterative schemes of the Uryson equation of the second kind.
 *
 * The three schemes of the solver ([UrysonSecondKindSolver.solveBase], `kulkarni`, `nystrom`)
 * build DIFFERENT systems `F(x) = 0` and solve the step with DIFFERENT Jacobians (the analytic
 * `I - cL B(c)` in the first two, a finite-difference one in the third), yet their control
 * logic used to be written out verbatim three times — and all three copies carried the same
 * two reporting defects:
 *
 *  1. the counter was incremented BEFORE the residual check, so on immediate convergence
 *     `iterations = 1` was reported with ZERO steps performed;
 *  2. on exit by the step criterion the vector of unknowns had already been updated while
 *     the residual norm was still the one computed at the PREVIOUS (worse) point — the
 *     quality report did not refer to the approximation being returned.
 *
 * Here the logic exists in a SINGLE copy; the schemes supply only `F` and the step.
 * That way the defect cannot reappear in a fourth scheme — the same device already
 * applied to simple iterations ([solvers.core.IterationStopCriterion]).
 *
 * TWO EXIT CONDITIONS, and both lead to ONE AND THE SAME success check.
 *  1. The residual dropped below the tolerance — success by definition.
 *  2. The step norm became negligible. This is a STALLING condition, NOT a statement about
 *     attained accuracy: it only says the iteration stopped moving. It is not obliged to
 *     coincide with success — a small step at a large residual arises routinely on an
 *     ill-conditioned Jacobian `I - cL B(c)` and on the APPROXIMATE finite-difference
 *     Jacobian of the `nystrom` scheme. That is why, once the residual is recomputed at
 *     the new point, it is COMPARED with the tolerance and `converged` is set by the
 *     actual result rather than unconditionally. The former code declared success in both
 *     cases: the `converged` flag could claim convergence at a residual ABOVE the
 *     requested one, i.e. it was unreliable exactly where it is needed.
 *
 * THE ORDER OF OPERATIONS IS PRESERVED BIT FOR BIT with respect to the former three loops:
 *  - the number of performed steps when the limit is exhausted is still `maxSteps`
 *    (the condition `performedSteps < maxSteps` is checked before the step, as before);
 *  - the residual check still precedes the step, so the exit point by the
 *    residual is the same;
 *  - the exit points by the step are the same: the comparison with the tolerance does NOT
 *    continue the loop, it only fixes the flag value, so the returned vector `x` is unchanged;
 *  - on exit by the step criterion the residual is recomputed by THE SAME [residualAt] as
 *    the main computation: no second residual formula appears.
 * The only computational difference is one EXTRA call to [residualAt] on exit by the step
 * criterion. It does not change the vector of unknowns, so the solution stays the same;
 * only the number reported about it changes.
 *
 * WHEN THE STEP LIMIT IS EXHAUSTED the residual is NOT re-measured, and this is not an oversight.
 * In that branch the process has not converged, so the last step may have taken the point
 * arbitrarily far away, and recomputation could turn a FINITE diagnostic value into
 * `Inf`/`NaN` — i.e. DEGRADE the divergence diagnostics exactly where it is needed most
 * (the same argument as for the field
 * [solvers.core.IterationStopCriterion.residual]: the FINITE value is what matters).
 * The defect fixed here concerned the CONVERGED result: there the number is read as an
 * estimate of the answer quality and must refer to that very answer.
 *
 * @param x vector of unknowns; MODIFIED IN PLACE, on return it holds the approximation
 *        that was found.
 * @param maxSteps limit on the number of Newton steps.
 * @param tolerance required value of the stopping criterion (shared by residual and step).
 * @param residualAt residual `F(x)` at the current point. Called once per iteration
 *        (plus once on exit by the step criterion).
 * @param stepAt Newton step `delta = -J^{-1} F(x)` from the current point and its residual.
 *        The residual is passed as an argument instead of being recomputed: in the Nyström
 *        scheme it costs `p` right-hand side evaluations.
 *
 *        CALL-ORDER GUARANTEE ONE MAY RELY ON: every call to `stepAt` is immediately
 *        preceded by a call to [residualAt] AT THE SAME POINT, with no calls in
 *        between. This is part of the contract, not an implementation detail:
 *        `nystrom` uses it to carry the intermediate `G(x)` from the residual
 *        computation into the Jacobian computation, so the right-hand side is not
 *        evaluated twice. Reordering or skipping a call breaks that hand-off, so the
 *        order must not be changed (a violation is caught by `error` on the `nystrom` side).
 */
internal fun runNewtonIterations(
    x: DoubleArray,
    maxSteps: Int,
    tolerance: Double,
    residualAt: (DoubleArray) -> DoubleArray,
    stepAt: (DoubleArray, DoubleArray) -> DoubleArray,
): NewtonRun {
    var performedSteps = 0
    var converged = false
    var stalled = false
    var lastResidual = Double.NaN
    while (performedSteps < maxSteps) {
        val f = residualAt(x)
        lastResidual = LinearAlgebra.normInf(f)
        // Comparison with `NaN` yields `false`, so a non-number is never taken for convergence.
        if (lastResidual < tolerance) {
            converged = true
            break
        }
        val delta = stepAt(x, f)
        for (i in x.indices) x[i] += delta[i]
        // The counter is incremented AFTER an actual step: `performedSteps` means the
        // number of steps performed, not the number of criterion checks.
        performedSteps++
        if (LinearAlgebra.normInf(delta) < tolerance) {
            // The point has already moved one line above, so the residual must be measured ANEW:
            // the former value referred to the point BEFORE the step and systematically
            // overstated the error of the returned answer. Measured discrepancy: base scheme,
            // problem B, `tol = 1e-10`, n=8 — `2.74e-10` was reported instead of the actual
            // `4.44e-15`, i.e. 6.2e4 times worse than reality.
            lastResidual = LinearAlgebra.normInf(residualAt(x))
            // A small step by itself is NOT success (see KDoc): the residual decides.
            converged = lastResidual < tolerance
            stalled = !converged
            break
        }
    }
    return NewtonRun(converged, performedSteps, stalled, lastResidual)
}
