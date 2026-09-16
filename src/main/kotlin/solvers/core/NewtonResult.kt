package solvers.core

/**
 * Result of the Newton iterations in the coefficient space.
 *
 * Previously [UrysonSecondKindSolver.solveBase] returned a `Pair<DoubleArray, Int>` from which
 * it was impossible to tell whether the iteration had converged: an iteration count equal to the limit
 * arises alike on convergence at the last step and on divergence.
 *
 * @param coeffs the spline coefficients found.
 * @param converged convergence flag.
 * @param iterations number of Newton steps ACTUALLY PERFORMED (see [NewtonRun]);
 *        `0` means that the coefficients did not change relative to the initial
 *        guess.
 * @param residual norm of the residual. When `converged == true` it is measured AT THE SAME
 *        point that is returned in [coeffs]. When the step limit is exhausted it is the residual
 *        BEFORE THE LAST step and NOT at the returned point: a deliberate limitation, the
 *        full rationale is in the KDoc of [runNewtonIterations] and [NewtonRun.residual].
 */
public class NewtonResult(
    public val coeffs: DoubleArray,
    public val converged: Boolean,
    public val iterations: Int,
    public val residual: Double,
)
