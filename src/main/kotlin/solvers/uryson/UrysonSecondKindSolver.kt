package solvers.uryson

import kotlin.math.abs
import numerics.DenseMatrix
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext
import numerics.ParallelAssembly
import solvers.core.SolutionFunc
import splines.functionals.ProjFunctionals
import solvers.core.SupportPoints
import solvers.core.reportConvergence
import solvers.core.NewtonResult

/**
 * Solvers for the nonlinear Uryson equation of the SECOND kind
 * `x(t) - cL \int_a^b K(t,s,x(s)) ds = f(t)`.
 *
 * Implemented schemes: base collocation, Sloan iteration, Kulkarni's modification and its
 * iterated variant, the plain spline Nyström method, and the combined Nyström method
 * `P_theta L + (I - P_theta) L^N_h` together with its iterated variant.
 * The sources are listed in `docs/REFERENCES.md`.
 *
 * The solver knows nothing about model problems: the right-hand side is supplied as the
 * function [rhs] and the multiplier as the parameter [cL]. Ready-made factories for the model
 * problems live in the `problems.uryson` package.
 *
 * WHY THERE IS NO SHARED `solvers.core.SecondKindSolverCore` HERE. The core builds the
 * matrix `M_{j,i} = chi_j(L omega_i)`, which exists only for a LINEAR `L`: the image of a
 * basis function does not depend on the solution. In the Uryson case the operator is nonlinear,
 * and the analogue of `M` is the Jacobian [CollocationCore.bMatrix], which DEPENDS on the
 * current approximation `c`; the base scheme is a Newton loop, not a single `LinearAlgebra.solve`.
 *
 * @param basis basis of minimal splines.
 * @param funcs family of projection functionals `theta_j`. The type is narrowed to
 *        [ProjFunctionals] NOT for historical reasons: the schemes traverse `th.nodes`/`th.coeffs`
 *        directly via `valueFunctional`, which fails for any other family, while in
 *        [UrysonFirstKindSolver.solveMorozov] the correctness of `cChi()` as a noise
 *        amplification estimate depends on `usesDerivative == false`.
 * @param space spline space (the Nyström weights `W_j` are needed).
 * @param op Uryson operator.
 * @param cL multiplier in front of the integral operator.
 * @param rhs right-hand side `f(t)` — a raw function, NOT an `RhsWithDerivatives`.
 *        Derivatives of `f` are read only by the `chi_j` of the `xi` families, whereas here
 *        [funcs] is always a [ProjFunctionals] with `usesDerivative == false`: every access to
 *        the right-hand side takes the value only. The triple would add two fields that are
 *        never read but would look as if they influenced the result.
 * @param tol stopping criterion on the residual norm and the step norm.
 * @param maxIter limit on the number of Newton iterations in the BASE scheme.
 * @param kulkarniMaxIter limit on the number of quasi-Newton iterations in Kulkarni's scheme.
 * @param nystromMaxIter limit on the number of Newton iterations in the Nyström scheme.
 * @param throwOnDivergence behaviour when Newton's method fails to converge:
 *        `true` (the default) — an exception, `false` — a result with
 *        `converged = false`. Formerly all three schemes only wrote a warning
 *        to the log and returned the result: when the library is used programmatically
 *        such a warning went unnoticed.
 */
public class UrysonSecondKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val space: SplineSpace,
    public val op: UrysohnOperator,
    public val cL: Double,
    public val rhs: (Double) -> Double,
    public val tol: Double = DEFAULT_TOLERANCE,
    public val maxIter: Int = DEFAULT_MAX_ITERATIONS,
    public val kulkarniMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    public val nystromMaxIter: Int = DEFAULT_FIXED_POINT_MAX_ITERATIONS,
    public val throwOnDivergence: Boolean = true,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // The family and the spline space compute PARTS OF THE SAME problem — with the same backend.
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "funcs", funcs.ctx)
        NumericsContext.requireSame("UrysonSecondKindSolver", ctx, "space", space.ctx)
    }

    public companion object {
        /** Default stopping criterion: close to machine precision. */
        public const val DEFAULT_TOLERANCE: Double = 1e-12

        /**
         * Limit on Newton iterations in the base scheme. Taken with a wide margin: the method
         * converges quadratically within a handful of iterations, and the limit only guards
         * against looping forever on degenerate data.
         */
        public const val DEFAULT_MAX_ITERATIONS: Int = 10_000

        /**
         * Iteration limit for the Kulkarni and Nyström schemes.
         *
         * It is substantially smaller than [DEFAULT_MAX_ITERATIONS] because the price of one
         * iteration here is incomparably higher: Kulkarni's scheme evaluates nested integrals
         * at every step, and Nyström builds a finite-difference Jacobian requiring
         * `P` full right-hand side evaluations. Formerly this value was hard-wired in the code
         * as a constant, so the `maxIter` parameter had no effect on those schemes, contrary to
         * the documentation.
         */
        public const val DEFAULT_FIXED_POINT_MAX_ITERATIONS: Int = 60

        /** Lower bound of the stopping criterion for schemes using the analytic Jacobian. */
        private const val NEWTON_TOLERANCE_FLOOR = 1e-13

        /** Lower bound of the stopping criterion for the scheme with a finite-difference Jacobian. */
        private const val FINITE_DIFFERENCE_TOLERANCE_FLOOR = 1e-12

        /**
         * Relative step of the finite-difference Jacobian in the Nyström scheme.
         *
         * The value is close to the square root of machine epsilon (`sqrt(2.2e-16) ~ 1.5e-8`) —
         * the classical trade-off for a one-sided difference between the truncation error
         * (grows with the step) and the round-off error (grows as the step shrinks).
         */
        private const val JACOBIAN_RELATIVE_STEP = 1e-7
    }

    public val grid: Grid = basis.grid
    public val n: Int = grid.n

    /** Precomputed values `theta_j(f)` for the exact right-hand side. */
    private val thetaF: DoubleArray = DoubleArray(n + 2) { k -> funcs.valueFunctional(k - 2).applyTo(rhs) }

    private val collocation = CollocationCore(basis, funcs, op, ctx)

    /**
     * Base scheme: `c = theta(f) + cL Xi(c)`.
     *
     * Solved by Newton's method for `F(c) = c - theta(f) - cL Xi(c) = 0` with the
     * analytic Jacobian `J = I - cL B(c)`. Newton is chosen over simple iteration because it
     * converges even in the absence of contraction (for instance at `cL = 1`
     * with a cubic kernel).
     *
     * @return spline coefficients together with the Newton convergence information.
     * @throws IllegalStateException on failure to converge, if [throwOnDivergence].
     */
    public fun solveBase(): NewtonResult {
        val c = thetaF.copyOf()
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        val run = runNewtonIterations(
            x = c,
            maxSteps = maxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val xi = collocation.xiVector(current)
                DoubleArray(n + 2) { current[it] - thetaF[it] - cL * xi[it] }
            },
            stepAt = analyticNewtonStep,
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Newton (base Uryson scheme)",
            iterations = run.performedSteps,
            maxIterations = maxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        return NewtonResult(c, run.converged, run.performedSteps, run.residual)
    }

    /**
     * Diagnostic note used when the run was stopped by STALLING ([NewtonRun.stalled]).
     *
     * Without it the [reportConvergence] message would claim "not attained in N
     * iterations (limit M)", from which a reader would conclude that raising the limit
     * helps. Under stalling that is wrong: the limit is NOT exhausted, the iteration has
     * simply stopped moving.
     */
    private fun stallHint(run: NewtonRun): String? =
        if (!run.stalled) {
            null
        } else {
            "run stopped by STALLING at step ${run.performedSteps}: the step norm fell below the tolerance, " +
                "while the residual stayed above it. The iteration limit is NOT exhausted, so raising it " +
                "will not help: the iteration has stopped moving (likely causes — poor " +
                "conditioning of the Jacobian or its being only approximate). One has to change the grid, " +
                "the initial guess or the requested accuracy"
        }

    /**
     * Newton step with the ANALYTIC Jacobian `I - cL B(c)`, SHARED by the base scheme
     * and Kulkarni's scheme.
     *
     * The coincidence HERE IS NOT ACCIDENTAL and is therefore kept in one place: in Kulkarni's
     * scheme the Jacobian of the base scheme acts as the PRECONDITIONER of the quasi-Newton
     * iteration AS PRESCRIBED BY THE SOURCE (see the KDoc of [kulkarni]) — that is, the identity
     * of the step is part of the definition of the scheme, not a coincidence of implementations.
     * The difference between the two schemes lives ENTIRELY in `residualAt` (`F(c)` versus
     * `c - G_K(c)`), and the shared step does not blur it. The `nystrom` scheme is NOT included
     * here: it has a different, finite-difference Jacobian and a different space of unknowns.
     */
    private val analyticNewtonStep: (DoubleArray, DoubleArray) -> DoubleArray =
        { current, residual -> newtonStep(current, DoubleArray(n + 2) { -residual[it] }) }

    /** Newton step with the Jacobian `J = I - cL B(c)` (the rows are assembled independently). */
    private fun newtonStep(c: DoubleArray, negativeResidual: DoubleArray): DoubleArray {
        val b = collocation.bMatrix(c)
        val jacobian = ParallelAssembly.assembleDense(n + 2, n + 2, ctx.parallel) { r, col ->
            val value = -cL * b[r, col]
            if (r == col) value + 1.0 else value
        }
        return LinearAlgebra.solve(jacobian, negativeResidual, ctx.backend)
    }

    /**
     * Reconstruction of the right-hand side of Kulkarni's scheme from the coefficients `c` of the spline `y_h`.
     *
     * Returns a triple:
     *  - `yhNodes` — the values of `y_h` at the quadrature nodes [UrysohnOperator.gNode];
     *  - `gAtSupport` — the function `g(t) = f(t) + cL (U y_h)(t)` (evaluated at any
     *    point, in particular at the support points of the functionals);
     *  - `gCoeffs` — the coefficients of the projection `P_theta g`.
     *
     * The block is needed twice: on every quasi-Newton iteration (inside `G_K`) and after
     * leaving the loop — when recovering `x_h^K = y_h + (I - P_theta) g`.
     */
    private fun projectedRhs(c: DoubleArray): Triple<DoubleArray, (Double) -> Double, DoubleArray> {
        val yhNodes = DoubleArray(op.gNode.size) { basis.evalSpline(c, op.gNode[it]) }
        val gAtSupport = { t: Double -> rhs(t) + cL * op.applyNodes(t, yhNodes) }
        val gCoeffs = funcs.projectorCoeffs(gAtSupport)
        return Triple(yhNodes, gAtSupport, gCoeffs)
    }

    /** Base approximation `x_h` as a spline. */
    public fun base(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        return SolutionFunc(
            eval = { t -> basis.evalSpline(c, t) },
            converged = newton.converged,
            iterations = newton.iterations,
            residual = newton.residual,
        )
    }

    /** Sloan iteration: `\tilde x_h(t) = f(t) + cL (U x_h)(t)`. */
    public fun sloan(): SolutionFunc {
        val newton = solveBase()
        val c = newton.coeffs
        val splineSolution = { t: Double -> basis.evalSpline(c, t) }
        val eval = { t: Double -> rhs(t) + cL * op.apply(t) { s -> splineSolution(s) } }
        return SolutionFunc(
            eval = eval,
            converged = newton.converged,
            iterations = newton.iterations,
            residual = newton.residual,
        )
    }

    /**
     * Kulkarni's modification.
     *
     * The system for `y_h = P_theta x_h^K`: `c = theta(f) + cL Theta(U(arg(c)))`, where
     * `arg = y_h + (I - P_theta)[f + cL U(y_h)]`. It is solved by a quasi-Newton iteration whose
     * preconditioner is the Jacobian of the BASE scheme `I - cL B(c)`.
     * That choice is prescribed by the source and secures convergence in the absence of contraction,
     * where simple iteration diverges.
     *
     * The final approximation is recovered as `x_h^K = y_h + (I - P_theta)[f + cL U(y_h)]`.
     */
    public fun kulkarni(): SolutionFunc {
        val fNodes = DoubleArray(op.gNode.size) { rhs(op.gNode[it]) }

        /** Right-hand side of Kulkarni's system `G_K(c)`. */
        fun gK(c: DoubleArray): DoubleArray {
            val (yhNodes, _, gCoeffs) = projectedRhs(c)
            val uyhNodes = DoubleArray(op.gNode.size) { op.applyNodes(op.gNode[it], yhNodes) }
            val gNodes = DoubleArray(op.gNode.size) { fNodes[it] + cL * uyhNodes[it] }
            // Projector remainder at the quadrature nodes: arg = y_h + (I - P_theta) g.
            val argNodes = DoubleArray(op.gNode.size) {
                yhNodes[it] + gNodes[it] - basis.evalSpline(gCoeffs, op.gNode[it])
            }
            return DoubleArray(n + 2) { k ->
                val th = funcs.valueFunctional(k - 2)
                var acc = 0.0
                for (q in th.nodes.indices) acc += th.coeffs[q] * op.applyNodes(th.nodes[q], argNodes)
                thetaF[k] + cL * acc
            }
        }

        val c = thetaF.copyOf()
        val newtonTol = maxOf(tol, NEWTON_TOLERANCE_FLOOR)
        val run = runNewtonIterations(
            x = c,
            maxSteps = kulkarniMaxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val g = gK(current)
                DoubleArray(n + 2) { current[it] - g[it] }
            },
            stepAt = analyticNewtonStep,
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Quasi-Newton (Uryson Kulkarni scheme)",
            iterations = run.performedSteps,
            maxIterations = kulkarniMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        val (_, gAtSupport, gCoeffs) = projectedRhs(c)
        val eval = { t: Double -> basis.evalSpline(c, t) + (gAtSupport(t) - basis.evalSpline(gCoeffs, t)) }
        return SolutionFunc(
            eval = eval,
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    /**
     * Spline Nyström method: `x_h^N(t) = f(t) + cL sum_j theta_j(g_t) W_j`,
     * where `g_t(s) = K(t, s, x_h^N(s))`.
     *
     * The unknowns are the values of the solution at the support points of the functionals. The
     * system is solved by Newton's method with a FINITE-DIFFERENCE Jacobian: the formula contains
     * no nested integrals, so evaluating the right-hand side is cheap and the difference
     * Jacobian turns out to be cheaper than the analytic one.
     *
     * The initial guess is the projection of the constant function `P_theta(1)`. A zero start
     * is unusable: for kernels with `dK/du(t,s,0) = 0` (cubic ones, for instance) the Jacobian at
     * zero is singular and the method never moves.
     */
    public fun nystrom(): SolutionFunc {
        // The order of the points is the ORDER OF FIRST OCCURRENCE while traversing `j = -2..n-1`: it fixes
        // the numbering of the unknowns, the row order of the Jacobian and the initial guess vector.
        // Indexing is by the pair (functional number, node number), and NOT by looking the point value
        // up in a `HashMap<Double, Int>`, which required bitwise equality of Doubles.
        val vfs = Array(n + 2) { funcs.valueFunctional(it - 2) }
        val support = SupportPoints.byFirstOccurrence(vfs, grid.breakpointInclusionEps)
        val pts = support.points
        val wInt = space.wInt

        /** Right-hand side of the Nyström scheme from the solution values at the support points. */
        fun evalAtVals(t: Double, xVals: DoubleArray): Double {
            var acc = 0.0
            for (j in -2..n - 1) {
                val th = vfs[j + 2]
                var gtVal = 0.0
                for (q in th.nodes.indices) {
                    val supportPoint = th.nodes[q]
                    gtVal += th.coeffs[q] * op.kernel.k(t, supportPoint, xVals[support.indexOf(j + 2, q)])
                }
                acc += gtVal * wInt[j + 2]
            }
            return rhs(t) + cL * acc
        }

        val p = pts.size
        val constantProjection = funcs.projectorCoeffs({ 1.0 })
        val x = DoubleArray(p) { basis.evalSpline(constantProjection, pts[it]) }
        val newtonTol = maxOf(tol, FINITE_DIFFERENCE_TOLERANCE_FLOOR)
        // Right-hand side `G(x)` at the CURRENT point, handed over from the residual computation to the
        // step computation. It is DELIBERATELY not recovered as `x - F(x)`: such a backward subtraction
        // in IEEE 754 reproduces the original bits only while `x` and `G(x)` are of comparable magnitude,
        // and loses accuracy when the magnitudes differ widely — that would silently shift the numbers.
        // Recomputing `G` is undesirable too: it costs `p` right-hand side
        // evaluations. The hand-over relies on the CALL-ORDER GUARANTEE written explicitly in the KDoc
        // of the `stepAt` parameter of [runNewtonIterations]: `residualAt` is always called
        // immediately before `stepAt` at the same point. `null` instead of a dead
        // zero array is not a micro-optimization but a CHECK: should the guarantee ever be
        // violated, the `error` below fails loudly, whereas a zero array would silently produce
        // a wrong Jacobian and plausible numbers.
        var currentG: DoubleArray? = null
        val run = runNewtonIterations(
            x = x,
            maxSteps = nystromMaxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val gx = DoubleArray(p) { evalAtVals(pts[it], current) }
                currentG = gx
                DoubleArray(p) { current[it] - gx[it] }
            },
            stepAt = { current, residual ->
                val gx = currentG
                    ?: error(
                        "call-order guarantee of runNewtonIterations violated: stepAt was called without " +
                            "a preceding residualAt, so G(x) at the current point is unknown.",
                    )
                val jacobian = DenseMatrix.zeros(p, p)
                for (col in 0 until p) {
                    val saved = current[col]
                    // The step is scaled by the magnitude of the variable, to keep accuracy
                    // both for large values and for values close to zero.
                    val step = JACOBIAN_RELATIVE_STEP * (abs(saved) + 1.0)
                    current[col] = saved + step
                    val perturbed = DoubleArray(p) { evalAtVals(pts[it], current) }
                    current[col] = saved
                    // F(x) = x - G(x), hence dF[row]/dx[col] = [row == col] - dG[row]/dx[col].
                    for (row in 0 until p) {
                        val identity = if (row == col) 1.0 else 0.0
                        jacobian[row, col] = (identity * step - (perturbed[row] - gx[row])) / step
                    }
                }
                LinearAlgebra.solve(jacobian, DoubleArray(p) { -residual[it] }, ctx.backend)
            },
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Newton (Uryson Nyström scheme)",
            iterations = run.performedSteps,
            maxIterations = nystromMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
            hint = stallHint(run),
        )
        val xFinal = x
        return SolutionFunc(
            eval = { t -> evalAtVals(t, xFinal) },
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    /**
     * COMBINED Nyström method: `u = f + cL L_n u`,
     * `L_n = P_theta L + (I - P_theta) L^N_h` — the exact operator on the range of the projector,
     * the quadrature on its complement. It is to this operator (and not to the plain [nystrom])
     * that the superconvergence estimates for polynomial quasi-interpolants apply
     * (Remogna–Sbibih–Tahrichi, Mathematics 11 (2023), Art. 3236; see `docs/REFERENCES.md`).
     *
     * The system is solved by Newton's method with an analytic Jacobian in the solution values
     * at the support points of the functionals and at the quadrature nodes; details in [CombinedNystromSolver].
     */
    public fun combinedNystrom(): SolutionFunc = CombinedNystromSolver(this).combined()

    /**
     * Iterated combined Nyström: `\hat u^N_h = f + cL L u^N_h`, where `u^N_h` is the
     * solution of [combinedNystrom]; a single application of the exact operator, with no new system.
     */
    public fun iteratedCombinedNystrom(): SolutionFunc = CombinedNystromSolver(this).iterated()

    /**
     * Iterated Kulkarni method: `\hat u^K_h = f + cL L u^K_h`, where `u^K_h` is the solution of
     * [kulkarni]; the analogue of the Sloan iteration applied to Kulkarni's approximation.
     */
    public fun iteratedKulkarni(): SolutionFunc {
        val k = kulkarni()
        val uNodes = DoubleArray(op.gNode.size) { k.eval(op.gNode[it]) }
        return SolutionFunc(
            eval = { t -> rhs(t) + cL * op.applyNodes(t, uNodes) },
            converged = k.converged,
            iterations = k.iterations,
            residual = k.residual,
        )
    }
}
