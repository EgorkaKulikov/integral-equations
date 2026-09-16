package solvers.volterra

import kotlin.math.abs
import numerics.*
import splines.*
import solvers.core.SolutionFunc
import solvers.core.SupportPoints
import solvers.core.reportConvergence
import splines.functionals.*
import splines.metrics.*
import solvers.core.ImageTriple
import solvers.core.IterationStopCriterion
import solvers.core.RhsWithDerivatives
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.COMBINED_NYSTROM_TOLERANCE
import solvers.core.SecondKindSolverCore

/**
 * Linear solver for the second-kind Volterra equation u - L u = f, L = c_L * \mathcal V,
 * where (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds (a VARIABLE upper limit).
 *
 * c_L = 1 for a second-kind equation; for the I->II kind reduction (see [VolterraFirstKindSolver])
 * c_L = 1 as well, but with a different (reduced) kernel and right-hand side.
 * The right-hand side f and its derivatives are supplied explicitly ([RhsWithDerivatives])
 * so that the solver is reused for first-kind problems too.
 *
 * Matrices of the discrete problem:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * @param throwOnDivergence behaviour of the ITERATIVE schemes ([kulkarni] for
 *        quasi-interpolants, [combinedNystrom]) when convergence is not reached:
 *        `true` (default) — an exception, `false` — a result with
 *        `converged = false` and the attained residual in [solvers.core.SolutionFunc.residual].
 *        Direct schemes are unaffected.
 */
public class VolterraSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    public val op: VolterraOperator,
    cL: Double,
    rhs: RhsWithDerivatives,
    throwOnDivergence: Boolean = true,
    ctx: NumericsContext = NumericsContext.default(),
) : SecondKindSolverCore<(Double) -> Double>(
    basis, funcs, cL, rhs, throwOnDivergence, ctx,
) {
    // The numerical parameters of the iterative schemes (KULKARNI_QUASI_*, COMBINED_NYSTROM_*)
    // live in [solvers.core.SecondKindDefaults]: they coincide for Fredholm and Volterra,
    // and a divergence while tuning a new value would be silent.
    // The shared part of the schemes (`base`, `sloan`, `kulkarni`, assembly of M/M2/g/d) is in
    // [SecondKindSolverCore]; what stays here is the Volterra specifics.
    //
    // NOTE (the difference from Fredholm): for the Volterra operator the integration domain
    // [a,t] depends on t, so precomputation on fixed nodes is impossible.
    // All applications of L = c_L \mathcal V are expressed through the closures op.apply / op.applyDeriv,
    // and the "prepared operand" is the function itself rather than a table of its values.

    /**
     * L g(t) = c_L (\mathcal V g)(t) and its derivative (Leibniz).
     *
     * The returned closure owns ITS OWN cache of the nodal values of `g` on the full cells
     * (see [VolterraOperator.IntegrandCache]): the lifetime of the cache coincides with the lifetime
     * of the closure, and the binding to `g` is fixed at creation time, so handing the values
     * of one function to another is impossible. The arithmetic does not change (see [VolterraOperator.apply]).
     *
     * The method DELIBERATELY did not move into the shared core: the cache is bound to a specific
     * [VolterraOperator] by the check `require(cache.owner === this)`, and a generic cache
     * type would weaken that check to a runtime cast.
     */
    private fun applyL(g: (Double) -> Double): (Double) -> Double {
        val cache = op.integrandCache(g)
        return { t -> cL * op.apply(t, cache) }
    }
    private fun applyLDeriv(g: (Double) -> Double): (Double) -> Double = { t -> cL * op.applyDeriv(t, g) }

    /**
     * (L g)''(t) = c_L (\mathcal V g)''(t) by (V2''): it requires g AND g' (the term K(t,t) g'(t)).
     * gD is the first derivative of the operand g itself.
     */
    private fun applyLDeriv2(g: (Double) -> Double, gD: (Double) -> Double): (Double) -> Double =
        { t -> cL * op.applyDeriv2(t, g, gD) }

    // --- Implementation of the [SecondKindSolverCore] extension points ---------

    override val equationName: String get() = "Volterra"

    override val kulkarniQuasiHint: String
        get() = "For quasi-interpolants (mu, lambda) the property P^2 = P does not hold, so the " +
            "Kulkarni reduction is inapplicable and a simple iteration is used"

    /**
     * Check points of the stopping criterion: `4n+1` equidistant points of the interval.
     *
     * The Volterra operator has no global nodes (they depend on `t`), so
     * a separate sample is needed. It takes part ONLY in the stopping criterion,
     * it is not used in the iteration itself.
     *
     * The field is lazy rather than eager for two reasons. First, the sample is needed
     * by only two of the eight schemes. Second and more importantly: it reads
     * `n` and `grid` from the base class through an overridden property, and such a property
     * can in principle be read before the subclass constructor has finished;
     * `by lazy` guarantees that the computation happens at the FIRST ACCESS from
     * a method, i.e. guaranteed after `n` has already been initialized.
     * An eager field here would be correct only by accident.
     */
    override val checkPoints: DoubleArray by lazy {
        DoubleArray(4 * n + 1) { grid.a + (grid.b - grid.a) * it / (4 * n) }
    }

    /** Precomputation is impossible: the operand is the function itself. */
    override fun prepare(u: (Double) -> Double): (Double) -> Double = u

    override fun image(o: (Double) -> Double): (Double) -> Double = applyL(o)

    override fun imageDeriv(o: (Double) -> Double): (Double) -> Double = applyLDeriv(o)

    /** The Leibniz term `K(t,t) u'(t)` makes `uD` a MANDATORY argument. */
    override fun imageDeriv2(o: (Double) -> Double, uD: (Double) -> Double): (Double) -> Double =
        applyLDeriv2(o, uD)

    override fun applyOperator(t: Double, u: (Double) -> Double): Double = op.apply(t, u)

    override fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double = op.applyDeriv(t, u)

    override fun applyOperatorDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double =
        op.applyDeriv2(t, u, uD)

    override fun omegaImages(i: Int): ImageTriple {
        val idx = i - 2
        val omega = { s: Double -> basis.omega(idx, s) }
        val omegaD = { s: Double -> basis.omegaDeriv(idx, s) }
        // The second derivative of the image requires both omega_i and omega_i' (the term K(t,t) omega_i').
        return ImageTriple(applyL(omega), applyLDeriv(omega), applyLDeriv2(omega, omegaD))
    }

    /**
     * The image `L omega_i` is built ANEW, without reusing the column from [matrixM].
     *
     * This is not redundancy but a deliberate decision: every closure [applyL] carries
     * ITS OWN integrand cache, and passing a ready image from outside would change
     * the number of kernel evaluations and possibly the last bits of the result.
     */
    override fun doubleOmegaImages(i: Int): ImageTriple {
        val idx = i - 2
        val omega = { s: Double -> basis.omega(idx, s) }
        val image = applyL(omega)
        val imageDeriv = applyLDeriv(omega)
        // The second derivative requires the image L omega_i itself and its derivative.
        return ImageTriple(applyL(image), applyLDeriv(image), applyLDeriv2(image, imageDeriv))
    }

    // --- Nyström: spline quadrature with t-dependent weights --------------------

    /**
     * Nyström support data for Volterra: the points {eta_r} (ascending in t),
     * the ValueFunctionals of the family and the indexing of the points. Unlike Fredholm,
     * the weights W_j(t)=int_a^t omega_j depend on t, so the aggregated weights b_r(t)
     * are computed on the fly (nystromB). The family xi (de Boor–Fix) is NOT supported:
     * its functionals use a derivative and do not reduce to a linear combination of values.
     *
     * THE POINTS ARE INDEXED by the pair (functional number, node number), see [SupportPoints].
     * Previously a `HashMap<Double, Int>` lived here with a lookup by the VALUE of the point, i.e. one
     * requiring a bit-exact match of Doubles and working only because both the map
     * and the query read the same array `vf.nodes`.
     */
    private class NystromSupport(
        val pts: DoubleArray,
        val vfs: Array<ValueFunctional>,
        val support: SupportPoints,
    )

    private fun nystromSupport(): NystromSupport {
        require(!funcs.usesDerivative) {
            "Nyström is not implemented for the family '${funcs.name}': the de Boor–Fix " +
                "functionals (xi) use a derivative and do not reduce to values."
        }
        val vfs = Array(dim) { k ->
            funcs.chi(k - 2) as? ValueFunctional
                ?: error("Nyström: the functional '${funcs.name}' (j=${k - 2}) is not a ValueFunctional.")
        }
        val support = SupportPoints.byAscendingValue(vfs, grid.breakpointInclusionEps)
        return NystromSupport(support.points, vfs, support)
    }

    /**
     * Aggregated weights b_r(t):
     * b_r(t) = sum_j sum_{q: s_{j,q}=eta_r} c_{j,q} W_j(t), W_j(t)=int_a^t omega_j.
     */
    private fun nystromB(sup: NystromSupport, t: Double): DoubleArray {
        val b = DoubleArray(sup.pts.size)
        for (k in 0 until dim) {
            val j = k - 2
            val lo = grid.x(j)                 // left end of the support of omega_j (>= a)
            val hi = minOf(grid.x(j + 3), t)   // right end, truncated by the upper limit t
            if (hi <= lo) continue             // the support lies to the right of t -> W_j(t)=0 (causality)
            val w = op.integrateRange(lo, hi) { s -> basis.omega(j, s) }
            val vf = sup.vfs[k]
            for (q in vf.nodes.indices) b[sup.support.indexOf(k, q)] += vf.coeffs[q] * w
        }
        return b
    }

    /** u^N_h(t) = f(t) + cL sum_r b_r(t) K(t, eta_r) u_hat_r. */
    private fun nystromEval(sup: NystromSupport, uHat: DoubleArray, t: Double): Double {
        val b = nystromB(sup, t)
        var acc = 0.0
        for (r in sup.pts.indices) acc += b[r] * op.kernel.k(t, sup.pts[r]) * uHat[r]
        return fEff(t) + cL * acc
    }

    /** Solves (I - A^{N,V}) u_hat = f_hat: A^{N,V}_{rho,r}=cL b_r(eta_rho) K(eta_rho,eta_r) (2.3). */
    private fun nystromSolve(sup: NystromSupport): DoubleArray {
        val p = sup.pts.size
        val a = DenseMatrix.zeros(p, p)
        for (rho in 0 until p) {
            val b = nystromB(sup, sup.pts[rho]) // the t-dependent weights at t=eta_rho
            for (r in 0 until p) a[rho, r] = -cL * b[r] * op.kernel.k(sup.pts[rho], sup.pts[r])
            a[rho, rho] += 1.0
        }
        return LinearAlgebra.solve(a, DoubleArray(p) { fEff(sup.pts[it]) }, ctx.backend)
    }

    /**
     * The CLASSICAL spline Nyström for the Volterra equation: a quadrature with
     * t-dependent weights W_j(t)=int_a^t omega_j. It leads to the linear system
     * (I - A^{N,V}) u_hat = f_hat in the values of the solution at the support points {eta_r}.
     * The approximation lies outside the spline space. The family xi is not supported.
     *
     * ON THE STRUCTURE OF THE MATRIX: it used to be claimed here that, with the points ordered
     * ascending, the matrix is (block-)lower-triangular "by causality". That claim has been
     * REMOVED as UNSUBSTANTIATED: b_r(eta_rho) aggregates the coefficients of functionals
     * whose support points may lie to the right of eta_rho (the supports of omega_j overlap),
     * so in general the upper entries are non-zero. The code solves the system
     * with a general LU decomposition anyway and does not rely on triangularity.
     *
     * IMPORTANT on the order: this is a "bare" quadrature which by itself does NOT raise the order;
     * see [combinedNystrom]. For the Volterra equation there are no theoretical superconvergence estimates
     * in the known literature for any of the variants (the variable upper limit
     * gives t-dependent weights and a truncation of the last cell — a separate analysis is required).
     * Any orders observed here are a numerical observation, not a proven result.
     */
    public fun nystrom(): SolutionFunc {
        val sup = nystromSupport()
        val uHat = nystromSolve(sup)
        return SolutionFunc(eval = { t -> nystromEval(sup, uHat, t) })
    }

    /**
     * Iterated Nyström: u_hat^N_h(t)=f(t)+(L u^N_h)(t) with the EXACT Volterra
     * operator L (the closure applyL, as in sloan()). A single integration of the computed
     * u^N_h, no new system is required (the analogue of the Sloan iteration).
     */
    public fun iteratedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        val uHat = nystromSolve(sup)
        val uN = applyL { s -> nystromEval(sup, uHat, s) }
        return SolutionFunc(eval = { t -> fEff(t) + uN(t) })
    }

    /**
     * The COMBINED Nyström operator for the Volterra equation:
     * u^N_h = f + L_n u^N_h, where L_n = P_chi L + (I - P_chi) L^N_h.
     *
     * On the range of the projector the EXACT operator acts, on the complement — the quadrature with
     * t-dependent weights. The difference from [nystrom]: there u = f + L^N_h u is solved.
     *
     * SOURCE STATUS: the construction L_n is taken from the theory for the Fredholm equation
     * (see [solvers.fredholm.FredholmSecondKindSolver.combinedNystrom]); for the Volterra equation this is
     * an ADAPTATION: there is NO proof of superconvergence in the known literature. The behaviour
     * should be treated as a numerical observation.
     *
     * @throws IllegalStateException if the iteration did not converge and [throwOnDivergence] is `true`.
     */
    public fun combinedNystrom(): SolutionFunc {
        val sup = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        // The stopping criterion is measured on THE SAME set [checkPoints] as in `kulkarniQuasi`.
        // A local recomputation by the same formula would be a second criterion in one class:
        // a fix to one of them would let the two schemes silently drift apart.
        val checkPoints = this.checkPoints
        var uAtCheck = DoubleArray(checkPoints.size) { uFun(checkPoints[it]) }
        val stop = IterationStopCriterion(COMBINED_NYSTROM_TOLERANCE)
        while (stop.performedIterations < COMBINED_NYSTROM_MAX_ITERATIONS) {
            val currentFun = uFun
            val currentAtPoints = DoubleArray(sup.pts.size) { currentFun(sup.pts[it]) }
            // The exact operator L and its projection P_chi(L u).
            val exactImage = applyL(currentFun)
            val projectedExact = funcs.projectorCoeffs(exactImage)
            // The quadrature operator L^N_h with t-dependent weights and its projection.
            val quadratureImage = { t: Double ->
                val b = nystromB(sup, t)
                var acc = 0.0
                for (r in sup.pts.indices) acc += b[r] * op.kernel.k(t, sup.pts[r]) * currentAtPoints[r]
                cL * acc
            }
            val projectedQuadrature = funcs.projectorCoeffs(quadratureImage)
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(projectedExact, t) +
                    quadratureImage(t) - basis.evalSpline(projectedQuadrature, t)
            }
            val nextAtCheck = DoubleArray(checkPoints.size) { nextFun(checkPoints[it]) }
            var diff = 0.0
            for (k in nextAtCheck.indices) diff = maxOf(diff, abs(nextAtCheck[k] - uAtCheck[k]))
            uFun = nextFun
            uAtCheck = nextAtCheck
            if (stop.accept(diff)) break
        }
        reportConvergence(
            converged = stop.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Combined Nyström (Volterra)",
            iterations = stop.performedIterations,
            maxIterations = COMBINED_NYSTROM_MAX_ITERATIONS,
            residual = stop.residual,
            tolerance = COMBINED_NYSTROM_TOLERANCE,
            hint = "Convergence of the simple iteration requires ||L_n|| < 1",
            diverged = stop.diverged,
        )
        val resultFun = uFun
        return SolutionFunc(
            eval = { t -> resultFun(t) },
            converged = stop.converged,
            iterations = stop.performedIterations,
            residual = stop.residual,
        )
    }

    /**
     * Iterated combined Nyström: \hat u^N_h = f + L u^N_h with the exact L.
     * The convergence flag is inherited from [combinedNystrom].
     */
    public fun iteratedCombinedNystrom(): SolutionFunc {
        val combined = combinedNystrom()
        val image = applyL { s -> combined.eval(s) }
        return SolutionFunc(
            eval = { t -> fEff(t) + image(t) },
            converged = combined.converged,
            iterations = combined.iterations,
            residual = combined.residual,
        )
    }
}
