package solvers.fredholm

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
 * Linear solver for the second-kind equation u - L u = f, L = c_L * \mathcal K
 * (c_L = 1 for F2; c_L = -1/alpha for F1 Wazwaz, \mathcal K_eff = -(1/alpha)\mathcal K).
 * The right-hand side f and its derivatives are supplied explicitly ([RhsWithDerivatives]) for reuse in F1.
 *
 * Matrices of the discrete problem:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * @param throwOnDivergence behaviour of the ITERATIVE schemes ([kulkarni] for
 *        quasi-interpolants, [combinedNystrom]) when convergence is not reached:
 *        `true` (default) — an exception, `false` — a result with
 *        `converged = false` and the attained residual in [SolutionFunc.residual].
 *        Direct schemes are unaffected. The parameter is set at the solver level
 *        rather than per method: it is an error-handling policy, not a property of
 *        an individual scheme.
 */
public class FredholmSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    public val op: FredholmOperator,
    cL: Double,
    rhs: RhsWithDerivatives,
    throwOnDivergence: Boolean = true,
    ctx: NumericsContext = NumericsContext.default(),
) : SecondKindSolverCore<DoubleArray>(
    basis, funcs, cL, rhs, throwOnDivergence, ctx,
) {
    // The numerical parameters of the iterative schemes (COMBINED_NYSTROM_*, KULKARNI_QUASI_*)
    // live in [solvers.core.SecondKindDefaults]: they coincide for Fredholm and Volterra,
    // and a divergence while tuning a new value would be silent.
    //
    // The fields `grid`, `n`, `dim` are declared in [SecondKindSolverCore] and initialized BEFORE
    // the fields of this class — exactly the order [omegaNodes] needs (it is
    // eager and reads `dim`). The reverse order would give `dim = 2` SILENTLY.

    private val ng = op.gNode.size

    /**
     * `L omega_i` at the global Gauss nodes:
     * `LomegaNodes[i][k] = c_L (\mathcal K omega_{i-2})(gNode[k])`. A LAZY field.
     *
     * Why lazy: the only consumer is [matrixM2] (the second application of L),
     * and inside the class `M2` is needed only by the Kulkarni scheme for PROJECTORS
     * ([kulkarni] → `kulkarniProjector`, where `I - M - M2 + M^2` is built); [matrixM2]
     * is public and is also called from the tests. The other schemes — [base], [sloan],
     * `kulkarniQuasi` for quasi-interpolants, the whole Nyström family ([nystrom],
     * [combinedNystrom]) — do not touch `M2` at all.
     *
     * The assembly cost is high: `dim * ng` integrals (one per pair
     * `i, k`), each of them a quadrature over `ng` nodes, i.e. O(dim * ng^2)
     * kernel evaluations. With an eager field ALL schemes paid that price, including
     * those that do not use it.
     *
     * The laziness mode is the default one (`SYNCHRONIZED`), and this is mandatory: the field
     * is read from the threads of the parallel assembly ([ParallelAssembly] in [matrixM2]),
     * so the first access may come from a worker thread, and `NONE` would be a race.
     * The synchronization does not become a bottleneck: the read happens once per column
     * (`dim` times in total), not in the inner loop over the nodes.
     */
    private val LomegaNodes: Array<DoubleArray> by lazy {
        Array(dim) { ki ->
            val i = ki - 2
            DoubleArray(ng) { k -> cL * op.apply(op.gNode[k]) { s -> basis.omega(i, s) } }
        }
    }

    /**
     * `omega_i` at the global nodes: `omegaNodes[i][k] = omega_{i-2}(gNode[k])` — the argument
     * for `applyNodes` while assembling [matrixM].
     *
     * The field is left EAGER NOT because all schemes need it (only [matrixM] reads it,
     * while the Nyström family and `kulkarniQuasi` do not require it), but because
     * it is cheap: `dim * ng` spline evaluations without kernel calls, i.e. `ng` times
     * cheaper than [LomegaNodes]. Laziness here would only add overhead.
     */
    private val omegaNodes: Array<DoubleArray> = Array(dim) { ki ->
        val i = ki - 2
        DoubleArray(ng) { k -> basis.omega(i, op.gNode[k]) }
    }

    // --- Implementation of the [SecondKindSolverCore] extension points ---------
    //
    // For Fredholm the prepared operand is the array of function values at the global
    // Gauss nodes: the integration limits are constant, so the same nodes
    // serve any `t`, and a repeated application of L reduces to a convolution with the kernel.

    override val equationName: String get() = "Fredholm"

    override val kulkarniQuasiHint: String
        get() = "For quasi-interpolants (mu, lambda) the property P^2 = P does not hold, so the " +
            "Kulkarni reduction is inapplicable and a simple iteration is used, which requires " +
            "contractivity (the spectral radius of the operator is less than one)"

    /**
     * The stopping criterion is measured at the Gauss nodes of the operator — the same ones the
     * iteration itself runs on. A synthetic sample like Volterra's is not needed here
     * and would give a different iteration count.
     */
    override val checkPoints: DoubleArray get() = op.gNode

    override fun prepare(u: (Double) -> Double): DoubleArray =
        DoubleArray(ng) { u(op.gNode[it]) }

    override fun image(o: DoubleArray): (Double) -> Double = { t -> cL * op.applyNodes(t, o) }

    override fun imageDeriv(o: DoubleArray): (Double) -> Double = { t -> cL * op.applyDerivNodes(t, o) }

    /** The integration limits are constant, there is no Leibniz term, so `uD` is not read. */
    @Suppress("UNUSED_PARAMETER")
    override fun imageDeriv2(o: DoubleArray, uD: (Double) -> Double): (Double) -> Double =
        { t -> cL * op.applyDeriv2Nodes(t, o) }

    /**
     * The check points COINCIDE with the nodes used to prepare the operand, so the values
     * are already computed — a second pass over `u` would be a pure doubling of the work
     * on every iteration (`u` there is the result of applying the operator, not a table).
     */
    override fun checkValues(u: (Double) -> Double, o: DoubleArray): DoubleArray = o

    override fun applyOperator(t: Double, u: (Double) -> Double): Double = op.apply(t, u)

    override fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double = op.applyDeriv(t, u)

    /** For Fredholm the second derivative of the image contains no `u'`: the limits are constant. */
    @Suppress("UNUSED_PARAMETER")
    override fun applyOperatorDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double =
        op.applyDeriv2(t, u)

    override fun omegaImages(i: Int): ImageTriple {
        val on = omegaNodes[i]
        return ImageTriple(
            { t -> cL * op.applyNodes(t, on) },
            { t -> cL * op.applyDerivNodes(t, on) },
            { t -> cL * op.applyDeriv2Nodes(t, on) },
        )
    }

    override fun doubleOmegaImages(i: Int): ImageTriple {
        val ln = LomegaNodes[i] // (L omega_i) at the nodes
        return ImageTriple(
            { t -> cL * op.applyNodes(t, ln) },
            { t -> cL * op.applyDerivNodes(t, ln) },
            { t -> cL * op.applyDeriv2Nodes(t, ln) },
        )
    }

    // --- Nyström (spline quadrature; see docs/REFERENCES.md, section 3) -------

    /**
     * Support points {eta_r} and aggregated weights b_r of the base Nyström scheme:
     * b_r = sum_j sum_{q: s_{j,q}=eta_r} c_{j,q} W_j, W_j = int_a^b omega_j.
     * The points are ordered ascending (for uniformity with Volterra; for F2 the order
     * is immaterial). The family xi (de Boor–Fix) is NOT supported: its functionals
     * use a derivative and do not reduce to a linear combination of values
     * (a known limitation of the method; it is worked around by the family xitilde).
     *
     * THE POINTS ARE INDEXED by the pair (functional number, node number), see [SupportPoints].
     * Previously the index was looked up by the VALUE of the point in a `HashMap<Double, Int>`, i.e. it relied
     * on a bit-exact match of Doubles; that worked only because both the map and the query
     * read the same array `vf.nodes`. Merging the coinciding points of different
     * functionals now goes by the EXPLICIT tolerance `grid.breakpointInclusionEps`.
     */
    private fun nystromSupport(): Pair<DoubleArray, DoubleArray> {
        require(!funcs.usesDerivative) {
            "Nyström is not implemented for the family '${funcs.name}': the de Boor–Fix " +
                "functionals (xi) use a derivative and do not reduce to values."
        }
        val vfs = Array(dim) { k ->
            funcs.chi(k - 2) as? ValueFunctional
                ?: error("Nyström: the functional '${funcs.name}' (j=${k - 2}) is not a ValueFunctional.")
        }
        // W_j = int_a^b omega_j (a high-accuracy composite quadrature over the grid breakpoints).
        val wJ = DoubleArray(dim) { k -> op.quad.integrate(grid.breakpoints) { s -> basis.omega(k - 2, s) } }
        val support = SupportPoints.byAscendingValue(vfs, grid.breakpointInclusionEps)
        val pts = support.points
        val bAgg = DoubleArray(support.size)
        for (k in 0 until dim) {
            val vf = vfs[k]; val w = wJ[k]
            for (q in vf.nodes.indices) bAgg[support.indexOf(k, q)] += vf.coeffs[q] * w
        }
        return pts to bAgg
    }

    /** u^N_h(t) = f(t) + cL sum_r b_r K(t, eta_r) u_hat_r — reconstruction of the solution. */
    private fun nystromEval(t: Double, pts: DoubleArray, bAgg: DoubleArray, uHat: DoubleArray): Double =
        fEff(t) + nystromQuadrature(t, pts, bAgg, uHat)

    /**
     * The quadrature operator (L^N_h u)(t) = cL sum_r b_r K(t, eta_r) u(eta_r).
     *
     * It depends on u only through its values at the support points [uAtPoints] — this is exactly
     * the property that makes the operator finite-rank.
     */
    private fun nystromQuadrature(
        t: Double,
        pts: DoubleArray,
        bAgg: DoubleArray,
        uAtPoints: DoubleArray,
    ): Double {
        var acc = 0.0
        for (r in pts.indices) acc += bAgg[r] * op.kernel.k(t, pts[r]) * uAtPoints[r]
        return cL * acc
    }

    /** The matrix (I - A^N): A^N_{rho,r} = cL b_r K(eta_rho, eta_r). */
    private fun nystromMatrix(pts: DoubleArray, bAgg: DoubleArray): DenseMatrix {
        val p = pts.size
        val a = DenseMatrix.zeros(p, p)
        for (rho in 0 until p) {
            for (r in 0 until p) a[rho, r] = -cL * bAgg[r] * op.kernel.k(pts[rho], pts[r])
            a[rho, rho] += 1.0
        }
        return a
    }

    /**
     * The CLASSICAL spline Nyström: the integrand g_t(s)=K(t,s)u(s)
     * is replaced by its spline (quasi-)projection and the integral by the quadrature
     * sum_j chi_j(g_t) W_j. The equation u = f + L^N_h u is solved, which gives the linear system
     * (I - A^N) u_hat = f_hat in the VALUES of the solution at the support points {eta_r}.
     * The approximation u^N_h lies OUTSIDE the spline space. The family xi is not supported.
     *
     * IMPORTANT on the convergence order: this is a "bare" quadrature which by itself
     * does NOT raise the order. The published superconvergence estimates O(h^7)/O(h^8)
     * refer NOT to it but to the combined operator — see [combinedNystrom].
     */
    public fun nystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        val uHat = LinearAlgebra.solve(nystromMatrix(pts, bAgg), DoubleArray(pts.size) { fEff(pts[it]) }, ctx.backend)
        return SolutionFunc(eval = { t -> nystromEval(t, pts, bAgg, uHat) })
    }

    /**
     * Iterated Nyström: u_hat^N_h(t)=f(t)+(L u^N_h)(t) with the
     * EXACT operator L (the high-accuracy quadrature op.applyNodes, as in sloan()). A single
     * integration of the computed u^N_h, no new system is required (the analogue of the Sloan iteration).
     */
    public fun iteratedNystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        val uHat = LinearAlgebra.solve(nystromMatrix(pts, bAgg), DoubleArray(pts.size) { fEff(pts[it]) }, ctx.backend)
        val uNodes = DoubleArray(ng) { nystromEval(op.gNode[it], pts, bAgg, uHat) }
        return SolutionFunc(eval = { t -> fEff(t) + cL * op.applyNodes(t, uNodes) })
    }

    /**
     * The COMBINED Nyström operator: u^N_h = f + L_n u^N_h, where
     *
     *     L_n = P_chi L + (I - P_chi) L^N_h,
     *
     * that is, on the range of the projector the EXACT operator acts, and on its complement —
     * the quadrature. The difference from [nystrom]: there u = f + L^N_h u is solved (the "bare"
     * quadrature, the classical Nyström).
     *
     * Why this is needed: in the difference L - L_n = (I - P_chi)(L - L^C_h) the projector remainder
     * (I - P_chi) enters TWICE — as an explicit factor and inside the quadrature remainder —
     * which is what yields the superconvergence. It is to this operator, not to the bare quadrature,
     * that the published order estimates O(h^7) and O(h^8) refer (see the list of sources
     * in docs/REFERENCES.md: Allouch, Remogna, Sbibih, Tahrichi, AMC 404 (2021), Art. 126227;
     * Remogna, Sbibih, Tahrichi, Mathematics 11 (2023), Art. 3236).
     *
     * Solution method: the simple iteration u^{(m+1)} = f + L_n u^{(m)}. The operator L_n has
     * finite rank, so the problem is equivalent to a finite-dimensional linear system; the iteration was chosen
     * as a substantially simpler implementation (a direct assembly requires P×P integrals
     * of the form ∫K(t,s)K(s,eta_r)ds). The convergence is linear with ratio ||L_n|| and requires
     * ||L_n|| < 1; if convergence is not reached an exception is thrown (rather than
     * silently returning a wrong result).
     *
     * @throws IllegalStateException if the iteration did not converge and [throwOnDivergence] is `true`.
     */
    public fun combinedNystrom(): SolutionFunc {
        val (pts, bAgg) = nystromSupport()
        var uFun: (Double) -> Double = { t -> fEff(t) }
        var uAtNodes = DoubleArray(ng) { uFun(op.gNode[it]) }
        val stop = IterationStopCriterion(COMBINED_NYSTROM_TOLERANCE)
        while (stop.performedIterations < COMBINED_NYSTROM_MAX_ITERATIONS) {
            val currentFun = uFun
            val currentNodes = uAtNodes
            val currentAtPoints = DoubleArray(pts.size) { currentFun(pts[it]) }
            // The exact operator L on the current iterate and its projection P_chi(L u).
            val exactImage = { t: Double -> cL * op.applyNodes(t, currentNodes) }
            val projectedExact = funcs.projectorCoeffs(exactImage)
            // The quadrature operator L^N_h and its projection P_chi(L^N_h u).
            val quadratureImage = { t: Double -> nystromQuadrature(t, pts, bAgg, currentAtPoints) }
            val projectedQuadrature = funcs.projectorCoeffs(quadratureImage)
            // u^{(m+1)} = f + P_chi(L u) + L^N_h u - P_chi(L^N_h u).
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(projectedExact, t) +
                    quadratureImage(t) - basis.evalSpline(projectedQuadrature, t)
            }
            val nextNodes = DoubleArray(ng) { nextFun(op.gNode[it]) }
            var diff = 0.0
            for (k in 0 until ng) diff = maxOf(diff, abs(nextNodes[k] - currentNodes[k]))
            uFun = nextFun
            uAtNodes = nextNodes
            if (stop.accept(diff)) break
        }
        reportConvergence(
            converged = stop.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Combined Nyström (Fredholm)",
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
     * Iterated combined Nyström: \hat u^N_h = f + L u^N_h with the EXACT
     * operator L (the analogue of the Sloan iteration; it requires no new system).
     */
    public fun iteratedCombinedNystrom(): SolutionFunc {
        val combined = combinedNystrom()
        val uNodes = DoubleArray(ng) { combined.eval(op.gNode[it]) }
        // The convergence flag is inherited from the underlying combined operator.
        return SolutionFunc(
            eval = { t -> fEff(t) + cL * op.applyNodes(t, uNodes) },
            converged = combined.converged,
            iterations = combined.iterations,
            residual = combined.residual,
        )
    }
}
