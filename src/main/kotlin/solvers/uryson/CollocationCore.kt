package solvers.uryson

import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.Grid
import splines.MinimalSplineBasis
import numerics.NumericsContext
import numerics.ParallelAssembly
import splines.functionals.ProjFunctionals
import solvers.core.SupportPoints

/**
 * Core of the collocation computations: the vector `Xi(c) = Theta_h(U x_h)` and the Jacobian
 * `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
 *
 * Used both by the second-kind schemes (Newton's method) and by the regularized first-kind
 * scheme (Gauss–Newton method).
 */
public class CollocationCore(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val op: UrysohnOperator,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    private val quad = op.quad
    private val kernel = op.kernel

    /**
     * Reference quadrature nodes and weights on [-1,1], obtained ONCE.
     *
     * [GaussLegendre.refNodesWeights] returns COPIES of the arrays, while [bMatrix] is called
     * ON EVERY Newton iteration (`UrysonSecondKindSolver.newtonStep`) and Gauss–Newton iteration
     * (`TikhonovSolver.solveFixedAlpha`). Fetching the nodes inside [bMatrix] would cost two
     * allocations per iteration for nothing; here the copy is made once.
     *
     * The values and the order of arithmetic are the same: the arrays are immutable and read-only.
     */
    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    /**
     * Distinct support points of all functionals `theta_j` (nodes and interval midpoints)
     * in the ORDER OF FIRST OCCURRENCE while traversing `j = -2..n-1`.
     *
     * The order is part of the contract, not an implementation detail: the numbering of the
     * rows of `G` in [bMatrix] and the summation order in [xiVector] depend on it. READ-ONLY by convention.
     */
    public val supportPts: DoubleArray

    /**
     * Indexing of the support points by the pair (functional number `j+2`, node number).
     *
     * This used to be a `HashMap<Double, Int>` keyed by the point VALUE: it required bitwise
     * equality of Doubles and worked only because both the filling and the reading came from
     * one and the same `nodes` array of the cached functional.
     */
    private val support: SupportPoints

    init {
        val (rn, rw) = quad.refNodesWeights()
        refNodes = rn
        refWeights = rw
        val vfs = Array(n + 2) { funcs.valueFunctional(it - 2) }
        support = SupportPoints.byFirstOccurrence(vfs, grid.breakpointInclusionEps)
        supportPts = support.points
    }

    /** Values of `(U x_h)` at the support points (one integral per point). */
    public fun uAtSupport(c: DoubleArray): DoubleArray =
        DoubleArray(supportPts.size) { p -> op.apply(supportPts[p]) { s -> basis.evalSpline(c, s) } }

    /** Vector `Xi(c)_j = theta_j(U x_h)`, `j = -2..n-1` (array index `j+2`). */
    public fun xiVector(c: DoubleArray): DoubleArray {
        val uVals = uAtSupport(c)
        return DoubleArray(n + 2) { k ->
            val th = funcs.valueFunctional(k - 2)
            var s = 0.0
            for (q in th.nodes.indices) s += th.coeffs[q] * uVals[support.indexOf(k, q)]
            s
        }
    }

    /**
     * Jacobian `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
     *
     * Computed in two passes: first `G[p][i] = \int dK/du(tau_p, s, x_h(s)) omega_i(s) ds`
     * for all support points `tau_p` in a single sweep over the quadrature nodes (on each interval
     * only three basis splines are nonzero), then the rows of `B` are assembled as linear
     * combinations of the rows of `G` with the coefficients of the functionals.
     */
    public fun bMatrix(c: DoubleArray): DenseMatrix {
        val np = supportPts.size
        val g = DenseMatrix.zeros(np, n + 2)
        // Nodes/weights were obtained once in init: [refNodesWeights] hands out copies, and this
        // method is called on every Newton / Gauss–Newton iteration.
        val nodes = refNodes
        val weights = refWeights
        for (m in 0 until n) {
            val lo = grid.x(m)
            val hi = grid.x(m + 1)
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in nodes.indices) {
                val s = mid + half * nodes[q]
                val weight = half * weights[q]
                val splineValue = basis.evalSpline(c, s)
                val activeOmega = basis.activeOmega(m, s) // omega_{m-2}, omega_{m-1}, omega_m
                for (p in 0 until np) {
                    val dk = kernel.dkdu(supportPts[p], s, splineValue) * weight
                    if (dk == 0.0) continue
                    g[p, m] += dk * activeOmega[0]
                    g[p, m + 1] += dk * activeOmega[1]
                    g[p, m + 2] += dk * activeOmega[2]
                }
            }
        }
        // The entries of B are independent: each task writes its own cell, g is only read.
        // The order of additions is preserved verbatim (increasing q), so the numbers are the same.
        return ParallelAssembly.assembleDense(n + 2, n + 2, ctx.parallel) { k, i ->
            val th = funcs.valueFunctional(k - 2)
            var acc = 0.0
            for (q in th.nodes.indices) acc += th.coeffs[q] * g[support.indexOf(k, q), i]
            acc
        }
    }
}
