package solvers.uryson

import numerics.GaussLegendre
import splines.Grid

/**
 * Kernel `K(t,s,u)` of the nonlinear Uryson equation and its partial derivative in `u`.
 *
 * The derivative `dK/du` is needed for the Fréchet derivative of the operator and, through
 * it, for the analytic Jacobian of Newton's method.
 */
public interface Kernel {
    /** Value of the kernel `K(t, s, u)`. */
    public fun k(t: Double, s: Double, u: Double): Double

    /** Partial derivative `dK/du(t, s, u)`. */
    public fun dkdu(t: Double, s: Double, u: Double): Double
}

/**
 * Nonlinear Uryson integral operator `(U x)(t) = \int_a^b K(t,s,x(s)) ds`.
 *
 * The integrals are evaluated by a composite Gauss–Legendre quadrature over grid intervals.
 *
 * @param kernel kernel of the equation together with its derivative in `u`.
 * @param grid grid defining the integration interval and the partition for the quadrature.
 * @param quad quadrature rule.
 */
public class UrysohnOperator(public val kernel: Kernel, public val grid: Grid, public val quad: GaussLegendre) {
    /** Value of `(U x)(t)` for an arbitrary function `x(s)`. */
    public fun apply(t: Double, x: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s, x(s)) }

    /**
     * Fréchet derivative `(U'(x) h)(t) = \int_a^b dK/du(t,s,x(s)) h(s) ds`.
     *
     * It defines the linearization of the operator at `x`; this very formula yields the
     * Jacobian [CollocationCore.bMatrix], which is computed more efficiently — in a single
     * pass over the quadrature nodes for all basis functions at once.
     */
    public fun frechet(t: Double, x: (Double) -> Double, h: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.dkdu(t, s, x(s)) * h(s) }

    /**
     * Global set of quadrature nodes over the whole grid: `\int h = sum_k gW[k] h(gNode[k])`.
     *
     * Precomputing it lets the Kulkarni and Nyström schemes avoid rebuilding the partition
     * on every evaluation of the nested integrals.
     *
     * READ-ONLY BY CONVENTION: the content MUST NOT be modified. The field is HOT — a copy is
     * deliberately not returned: the array is read in the [applyNodes] loop and on every
     * quasi-Newton iteration. There are no writes in the project.
     */
    public val gNode: DoubleArray

    /** Quadrature weights matching the nodes [gNode]. READ-ONLY by convention (hot). */
    public val gW: DoubleArray

    init {
        val (referenceNodes, referenceWeights) = quad.refNodesWeights()
        val breakpoints = grid.breakpoints
        val nodes = ArrayList<Double>()
        val weights = ArrayList<Double>()
        for (m in 0 until breakpoints.size - 1) {
            val lo = breakpoints[m]
            val hi = breakpoints[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in referenceNodes.indices) {
                nodes.add(mid + half * referenceNodes[q])
                weights.add(half * referenceWeights[q])
            }
        }
        gNode = nodes.toDoubleArray()
        gW = weights.toDoubleArray()
    }

    /** Value of `(U x)(tau)` from the precomputed values of `x` at the nodes [gNode]. */
    public fun applyNodes(tau: Double, xNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k], xNodes[k])
        return s
    }
}
