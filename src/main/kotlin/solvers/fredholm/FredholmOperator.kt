package solvers.fredholm

import numerics.*
import splines.*
import splines.functionals.*
import splines.metrics.*

/**
 * The kernel K(t,s) of a linear Fredholm equation together with its analytic partial
 * derivatives.
 *
 * @param k the kernel `K(t,s)` itself.
 * @param kT the derivative `K_t(t,s)`; required by the de Boor–Fix functional families
 *        `xi^<1>`, `xi^<2>`.
 * @param kTT the second derivative `K_tt(t,s)`; required by the family `xi^<0>`.
 *
 * The default values are zero and are admissible ONLY when the corresponding
 * derivative is indeed identically zero, or when the chosen family of
 * functionals does not use it: otherwise the system is built incorrectly without any
 * diagnostics.
 */
public class KernelF(
    public val k: (Double, Double) -> Double,
    public val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
)

/**
 * Fredholm operator `(K u)(t) = \int_a^b K(t,s) u(s) ds` with constant integration
 * limits.
 *
 * At construction time the global Gauss nodes [gNode] and weights [gW] of the composite
 * quadrature are precomputed, so that `\int h = sum_k gW[k] * h(gNode[k])`. This makes it possible
 * to apply the operator repeatedly to the function values already computed at those nodes
 * (see [applyNodes]) without recomputing them.
 *
 * @param kernel kernel of the equation.
 * @param grid grid defining the interval `[a,b]` and the breakpoints of the composite quadrature.
 * @param quad Gauss–Legendre quadrature rule on a cell.
 */
public class FredholmOperator(public val kernel: KernelF, public val grid: Grid, public val quad: GaussLegendre) {
    /**
     * Global nodes of the composite quadrature.
     *
     * READ-ONLY BY CONVENTION: the contents MUST NOT be modified. A HOT field — a copy is
     * deliberately not returned: the array is read in the inner loops of [applyNodes],
     * [applyDerivNodes], [applyDeriv2Nodes] and during matrix assembly — copying on every
     * access would give a quadratic growth of allocations. There are no writes in the project.
     */
    public val gNode: DoubleArray

    /** Quadrature weights at the nodes [gNode]. READ-ONLY by convention (hot, see [gNode]). */
    public val gW: DoubleArray

    init {
        val (rn, rw) = quad.refNodesWeights()
        val bp = grid.breakpoints
        val nodes = ArrayList<Double>()
        val ws = ArrayList<Double>()
        for (m in 0 until bp.size - 1) {
            val lo = bp[m]; val hi = bp[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo); val mid = 0.5 * (hi + lo)
            for (qi in rn.indices) { nodes.add(mid + half * rn[qi]); ws.add(half * rw[qi]) }
        }
        gNode = nodes.toDoubleArray()
        gW = ws.toDoubleArray()
    }

    /** (\mathcal K u)(t) for an arbitrary u(s). */
    public fun apply(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s) * u(s) }

    /** d/dt (\mathcal K u)(t) = \int_a^b dK/dt(t,s) u(s) ds (for the xi functionals). */
    public fun applyDeriv(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kT(t, s) * u(s) }

    /** d^2/dt^2 (\mathcal K u)(t) = \int_a^b d^2K/dt^2(t,s) u(s) ds (for xi^<0>). */
    public fun applyDeriv2(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kTT(t, s) * u(s) }

    /** (\mathcal K u)(tau) from the precomputed values of u at the global nodes. */
    public fun applyNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k]) * uNodes[k]
        return s
    }

    /** d/dt (\mathcal K u)(tau) from the precomputed uNodes. */
    public fun applyDerivNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kT(tau, gNode[k]) * uNodes[k]
        return s
    }

    /** d^2/dt^2 (\mathcal K u)(tau) from the precomputed uNodes (for xi^<0>). */
    public fun applyDeriv2Nodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kTT(tau, gNode[k]) * uNodes[k]
        return s
    }
}
