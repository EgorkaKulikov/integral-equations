package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import splines.Grid
import kotlin.math.abs
import kotlin.math.min

/**
 * Weakly singular kernel `|t − s|^(−alpha) · k(t, s)` with `0 < alpha < 1` and a smooth factor `k`.
 *
 * The singular factor is stored apart from `k` because it is never sampled: the operators of this file hand it to
 * [AlgebraicSingularQuadrature], which absorbs it into Gauss–Jacobi weights, and only `k(t, s)·u(s)` is evaluated
 * at quadrature nodes. Folding the factor into `k` would put an unbounded function under a Gauss–Legendre rule.
 * `alpha = 0` is excluded on purpose: the smooth kernel is served by the existing Volterra and Fredholm operators,
 * and `alpha ≥ 1` makes the kernel non-integrable.
 *
 * @param alpha exponent of the singularity, `0 < alpha < 1`
 * @param k smooth factor of the kernel; identically one by default (the Abel kernel)
 * @throws IllegalArgumentException if `alpha` is not in `(0, 1)`
 */
public class KernelWS(public val alpha: Double, public val k: (Double, Double) -> Double = { _, _ -> 1.0 }) {
    init {
        require(alpha > 0.0 && alpha < 1.0) { "weakly singular kernel exponent alpha must lie in (0, 1), got $alpha" }
    }
}

/**
 * Integral operator with a weakly singular kernel [KernelWS] on `[grid.a, grid.b]`.
 *
 * The interface is sealed because the set of operators is closed: the only difference between the
 * implementations is whether the upper limit of integration is `t` (Volterra) or `b` (Fredholm), and a solver
 * that assembles a system must know which of the two it is applying.
 */
public sealed interface WeaklySingularOperator {
    /** The kernel `|t − s|^(−alpha) k(t, s)`. */
    public val kernel: KernelWS

    /** The grid whose breakpoints form the composite partition of the integration interval. */
    public val grid: Grid

    /** Value `(Ku)(t)` over the whole integration interval; NaN for a NaN `t`. */
    public fun apply(t: Double, u: (Double) -> Double): Double

    /**
     * The same integral restricted to `[lo, hi]` (the support of `u`, typically grid breakpoints or `a`, `b`).
     * Restricting the partition to the support keeps the cost per point independent of the number of cells
     * when `u` is a basis function with compact support. Returns 0 for an empty interval and NaN for a NaN `t`.
     */
    public fun applyOnSupport(t: Double, lo: Double, hi: Double, u: (Double) -> Double): Double
}

/**
 * Weakly singular Volterra operator `(Vu)(t) = ∫_a^t (t − s)^(−alpha) k(t, s) u(s) ds`.
 *
 * The partition of `[lo, min(hi, t)]` consists of its ends and the grid breakpoints strictly inside it, selected
 * with the single tolerance [Grid.breakpointInclusionEps] exactly as `solvers.volterra.VolterraOperator` does;
 * `t` is therefore always the right end of the last cell and the quadrature uses the Gauss–Jacobi rule there.
 * All quadrature is delegated to [quad] (AGENTS.md §2): this class only chooses the partition.
 *
 * @param kernel the kernel; its exponent must coincide with `quad.alpha`
 * @param grid the grid that supplies the breakpoints
 * @param quad product quadrature for the factor `|t − s|^(−alpha)`
 * @param endpointRefinement number `m ≥ 0` of extra breakpoints `a + (x_1 − a)·2^(−i)`, `i = 1..m`, inserted into
 *   the first cell when the integration interval starts at `grid.a`. A density behaving like `(s − a)^beta` is
 *   not smooth on the first cell, and no number of nodes per cell removes the resulting `O(h^(1+beta))` error;
 *   the geometric partition confines the non-smooth part to a cell of length `(x_1 − a)·2^(−m)`. The value `m`
 *   is chosen by the caller: the observed effect (see `WeaklySingularOperatorsTest`) is a numerical observation,
 *   not a proven bound. With `m = 0` the partition coincides with the grid.
 * @throws IllegalArgumentException if `quad.alpha != kernel.alpha` or `endpointRefinement < 0`
 */
public class WeaklySingularVolterraOperator(
    public override val kernel: KernelWS,
    public override val grid: Grid,
    public val quad: AlgebraicSingularQuadrature,
    public val endpointRefinement: Int = 0,
) : WeaklySingularOperator {
    init {
        requireMatchingExponent(kernel, quad)
        requireRefinement(endpointRefinement)
    }

    /** `∫_a^t`; `t` beyond `b` is integrated up to `t`, as in `solvers.volterra.VolterraOperator`. */
    public override fun apply(t: Double, u: (Double) -> Double): Double = applyOnSupport(t, grid.a, t, u)

    /** `∫_lo^min(hi, t)`; zero when `t ≤ lo`. */
    public override fun applyOnSupport(t: Double, lo: Double, hi: Double, u: (Double) -> Double): Double {
        if (t.isNaN()) return Double.NaN
        val upper = min(hi, t)
        if (upper <= lo) return 0.0
        val k = kernel.k
        return quad.integrate(compositePartition(grid, lo, upper, endpointRefinement, 0), t) { s -> k(t, s) * u(s) }
    }
}

/**
 * Weakly singular Fredholm operator `(Fu)(t) = ∫_a^b |t − s|^(−alpha) k(t, s) u(s) ds`.
 *
 * The partition is built as for [WeaklySingularVolterraOperator]; `t` may lie inside a cell, at a breakpoint or
 * outside `[lo, hi]` — [AlgebraicSingularQuadrature.integrate] splits the cell at `t` or refines toward it.
 *
 * @param endpointRefinement as for [WeaklySingularVolterraOperator], applied at `grid.a` when the interval starts
 *   there and mirrored at `grid.b` (points `b − (b − x_{n−1})·2^(−i)`) when the interval ends there
 * @throws IllegalArgumentException if `quad.alpha != kernel.alpha` or `endpointRefinement < 0`
 */
public class WeaklySingularFredholmOperator(
    public override val kernel: KernelWS,
    public override val grid: Grid,
    public val quad: AlgebraicSingularQuadrature,
    public val endpointRefinement: Int = 0,
) : WeaklySingularOperator {
    init {
        requireMatchingExponent(kernel, quad)
        requireRefinement(endpointRefinement)
    }

    /** `∫_a^b`. */
    public override fun apply(t: Double, u: (Double) -> Double): Double = applyOnSupport(t, grid.a, grid.b, u)

    /** `∫_lo^hi`; zero when `hi ≤ lo`. */
    public override fun applyOnSupport(t: Double, lo: Double, hi: Double, u: (Double) -> Double): Double {
        if (t.isNaN()) return Double.NaN
        if (hi <= lo) return 0.0
        val k = kernel.k
        val partition = compositePartition(grid, lo, hi, endpointRefinement, endpointRefinement)
        return quad.integrate(partition, t) { s -> k(t, s) * u(s) }
    }
}

private fun requireMatchingExponent(kernel: KernelWS, quad: AlgebraicSingularQuadrature) {
    require(quad.alpha == kernel.alpha) {
        "quadrature exponent ${quad.alpha} does not match kernel exponent ${kernel.alpha}"
    }
}

private fun requireRefinement(m: Int) {
    require(m >= 0) { "endpointRefinement must be non-negative, got $m" }
}

/**
 * Strictly increasing partition of `[lo, hi]`: `lo`, the optional geometric points toward `grid.a` (when
 * `lo == grid.a`), the grid breakpoints strictly inside, the optional mirrored points toward `grid.b` (when
 * `hi == grid.b`), and `hi`. "Strictly inside" uses [Grid.breakpointInclusionEps], the single source of the
 * tolerance (AGENTS.md §5). A candidate is also required to exceed the previous point by more than the tolerance,
 * so that coincident breakpoints or refinement points that underflow toward `a` never produce an empty cell.
 */
private fun compositePartition(grid: Grid, lo: Double, hi: Double, refineAtA: Int, refineAtB: Int): DoubleArray {
    val eps = grid.breakpointInclusionEps
    val bp = grid.breakpoints
    val points = ArrayList<Double>(bp.size + refineAtA + refineAtB + 2)
    points.add(lo)
    fun addInside(x: Double) {
        if (x > points.last() + eps && x < hi - eps) points.add(x)
    }
    if (refineAtA > 0 && abs(lo - grid.a) <= eps) {
        val first = bp.first { it > grid.a + eps }
        for (i in refineAtA downTo 1) addInside(grid.a + Math.scalb(first - grid.a, -i))
    }
    for (x in bp) addInside(x)
    if (refineAtB > 0 && abs(hi - grid.b) <= eps) {
        val last = bp.last { it < grid.b - eps }
        for (i in 1..refineAtB) addInside(grid.b - Math.scalb(grid.b - last, -i))
    }
    points.add(hi)
    return points.toDoubleArray()
}
