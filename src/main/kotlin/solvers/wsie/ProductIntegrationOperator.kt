package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import numerics.GaussLegendre
import splines.Grid
import kotlin.math.max
import kotlin.math.min

/**
 * Discrete (product-integration) operator `𝓛_m` of the discrete Kulkarni scheme M-DK (spec K3):
 *   (𝓛_m v)(t) = Σ_p W_p(t) k̃(t, s_p) v(s_p),   W_p(t) = ∫_{cell(p)} κ_0(t, s) ℓ_p(s) ds.
 * The fine grid Y_m ([fineGrid], `m` cells) carries `q` ([nodesPerCell]) Gauss–Legendre nodes `s_p` strictly inside
 * every cell, `ℓ_p` is the Lagrange basis of the nodes of its cell, `κ_0 = |t − s|^(−alpha)` (Fredholm) or
 * `(t − s)^(−alpha)` on `s < t` and zero otherwise (Volterra). The factor `k̃(t, ·)v` is interpolated by `π_m`, the
 * singular factor is integrated exactly: `𝓛_m v = 𝓛(π_m[k̃(t, ·)v])` with `k ≡ 1` in the outer integral.
 * For the Volterra kernel `k` is needed on the triangle `s ≤ t` only, but the interpolation in the cell that contains
 * `t` uses all its nodes, so `k̃` is an extension of `k` to the square; the proof draft (T3′ §0) admits any continuous
 * `k̃` with `k̃ = k` on the triangle. It is chosen by [extension] (for Fredholm `k̃ = k` and [extension] is ignored):
 *  - [KernelExtension.NATURAL] (default): `k̃(t, s) = k(t, s)`, the formula of `kernel.k` on the whole square;
 *  - [KernelExtension.MAX]: `k̃(t, s) = k(max(t, s), s)`, for kernels given on the triangle only.
 *
 * WHY NATURAL IS THE DEFAULT. When `kernel.k` is smooth on the square, the natural `k̃(t, ·)` is smooth in `s`, while
 * `k(max(t, s), s)` has a kink at `s = t` wherever `∂_t k(t, t) ≠ 0` (for `k = 1 + t s`: `1 + t s` on the left,
 * `1 + s²` on the right). The kink lies inside the cell that contains `t`; the interpolation error there is of the
 * order of the cell length times the jump of `∂_s k̃` and does not decrease with `q`. Numerical observation (C8,
 * V-d, theta, B, power:3, n = 32, m = 64, q = 8): E_h of M-DK is 8.74e-3 with MAX and 3.517e-4 with NATURAL, the
 * classical M-K gives 3.517e-4. When `k` does not depend on `t` (in particular `k ≡ 1`) the two extensions coincide.
 *
 * The nodes do not depend on `t`, so `𝓛_m v` depends on `v` only through `v(s_p)`; this is the difference from
 * [WeaklySingularVolterraOperator] / [WeaklySingularFredholmOperator], which split the cell at `t`.
 *
 * WHY THE WEIGHTS ARE COMPUTED BY [AlgebraicSingularQuadrature]. `W_p(t)` is a moment of a polynomial of degree
 * `q − 1` against `κ_0(t, ·)`. When `t` lies in the cell or at its end the quadrature uses Gauss–Jacobi rules with
 * `nodesPerSub ≥ q` nodes and is exact; when `t` lies outside, the geometric Gauss–Legendre refinement of the
 * quadrature is accurate to roundoff (numerical observation, see `ProductIntegrationOperatorTest`). No new
 * quadrature primitive is introduced (AGENTS.md §2).
 *
 * Status (AGENTS.md §8): the operator is the classical product integration [Atkinson 1997, ch. 4] on the cells of
 * Y_m; norm convergence `𝓛_m → 𝓛` does NOT hold (collectively compact setting), only pointwise convergence.
 *
 * @param kernel the kernel; `kernel.k` gives `k`, `kernel.alpha` the exponent
 * @param fineGrid the grid Y_m; its distinct breakpoints are the cell ends
 * @param nodesPerCell `q ≥ 1`
 * @param volterra `true` for `∫_a^t`, `false` for `∫_a^b`
 * @param extension the extension `k̃` of `k` beyond `s ≤ t` (Volterra only), [KernelExtension.NATURAL] by default
 */
public class ProductIntegrationOperator(
    public val kernel: KernelWS,
    public val fineGrid: Grid,
    public val nodesPerCell: Int,
    public val volterra: Boolean,
    public val extension: KernelExtension = KernelExtension.NATURAL,
) {
    /** Extension `k̃` of the smooth factor `k` from the triangle `s ≤ t` to the square (Volterra only). */
    public enum class KernelExtension {
        /** `k̃(t, s) = k(t, s)`: `kernel.k` is defined by its formula on the whole square. */
        NATURAL,

        /** `k̃(t, s) = k(max(t, s), s)`: for kernels given on `s ≤ t` only; kink at `s = t` where `∂_t k(t, t) ≠ 0`. */
        MAX,
    }

    init {
        require(nodesPerCell >= 1) { "nodesPerCell must be at least 1, got $nodesPerCell" }
    }

    private val quad = AlgebraicSingularQuadrature(kernel.alpha, max(12, nodesPerCell))
    /** Distinct breakpoints of Y_m (coincident ones merged with [Grid.breakpointInclusionEps]). */
    private val cells: DoubleArray = fineGrid.breakpoints.fold(ArrayList<Double>()) { acc, x ->
        if (acc.isEmpty() || x > acc.last() + fineGrid.breakpointInclusionEps) acc.add(x)
        acc
    }.toDoubleArray()
    private val ref: DoubleArray = GaussLegendre(nodesPerCell).refNodesWeights().first.map { 0.5 * (it + 1.0) }
        .toDoubleArray()

    /** Number of cells `m` of Y_m. */
    public val cellCount: Int get() = cells.size - 1

    private val nodeArray = DoubleArray(cellCount * nodesPerCell) { p ->
        val l = p / nodesPerCell
        cells[l] + ref[p % nodesPerCell] * (cells[l + 1] - cells[l])
    }

    /** The nodes `s_p`, `p = 0..N−1`, `N = m·q`, cell by cell in ascending order (a copy). */
    public fun nodes(): DoubleArray = nodeArray.copyOf()

    /** `N = m·q`. */
    public val nodeCount: Int get() = nodeArray.size

    /** The singular weights `W_p(t)`, `p = 0..N−1`. */
    public fun weightRow(t: Double): DoubleArray {
        val w = DoubleArray(nodeArray.size)
        val q = nodesPerCell
        for (l in 0 until cellCount) {
            val c0 = cells[l]
            val c1 = if (volterra) min(cells[l + 1], t) else cells[l + 1]
            if (c1 <= c0) continue
            val part = doubleArrayOf(c0, c1)
            for (r in 0 until q) {
                val p = l * q + r
                w[p] = quad.integrate(part, t) { s -> lagrange(l, r, s) }
            }
        }
        return w
    }

    /** `k̃(t, s)`: `k(max(t, s), s)` for Volterra with [KernelExtension.MAX], `k(t, s)` otherwise. */
    public fun kTilde(t: Double, s: Double): Double =
        if (volterra && extension == KernelExtension.MAX) kernel.k(max(t, s), s) else kernel.k(t, s)

    /** Row `W_p(t)·k̃(t, s_p)`; `(𝓛_m v)(t)` is its scalar product with `v(s_p)`. */
    public fun kernelRow(t: Double): DoubleArray {
        val w = weightRow(t)
        for (p in w.indices) if (w[p] != 0.0) w[p] *= kTilde(t, nodeArray[p])
        return w
    }

    /** `(𝓛_m v)(t)` from the node values `v(s_p)`. */
    public fun apply(t: Double, values: DoubleArray): Double {
        require(values.size == nodeArray.size) { "expected ${nodeArray.size} node values, got ${values.size}" }
        val row = kernelRow(t)
        var acc = 0.0
        for (p in row.indices) acc += row[p] * values[p]
        return acc
    }

    /** `(𝓛_m u)(t)`; `u` is sampled at the nodes only. */
    public fun apply(t: Double, u: (Double) -> Double): Double = apply(t, DoubleArray(nodeArray.size) { u(nodeArray[it]) })

    private fun lagrange(l: Int, r: Int, s: Double): Double {
        val base = l * nodesPerCell
        val sr = nodeArray[base + r]
        var v = 1.0
        for (j in 0 until nodesPerCell) if (j != r) {
            val sj = nodeArray[base + j]
            v *= (s - sj) / (sr - sj)
        }
        return v
    }
}
