package verification

import numerics.AlgebraicSingularQuadrature
import numerics.GaussJacobi
import org.junit.jupiter.api.Tag
import verification.WsieOracles.U
import java.math.BigDecimal
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * VALIDATION ITEM 1 (new-wsie r1, role 12): moments of [GaussJacobi] and [AlgebraicSingularQuadrature] against
 * the Gamma-free oracles of [WsieOracles].
 *
 * Tolerance: relative 32·u·(j + 1) (validation plan, item 1). For f = s^j with j <= 2m − 1 the Gauss rule is exact,
 * so only rounding is left: m positive terms of one sign (sum error <= m·u, m <= 16), the evaluation of s^j at a node
 * carrying j relative roundings, and the node error amplified by j. For a point t outside a cell the rule is
 * Gauss–Legendre on pieces no longer than their distance to t: with 12 nodes the truncation error is below
 * (3 + √8)^(−24) ≈ 4e-19 (Bernstein ellipse through the singularity at relative distance 1), below u, so the same
 * tolerance applies there with the default nodesPerSub = 12 only.
 */
@Tag("fast")
class WsieQuadratureOracleTest {

    private val alphas = doubleArrayOf(1.0 / 3.0, 0.5, 2.0 / 3.0, 0.9)

    private fun tol(j: Int) = 32.0 * U * (j + 1)

    private class Worst(val label: String) {
        var ratio = 0.0
        var dev = 0.0
        var where = ""
        val failures = mutableListOf<String>()
        fun add(dev: Double, tol: Double, where: String) {
            if (dev.isNaN() || dev / tol > ratio) { ratio = if (dev.isNaN()) Double.POSITIVE_INFINITY else dev / tol; this.dev = dev; this.where = where }
            if (dev.isNaN() || dev > tol) failures += "$where dev=$dev tol=$tol"
        }
        fun report() {
            println("V1-METRIC $label maxRelDev=$dev ratioToTol=$ratio at $where failures=${failures.size}")
            assertTrue(failures.isEmpty(), "$label: ${failures.size} cases above tolerance, first: ${failures.take(5)}")
        }
    }

    @Test
    fun `Gauss-Jacobi moments on 0 1 equal rational Beta values`() {
        val w = Worst("GaussJacobi-moments")
        for (alpha in alphas) for (m in intArrayOf(1, 2, 3, 4, 8, 12, 16)) {
            // (a, b, shift): ∫_0^1 (1 − s)^a s^b s^j = B(b + j + 1, a + 1), rational when a is an integer or b is 0.
            val cases = listOf(
                Triple(-alpha, 0.0, "a=-alpha,b=0"), Triple(0.0, -alpha, "a=0,b=-alpha"),
                Triple(1.0, -alpha, "a=1,b=-alpha"), Triple(-alpha, 2.0, "a=-alpha,b=2"),
            )
            for ((a, b, name) in cases) {
                val rule = GaussJacobi(m, a, b)
                for (j in 0..(2 * m - 1)) {
                    val code = rule.integrate(0.0, 1.0) { s -> powInt(s, j) }
                    val ref = when (name) {
                        "a=-alpha,b=0" -> WsieOracles.betaMoment(alpha, j)
                        "a=-alpha,b=2" -> WsieOracles.betaMoment(alpha, j + 2)
                        // B(j + 1 − alpha, a + 1) = a!/∏_{k=0}^{a} (j + 1 − alpha + k)
                        else -> WsieOracles.betaPj(BigDecimal(j + 1).subtract(WsieOracles.bd(alpha)), a.toInt())
                    }
                    w.add(WsieOracles.relDev(code, ref), tol(j), "alpha=$alpha m=$m $name j=$j")
                }
            }
        }
        w.report()
    }

    @Test
    fun `Volterra moments of AlgebraicSingularQuadrature equal t^(j+1-alpha) B(1-alpha, j+1)`() {
        val w = Worst("ASQ-Volterra-moments")
        for (alpha in alphas) for (m in intArrayOf(2, 4, 12)) {
            val q = AlgebraicSingularQuadrature(alpha, m)
            // With several cells the cells left of t are integrated by Gauss–Legendre (not exact): m = 12 only.
            for (n in if (m == 12) intArrayOf(1, 8) else intArrayOf(1)) {
                val h = 1.0 / n
                for (t in doubleArrayOf(0.3 * h, h, 0.5, 0.77, 1.0)) {
                    val bp = (0..n).map { it * h }.filter { it < t }.toMutableList().apply { add(t) }.toDoubleArray()
                    for (j in 0..(2 * m - 1)) {
                        val code = q.integrate(bp, t) { s -> powInt(s, j) }
                        w.add(WsieOracles.relDev(code, WsieOracles.volterraMoment(alpha, t, j)), tol(j), "alpha=$alpha m=$m n=$n t=$t j=$j")
                    }
                }
            }
        }
        w.report()
    }

    @Test
    fun `two-sided moments of AlgebraicSingularQuadrature on one cell for t inside, at the ends and outside`() {
        val w = Worst("ASQ-two-sided-cell")
        val c0 = 0.375
        val c1 = 0.5
        val h = c1 - c0
        val ts = mutableListOf(c0, c1, c0 + 0.3 * h, 0.4375, c0 + 0.999 * h, c0 + 1e-9 * h)
        for (d in doubleArrayOf(1e-6, 1e-3, 0.1, 5.0)) { ts += c1 + d * h; ts += c0 - d * h }
        for (alpha in alphas) for (m in intArrayOf(2, 4, 12)) {
            val q = AlgebraicSingularQuadrature(alpha, m)
            for (t in ts) {
                val outside = t < c0 || t > c1
                if (outside && m != 12) continue // Gauss–Legendre pieces: exact to rounding only with 12 nodes
                for (j in 0..(2 * m - 1)) {
                    val code = q.integrate(doubleArrayOf(c0, c1), t) { s -> powInt(s, j) }
                    w.add(WsieOracles.relDev(code, WsieOracles.momentAbs(alpha, c0, c1, t, j)), tol(j), "alpha=$alpha m=$m t=$t j=$j")
                }
            }
        }
        w.report()
    }

    @Test
    fun `two-sided moments over a uniform partition of 0 1 equal the sum of cell oracles`() {
        val w = Worst("ASQ-two-sided-partition")
        val n = 8
        val bp = DoubleArray(n + 1) { it / n.toDouble() }
        for (alpha in alphas) {
            val q = AlgebraicSingularQuadrature(alpha)
            for (t in doubleArrayOf(0.0, 0.0625, 0.5, 0.53125 + 1e-7, 0.999, 1.0, 1.2, -0.05)) {
                for (j in 0..23) {
                    val code = q.integrate(bp, t) { s -> powInt(s, j) }
                    var ref = BigDecimal.ZERO
                    for (i in 0 until n) ref = ref.add(WsieOracles.momentAbs(alpha, bp[i], bp[i + 1], t, j))
                    // The m·n positive terms of the sum add one rounding per cell: n·u on top of the cell bound.
                    w.add(WsieOracles.relDev(code, ref), tol(j) + n * U, "alpha=$alpha t=$t j=$j")
                }
            }
        }
        w.report()
    }

    @Test
    fun `oracle self-check - two-sided moment at a cell end equals the Volterra Beta moment`() {
        // Independent formulas: binomial expansion with 100-digit arithmetic vs t^(j+1-alpha) B(1-alpha, j+1).
        var worst = 0.0
        for (alpha in alphas) for (t in doubleArrayOf(0.3, 1.0)) for (j in 0..23) {
            val a = WsieOracles.momentAbs(alpha, 0.0, t, t, j)
            val b = WsieOracles.volterraMoment(alpha, t, j)
            worst = maxOf(worst, a.subtract(b).abs().divide(b, WsieOracles.MC).toDouble())
        }
        println("V1-METRIC oracle-self-check momentAbs-vs-Beta maxRel=$worst")
        assertTrue(worst < 1e-40, "oracle disagreement $worst")
    }

    private fun powInt(s: Double, j: Int): Double {
        var r = 1.0
        repeat(j) { r *= s }
        return r
    }
}
