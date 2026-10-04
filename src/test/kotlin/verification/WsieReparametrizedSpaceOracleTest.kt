package verification

import org.junit.jupiter.api.Tag
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.Reparametrization
import verification.WsieOracles.bd
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * VALIDATION ITEM 4 (new-wsie r1, role 12): the minimal splines of the reparametrized system (1, g, g²),
 * g(t) = t^beta, against omega_j = B_j ∘ g with B_j the quadratic B-splines on the knots g(x_j) (triple end knots),
 * evaluated by the oracle's own Cox–de Boor recursion in 60-digit arithmetic on the exact binary grid nodes.
 *
 * Tolerance (validation plan, item 4): absolute 1e-13 for omega_j in [0, 1]. Budget: a B-spline is Lipschitz in y
 * with constant <= 2/h_g. A code that rounds g(x) itself (relative 4u) would move omega by up to 8u·g/h_g, which is
 * not below 1e-13 on every grid here (g/h_g ≈ 1.9e2, 8u·g/h_g ≈ 1.7e-13 in the last cell of the uniform grid,
 * n = 64, beta = 1/3).
 * The reparametrized frame of the code works with local differences g(t) − g(c) (expm1/log1p, local coordinates of
 * CONTRIBUTING), whose relative error is a few u, so the expected deviation is O(u) and 1e-13 leaves a margin of
 * about 100 over it. For omega_j' the same tolerance is applied relative to the scale S(t) = Σ_j |omega_j'(t)| at
 * the point (omega_j' grows like g'(t) ~ t^(beta − 1) near 0, so an absolute bound is meaningless there).
 */
@Tag("fast")
class WsieReparametrizedSpaceOracleTest {

    private val mc = MathContext(60, RoundingMode.HALF_EVEN)
    private val tol = 1e-13

    private fun grids(n: Int): List<Pair<String, Grid>> = listOf(
        "uniform" to Grid.uniform(n),
        "power(r=2)" to Grid.power(n, r = 2.0),
        "power(r=3)" to Grid.power(n, r = 3.0),
        "symmetricPower(r=2)" to Grid.symmetricPower(n, r = 2.0),
    )

    @Test
    fun `omega_j and omega_j' of G = (1, t^beta, t^(2 beta)) equal B_j(g) and B_j'(g) g' (own Cox-de Boor)`() {
        var worstValue = 0.0
        var worstDeriv = 0.0
        var whereValue = ""
        var whereDeriv = ""
        var cases = 0
        val failures = mutableListOf<String>()
        for (beta in doubleArrayOf(1.0 / 3.0, 0.5, 2.0 / 3.0)) for (n in intArrayOf(8, 64)) for ((gridName, grid) in grids(n)) {
            val basis = MinimalSplineBasis(GeneratingSystem.reparametrized(Reparametrization.power(beta)), grid)
            val b = bd(beta)
            val y = Array(n + 5) { i -> WsieOracles.pow(bd(grid.x(i - 2)), b, mc) }
            val points = mutableListOf<Pair<Int, Double>>(0 to 1e-12, 0 to 1e-8)
            for (k in 0 until n) for (i in 1..7) {
                val t = grid.x(k) + (grid.x(k + 1) - grid.x(k)) * (i / 8.0)
                if (t > grid.x(k) && t < grid.x(k + 1)) points += k to t
            }
            for ((k, t) in points) {
                val tt = bd(t)
                val (values, derivs) = WsieOracles.quadraticBSplines(y, k, WsieOracles.pow(tt, b, mc), mc)
                val gPrime = b.multiply(WsieOracles.pow(tt, b.subtract(BigDecimal.ONE), mc), mc)
                var scale = BigDecimal.ZERO
                val refDerivs = Array(3) { derivs[it].multiply(gPrime, mc) }
                for (d in refDerivs) scale = scale.add(d.abs(), mc)
                val label = "beta=$beta n=$n $gridName k=$k t=$t"
                for (a in 0..2) {
                    val j = k - 2 + a
                    val dv = WsieOracles.absDev(basis.omega(j, t), values[a])
                    val dd = WsieOracles.absDev(basis.omegaDeriv(j, t), refDerivs[a]) / scale.toDouble()
                    cases++
                    if (!(dv <= worstValue)) { worstValue = dv; whereValue = "$label j=$j" }
                    if (!(dd <= worstDeriv)) { worstDeriv = dd; whereDeriv = "$label j=$j" }
                    if (!(dv <= tol)) failures += "value $label j=$j dev=$dv"
                    if (!(dd <= tol)) failures += "deriv $label j=$j relToScale=$dd"
                }
                // outside the support the minimal splines vanish
                for (j in intArrayOf(k - 3, k + 1)) if (j >= -2 && j <= n - 1) {
                    if (basis.omega(j, t) != 0.0) failures += "nonzero outside support $label j=$j"
                }
            }
        }
        println("V1-METRIC G-space omega maxAbsDev=$worstValue at $whereValue")
        println("V1-METRIC G-space omega' maxDevRelToScale=$worstDeriv at $whereDeriv cases=$cases")
        assertTrue(failures.isEmpty(), "${failures.size} cases above tolerance $tol, first: ${failures.take(5)}")
    }

    @Test
    fun `oracle self-check - Cox-de Boor values form a partition of unity and derivatives sum to zero`() {
        // n = 7 cells, knots y_{-2} = y_{-1} = y_0 = 0 < y_1 < ... < y_7 = y_8 = y_9 = 1
        val y = Array(12) { i -> bd(listOf(0.0, 0.0, 0.0, 0.1, 0.25, 0.3, 0.5, 0.55, 0.8, 1.0, 1.0, 1.0)[i]) }
        var worst = BigDecimal.ZERO
        for (k in 0..6) for (f in listOf("0.1", "0.5", "0.9")) {
            val at = y[k + 2].add(y[k + 3].subtract(y[k + 2]).multiply(BigDecimal(f)))
            val (v, d) = WsieOracles.quadraticBSplines(y, k, at, mc)
            worst = worst.max(v[0].add(v[1]).add(v[2]).subtract(BigDecimal.ONE).abs()).max(d[0].add(d[1]).add(d[2]).abs())
        }
        println("V1-METRIC oracle-self-check cox-de-boor maxDefect=$worst")
        assertTrue(worst < BigDecimal("1e-50"), "oracle defect $worst")
    }
}
