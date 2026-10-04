package verification

import numerics.SpecialFunctions
import org.junit.jupiter.api.Tag
import verification.WsieOracles.U
import verification.WsieOracles.bd
import java.math.BigDecimal
import java.math.MathContext
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * VALIDATION ITEM 3 (new-wsie r1, role 12): [SpecialFunctions] against [WsieOracles].
 *
 * Tolerances (validation plan, item 3): 16·u per elementary operation of the checked expression, an evaluation of
 * Gamma counting as one operation; transcendental results are never compared bitwise. Independence: the identity
 * checks (integers, half-integers, reflection, duplication, rational Beta moments, Gamma-free Mittag-Leffler series
 * for beta in {1/2, 1, 2}, erfc series) contain no Gamma on the reference side. The direct comparison of Gamma and of
 * Mittag-Leffler at beta in {1/3, 2/3} relies on the oracle's own Stirling series: same algorithm class, independent
 * only in precision (known limitation of the plan).
 */
@Tag("fast")
class WsieSpecialFunctionsOracleTest {

    private class Worst(val label: String) {
        var ratio = 0.0
        var dev = 0.0
        var where = ""
        val failures = mutableListOf<String>()
        fun add(dev: Double, tol: Double, where: String) {
            val r = if (dev.isNaN()) Double.POSITIVE_INFINITY else dev / tol
            if (r > ratio) { ratio = r; this.dev = dev; this.where = where }
            if (!(dev <= tol)) failures += "$where dev=$dev tol=$tol"
        }
        fun report() {
            println("V1-METRIC $label maxRelDev=$dev ratioToTol=$ratio at $where failures=${failures.size}")
            assertTrue(failures.isEmpty(), "$label: ${failures.size} cases above tolerance, first: ${failures.take(5)}")
        }
    }

    private fun ops(k: Int) = 16.0 * U * k

    @Test
    fun `Gamma at integers and half-integers`() {
        val w = Worst("Gamma-integer-halfinteger")
        for (n in 1..171) w.add(WsieOracles.relDev(SpecialFunctions.gamma(n.toDouble()), WsieOracles.gammaInteger(n)), ops(1), "Gamma($n)")
        for (n in 0..170) w.add(WsieOracles.relDev(SpecialFunctions.gamma(n + 0.5), WsieOracles.gammaHalfInteger(n)), ops(1), "Gamma($n+1/2)")
        w.report()
    }

    @Test
    fun `Gamma reflection Gamma(x) Gamma(1-x) = pi over sin(pi x)`() {
        val w = Worst("Gamma-reflection")
        // x in [1/2, 1): 1 − x is exact (Sterbenz), so both factors are evaluated at the exact complement.
        val xs = (0 until 64).map { 0.5 + it / 128.0 + 1.0 / 1024.0 } + listOf(0.5, 0.75, 1.0 - 1.0 / 1024, 1.0 - 1.0 / (1 shl 20), 0.6180339887498949)
        for (x in xs) {
            val code = SpecialFunctions.gamma(x) * SpecialFunctions.gamma(1.0 - x)
            w.add(WsieOracles.relDev(code, WsieOracles.reflection(x)), ops(3), "x=$x")
        }
        w.report()
    }

    @Test
    fun `Gamma duplication Gamma(x) Gamma(x+half) = 2^(1-2x) sqrt(pi) Gamma(2x)`() {
        val w = Worst("Gamma-duplication")
        // x on a 2^-7 lattice: x + 1/2 and 2x are exact; 2x <= 171 keeps Gamma(2x) finite.
        var x = 1.0 / 128
        while (x <= 85.0) {
            val lhs = SpecialFunctions.gamma(x) * SpecialFunctions.gamma(x + 0.5)
            val rhs = WsieOracles.duplicationFactor(x).multiply(bd(SpecialFunctions.gamma(2 * x)))
            // three Gamma evaluations + one product on the left; the right-hand factor is exact to 60 digits
            w.add(WsieOracles.relDev(lhs, rhs), ops(4), "x=$x")
            x += if (x < 2.0) 3.0 / 128 else 1.171875
        }
        w.report()
    }

    @Test
    fun `Gamma against the 60-digit Stirling oracle (same algorithm class)`() {
        val w = Worst("Gamma-vs-Stirling-oracle")
        val rnd = java.util.Random(20261004L)
        val xs = mutableListOf(1e-3, 0.1, 0.3333333333333333, 0.6666666666666666, 9.999999999999998, 10.0, 10.000000000000002, 170.6)
        repeat(120) { xs += 1e-3 + rnd.nextDouble() * 171.0 }
        repeat(40) { xs += rnd.nextDouble() * 12.0 + 1e-6 }
        for (x in xs) w.add(WsieOracles.relDev(SpecialFunctions.gamma(x), WsieOracles.gamma(bd(x))), ops(1), "x=$x")
        w.report()
    }

    @Test
    fun `Beta(1-alpha, j+1) equals the rational moment`() {
        val w = Worst("Beta-rational")
        for (alpha in doubleArrayOf(1.0 / 3.0, 0.5, 2.0 / 3.0, 0.9, 0.99)) for (j in 0..40) {
            val p = 1.0 - alpha // the Double actually passed; the oracle uses it exactly
            val code = SpecialFunctions.beta(p, j + 1.0)
            // three Gamma evaluations, a product, a quotient and the first-order sum correction
            w.add(WsieOracles.relDev(code, WsieOracles.betaPj(bd(p), j)), ops(6), "alpha=$alpha j=$j")
        }
        w.report()
    }

    private val betaGrid = listOf(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0)

    /** 16·u per relative rounding of a term (kappa, from the oracle) plus u per addition of the positive series. */
    private fun mlTol(ref: WsieOracles.MittagLefflerRef) = 16.0 * U * (ref.kappa + 1.0) + U * ref.terms

    @Test
    fun `Mittag-Leffler for beta in 1-2, 1, 2 against Gamma-free series`() {
        val w = Worst("ML-gamma-free")
        for (beta in WsieOracles.GammaFreeBeta.values()) for (z in betaGrid) {
            val ref = WsieOracles.mittagLefflerGammaFree(beta, z)
            val code = SpecialFunctions.mittagLeffler(beta.value, z)
            w.add(WsieOracles.relDev(code, ref.value), mlTol(ref), "beta=${beta.value} z=$z kappa=${ref.kappa}")
        }
        w.report()
    }

    @Test
    fun `Mittag-Leffler for beta in 1-3, 2-3 against the Stirling oracle (same algorithm class)`() {
        val w = Worst("ML-stirling-oracle")
        val cases = listOf(1.0 / 3.0 to doubleArrayOf(0.5, 2.0, 5.0, 8.5), 2.0 / 3.0 to doubleArrayOf(0.5, 2.0, 5.0, 10.0))
        for ((beta, zs) in cases) for (z in zs) {
            val ref = WsieOracles.mittagLeffler(beta, z)
            val code = SpecialFunctions.mittagLeffler(beta, z)
            w.add(WsieOracles.relDev(code, ref.value), mlTol(ref), "beta=$beta z=$z kappa=${ref.kappa}")
        }
        w.report()
    }

    @Test
    fun `erfc against the 60-digit series within the stated accuracy 1e-14`() {
        val w = Worst("erfc")
        val xs = mutableListOf(-3.0, -1.0, -0.999, -0.5, -1e-8, 1e-300, 1e-8, 0.5, 0.999, 1.0, 1.001, 2.0, 5.0, 10.0, 26.0, 26.5)
        var x = -2.0
        while (x <= 26.0) { xs += x; x += 0.0703125 }
        for (v in xs) w.add(WsieOracles.relDev(SpecialFunctions.erfc(v), WsieOracles.erfc(v)), 1e-14, "x=$v")
        w.report()
    }

    @Test
    fun `oracle self-checks - E_1 = exp, E_(1-2)(z) = exp(z^2) erfc(-z), Stirling vs half-integers`() {
        var worst = 0.0
        for (z in betaGrid) {
            val e1 = WsieOracles.mittagLefflerGammaFree(WsieOracles.GammaFreeBeta.ONE, z).value
            worst = maxOf(worst, e1.subtract(WsieOracles.exp(bd(z))).abs().divide(e1, MathContext(10)).toDouble())
            val eh = WsieOracles.mittagLefflerGammaFree(WsieOracles.GammaFreeBeta.HALF, z).value
            val viaErfc = WsieOracles.exp(bd(z).multiply(bd(z))).multiply(WsieOracles.erfc(-z))
            worst = maxOf(worst, eh.subtract(viaErfc).abs().divide(eh, MathContext(10)).toDouble())
        }
        for (n in intArrayOf(0, 3, 40, 170)) {
            val st = WsieOracles.gamma(BigDecimal(n).add(BigDecimal("0.5")))
            val hi = WsieOracles.gammaHalfInteger(n)
            worst = maxOf(worst, st.subtract(hi).abs().divide(hi, MathContext(10)).toDouble())
        }
        println("V1-METRIC oracle-self-check special maxRel=$worst")
        assertTrue(worst < 1e-45, "oracle disagreement $worst")
    }
}
