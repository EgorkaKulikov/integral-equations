package verification

import java.math.BigDecimal
import java.math.BigInteger
import java.math.MathContext
import java.math.RoundingMode

/**
 * INDEPENDENT ORACLES for the weakly singular extension (new-wsie, round r1, validation items 1, 3, 4).
 *
 * Independence follows [ReferenceNystromSolver]: the file imports only `java.math`, `java.io` and
 * `kotlin.test`; nothing from `src/main`, `numerics.*` or `splines.*` is used. Every `Double` input is
 * converted to `BigDecimal` exactly, so an oracle evaluates the mathematical function at the very number
 * the code under test receives; the only error left in a comparison is that of the code under test.
 *
 * Arithmetic is decimal with [DIGITS] significant digits plus guard digits inside every function.
 * Transcendental functions are summed from their Taylor/atanh series after argument reduction; pi comes
 * from Machin's formula. The moments use no Gamma function: B(p, j + 1) = j!/prod_{k=0}^{j}(k + p), and
 * |t - s|^(-alpha) s^j is integrated through a binomial expansion around t.
 *
 * Known limitation (validation plan, item 3): there is no independent oracle for Gamma at arbitrary
 * arguments. Gamma is checked through identities whose right-hand sides contain no Gamma (reflection,
 * duplication, half-integers, integers) and through the rational Beta moments; Mittag-Leffler only for
 * beta in {1/2, 1, 2}, where Gamma(beta k + 1) is a factorial or a half-integer value.
 */
internal object WsieOracles {
    const val DIGITS: Int = 60
    val MC: MathContext = MathContext(DIGITS, RoundingMode.HALF_EVEN)

    /** Unit roundoff of binary64, u = 2^-53. */
    const val U: Double = 1.1102230246251565e-16

    private val TWO = BigDecimal(2)
    private val HI = MathContext(DIGITS + 40, RoundingMode.HALF_EVEN)
    private val ln2Cache = HashMap<Int, BigDecimal>()

    val PI_HI: BigDecimal by lazy { pi(HI) }
    val SQRT_PI_HI: BigDecimal by lazy { PI_HI.sqrt(HI) }

    fun bd(x: Double): BigDecimal = BigDecimal(x)

    /** |code - ref| / |ref| for ref != 0, evaluated in BigDecimal; NaN if code is not finite. */
    fun relDev(code: Double, ref: BigDecimal): Double =
        if (!code.isFinite()) Double.NaN
        else bd(code).subtract(ref).abs().divide(ref.abs(), MathContext(20)).toDouble()

    /** |code - ref|; NaN if code is not finite. */
    fun absDev(code: Double, ref: BigDecimal): Double =
        if (!code.isFinite()) Double.NaN else bd(code).subtract(ref).abs().toDouble()

    private fun guard(mc: MathContext, extra: Int = 15) = MathContext(mc.precision + extra, RoundingMode.HALF_EVEN)

    /** Absolute stopping threshold of a series whose sum is of order one. */
    private fun eps(mc: MathContext): BigDecimal = BigDecimal.ONE.movePointLeft(mc.precision + 3)

    /** pi = 16 atan(1/5) - 4 atan(1/239) (Machin). */
    fun pi(mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        return atanInv(5, w).multiply(BigDecimal(16)).subtract(atanInv(239, w).multiply(BigDecimal(4))).round(mc)
    }

    private fun atanInv(q: Int, w: MathContext): BigDecimal {
        val x = BigDecimal.ONE.divide(BigDecimal(q), w)
        val x2 = x.multiply(x, w)
        var power = x
        var sum = BigDecimal.ZERO
        var k = 0
        val e = eps(w)
        while (power.abs() > e) {
            val term = power.divide(BigDecimal(2 * k + 1), w)
            sum = if (k % 2 == 0) sum.add(term, w) else sum.subtract(term, w)
            power = power.multiply(x2, w)
            k++
        }
        return sum
    }

    /** e^x: x/2^s with |x/2^s| <= 1/64, Taylor series, s squarings (guard digits cover the 2^s growth). */
    fun exp(x: BigDecimal, mc: MathContext = MC): BigDecimal {
        var s = 0
        var r = x
        val bound = BigDecimal("0.015625")
        while (r.abs() > bound) {
            r = r.divide(TWO)
            s++
        }
        val w = guard(mc, 16 + s / 3)
        val e = eps(w)
        var term = BigDecimal.ONE
        var sum = BigDecimal.ONE
        var k = 1
        while (term.abs() > e) {
            term = term.multiply(r, w).divide(BigDecimal(k), w)
            sum = sum.add(term, w)
            k++
        }
        repeat(s) { sum = sum.multiply(sum, w) }
        return sum.round(mc)
    }

    /** ln x, x > 0 (in the Double range): x = y 2^k, y in [1, 2), ln y = 2 atanh((y - 1)/(y + 1)). */
    fun ln(x: BigDecimal, mc: MathContext = MC): BigDecimal {
        require(x.signum() > 0) { "ln of a non-positive number $x" }
        val w = guard(mc)
        val k = Math.getExponent(x.toDouble())
        val scale = BigDecimal(BigInteger.TWO.pow(kotlin.math.abs(k)))
        val y = if (k >= 0) x.divide(scale, w) else x.multiply(scale, w)
        val lnY = TWO.multiply(atanh(y.subtract(BigDecimal.ONE).divide(y.add(BigDecimal.ONE), w), w), w)
        return lnY.add(ln2(w).multiply(BigDecimal(k), w), w).round(mc)
    }

    @Synchronized
    private fun ln2(w: MathContext): BigDecimal = ln2Cache.getOrPut(w.precision) {
        TWO.multiply(atanh(BigDecimal.ONE.divide(BigDecimal(3), w), w), w)
    }

    private fun atanh(u: BigDecimal, w: MathContext): BigDecimal {
        val u2 = u.multiply(u, w)
        var power = u
        var sum = BigDecimal.ZERO
        var k = 0
        val e = eps(w)
        while (power.abs() > e) {
            sum = sum.add(power.divide(BigDecimal(2 * k + 1), w), w)
            power = power.multiply(u2, w)
            k++
        }
        return sum
    }

    /** x^p for x >= 0 and p > 0 (0^p = 0). */
    fun pow(x: BigDecimal, p: BigDecimal, mc: MathContext = MC): BigDecimal =
        if (x.signum() == 0) BigDecimal.ZERO else exp(p.multiply(ln(x, guard(mc)), guard(mc)), mc)

    /** sin x by its Taylor series; intended for |x| <= 4. */
    fun sin(x: BigDecimal, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        val x2 = x.multiply(x, w)
        var term = x
        var sum = x
        var k = 1
        val e = eps(w)
        while (term.abs() > e) {
            term = term.multiply(x2, w).divide(BigDecimal((2 * k) * (2 * k + 1)), w).negate()
            sum = sum.add(term, w)
            k++
        }
        return sum.round(mc)
    }

    fun factorial(n: Int): BigInteger {
        var f = BigInteger.ONE
        for (k in 2..n) f = f.multiply(BigInteger.valueOf(k.toLong()))
        return f
    }

    private fun binomial(n: Int, k: Int): BigInteger = factorial(n).divide(factorial(k).multiply(factorial(n - k)))

    /** B(p, j + 1) = j!/prod_{k=0}^{j}(k + p), p > 0: rational in p, no Gamma function involved. */
    fun betaPj(p: BigDecimal, j: Int, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        var den = BigDecimal.ONE
        for (k in 0..j) den = den.multiply(p.add(BigDecimal(k)), w)
        return BigDecimal(factorial(j)).divide(den, w).round(mc)
    }

    /** B(1 - alpha, j + 1) = ∫_0^1 (1 - s)^(-alpha) s^j ds with the exact binary value of alpha. */
    fun betaMoment(alpha: Double, j: Int, mc: MathContext = MC): BigDecimal =
        betaPj(BigDecimal.ONE.subtract(bd(alpha)), j, mc)

    /**
     * ∫_{c0}^{c1} |t - s|^(-alpha) s^j ds for any finite t (inside, at an end of, or outside [c0, c1]).
     * s^j = Σ_k C(j, k) t^(j-k) (s - t)^k, and each (s - t)^k |s - t|^(-alpha) is integrated in closed form on
     * the parts of the cell to the right and to the left of t. The alternating binomial sum cancels at most
     * log10(Σ_k C(j,k)|t|^(j-k)(|c|+|t|)^k) digits, far fewer than the 40 guard digits used here.
     */
    fun momentAbs(alpha: Double, c0: Double, c1: Double, t: Double, j: Int, mc: MathContext = MC): BigDecimal {
        val w = HI
        val one = BigDecimal.ONE
        val q = one.subtract(bd(alpha)) // 1 - alpha
        val tt = bd(t)
        val lo = bd(c0)
        val hi = bd(c1)
        // F(x, k) = x^(k + 1 - alpha)/(k + 1 - alpha) for x >= 0
        fun primitiveParts(x: BigDecimal): Array<BigDecimal> {
            val base = pow(x, q, w)
            return Array(j + 1) { k -> base.multiply(x.pow(k, w), w).divide(q.add(BigDecimal(k)), w) }
        }
        val right = Array(j + 1) { BigDecimal.ZERO }
        val rLo = lo.max(tt)
        if (hi > rLo) {
            val fHi = primitiveParts(hi.subtract(tt))
            val fLo = primitiveParts(rLo.subtract(tt))
            for (k in 0..j) right[k] = fHi[k].subtract(fLo[k], w)
        }
        val left = Array(j + 1) { BigDecimal.ZERO }
        val lHi = hi.min(tt)
        if (lHi > lo) {
            val fFar = primitiveParts(tt.subtract(lo))
            val fNear = primitiveParts(tt.subtract(lHi))
            for (k in 0..j) left[k] = fFar[k].subtract(fNear[k], w)
        }
        var sum = BigDecimal.ZERO
        for (k in 0..j) {
            val part = if (k % 2 == 0) right[k].add(left[k], w) else right[k].subtract(left[k], w)
            sum = sum.add(BigDecimal(binomial(j, k)).multiply(tt.pow(j - k, w), w).multiply(part, w), w)
        }
        return sum.round(mc)
    }

    /** Volterra moment ∫_0^t (t - s)^(-alpha) s^j ds = t^(j + 1 - alpha) B(1 - alpha, j + 1), t > 0. */
    fun volterraMoment(alpha: Double, t: Double, j: Int, mc: MathContext = MC): BigDecimal {
        val q = BigDecimal.ONE.subtract(bd(alpha))
        return pow(bd(t), q.add(BigDecimal(j)), guard(mc)).multiply(betaPj(q, j, guard(mc)), guard(mc)).round(mc)
    }

    /** Gamma(n) = (n - 1)! for integer n >= 1. */
    fun gammaInteger(n: Int): BigDecimal = BigDecimal(factorial(n - 1))

    /** Gamma(n + 1/2) = sqrt(pi) (2n)!/(4^n n!), n >= 0. */
    fun gammaHalfInteger(n: Int, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        val ratio = BigDecimal(factorial(2 * n)).divide(BigDecimal(BigInteger.valueOf(4).pow(n).multiply(factorial(n))), w)
        return SQRT_PI_HI.multiply(ratio, w).round(mc)
    }

    /** Right-hand side of the reflection formula, pi/sin(pi x), 0 < x < 1. */
    fun reflection(x: Double, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        return PI_HI.divide(sin(PI_HI.multiply(bd(x), w), w), w).round(mc)
    }

    /** Factor of the duplication formula Gamma(x) Gamma(x + 1/2) = 2^(1 - 2x) sqrt(pi) Gamma(2x). */
    fun duplicationFactor(x: Double, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        val e = BigDecimal.ONE.subtract(TWO.multiply(bd(x)))
        return exp(e.multiply(ln(TWO, w), w), w).multiply(SQRT_PI_HI, w).round(mc)
    }

    /**
     * Mittag-Leffler series E_beta(z) = Σ z^k/Gamma(beta k + 1) for beta in {1/2, 1, 2}, where every Gamma value
     * is a factorial or a half-integer value: t_k = t_{k-1} z/k (beta = 1), t_k = t_{k-1} z/((2k)(2k - 1))
     * (beta = 2), t_k = t_{k-2} z^2/(k/2) with t_1 = 2z/sqrt(pi) (beta = 1/2).
     *
     * Also returns kappa = Σ t_k A_k / Σ t_k with A_k = k |ln z| + |ln Gamma(beta k + 1)|: a code that forms each
     * term as exp(k ln z - lnGamma(beta k + 1)) commits an absolute error of order u A_k in the exponent, hence a
     * relative error u A_k in t_k; kappa is the resulting first-order condition number of that formula.
     */
    fun mittagLefflerGammaFree(beta: GammaFreeBeta, z: Double, mc: MathContext = MC): MittagLefflerRef {
        val w = guard(mc)
        val zz = bd(z)
        if (z == 0.0) return MittagLefflerRef(BigDecimal.ONE, 0.0, 1)
        val lnZ = kotlin.math.abs(Math.log(z))
        var sum = BigDecimal.ONE
        var weighted = 0.0
        var prev2 = BigDecimal.ONE // t_{k-2}
        var prev = BigDecimal.ONE // t_{k-1}
        var lnGammaPrev2 = 0.0
        var lnGammaPrev = 0.0
        var k = 1
        val stop = BigDecimal.ONE.movePointLeft(mc.precision + 5)
        var decreasing = false
        while (true) {
            val term: BigDecimal
            val lnGamma: Double
            when (beta) {
                GammaFreeBeta.ONE -> { term = prev.multiply(zz, w).divide(BigDecimal(k), w); lnGamma = lnGammaPrev + Math.log(k.toDouble()) }
                GammaFreeBeta.TWO -> {
                    term = prev.multiply(zz, w).divide(BigDecimal(2L * k * (2 * k - 1)), w)
                    lnGamma = lnGammaPrev + Math.log(2.0 * k) + Math.log(2.0 * k - 1)
                }
                GammaFreeBeta.HALF -> if (k == 1) {
                    term = zz.multiply(TWO, w).divide(SQRT_PI_HI, w); lnGamma = Math.log(Math.sqrt(Math.PI) / 2)
                } else {
                    term = prev2.multiply(zz.multiply(zz, w), w).multiply(TWO, w).divide(BigDecimal(k), w)
                    lnGamma = lnGammaPrev2 + Math.log(k / 2.0)
                }
            }
            sum = sum.add(term, w)
            weighted += term.toDouble() * (k * lnZ + kotlin.math.abs(lnGamma))
            if (term < prev) decreasing = true
            prev2 = prev; prev = term
            lnGammaPrev2 = lnGammaPrev; lnGammaPrev = lnGamma
            if (decreasing && term < stop.multiply(sum)) break
            k++
        }
        return MittagLefflerRef(sum.round(mc), weighted / sum.toDouble(), k + 1)
    }

    /** The three admissible beta values with Gamma-free series. */
    enum class GammaFreeBeta(val value: Double) { HALF(0.5), ONE(1.0), TWO(2.0) }

    /**
     * erfc(x) from the positive-term series erf(x) = (2/sqrt(pi)) e^(-x^2) Σ_n 2^n x^(2n+1)/(1·3···(2n+1)),
     * erfc = 1 - erf for x > 0 and 1 + erf(|x|) for x <= 0. The subtraction cancels about x^2/ln 10 digits;
     * the working precision is raised by that amount, so the result keeps [DIGITS] digits.
     */
    fun erfc(x: Double, mc: MathContext = MC): BigDecimal {
        if (x == 0.0) return BigDecimal.ONE
        val ax = kotlin.math.abs(x)
        val w = MathContext(mc.precision + 25 + (ax * ax / 2.302585).toInt(), RoundingMode.HALF_EVEN)
        val xx = bd(ax)
        val x2 = xx.multiply(xx, w)
        var term = xx
        var sum = xx
        var n = 1
        val stop = BigDecimal.ONE.movePointLeft(w.precision + 2)
        while (true) {
            term = term.multiply(x2, w).multiply(TWO, w).divide(BigDecimal(2 * n + 1), w)
            sum = sum.add(term, w)
            if (BigDecimal(2 * n + 1) > x2.multiply(TWO) && term < stop.multiply(sum)) break
            n++
        }
        val sqrtPi = pi(w).sqrt(w)
        val erf = TWO.divide(sqrtPi, w).multiply(exp(x2.negate(), w), w).multiply(sum, w)
        return (if (x > 0.0) BigDecimal.ONE.subtract(erf, w) else BigDecimal.ONE.add(erf, w)).round(mc)
    }

    /** Bernoulli numbers B_0..B_{2K} from Σ_{k=0}^{m} C(m + 1, k) B_k = 0, in 160-digit arithmetic. */
    private val BERNOULLI: Array<BigDecimal> by lazy {
        val w = MathContext(160, RoundingMode.HALF_EVEN)
        val b = Array(2 * STIRLING_TERMS + 1) { BigDecimal.ZERO }
        b[0] = BigDecimal.ONE
        for (m in 1..2 * STIRLING_TERMS) {
            var s = BigDecimal.ZERO
            for (k in 0 until m) s = s.add(BigDecimal(binomial(m + 1, k)).multiply(b[k], w), w)
            b[m] = s.negate().divide(BigDecimal(m + 1), w)
        }
        b
    }
    private const val STIRLING_TERMS = 30
    private const val STIRLING_SHIFT = 60

    /**
     * ln Gamma(x), x > 0, by the Stirling series with [STIRLING_TERMS] Bernoulli terms after the shift
     * y = x + N >= [STIRLING_SHIFT]; the truncation error is below 1e-76. The ALGORITHM CLASS coincides with the
     * code under test (validation plan, item 3); only the precision is independent.
     */
    fun lnGamma(x: BigDecimal, mc: MathContext = MC): BigDecimal {
        val w = guard(mc)
        var y = x
        var shiftProduct = BigDecimal.ONE
        while (y < BigDecimal(STIRLING_SHIFT)) {
            shiftProduct = shiftProduct.multiply(y, w)
            y = y.add(BigDecimal.ONE)
        }
        val half = BigDecimal("0.5")
        var s = y.subtract(half).multiply(ln(y, w), w).subtract(y, w).add(half.multiply(ln(TWO.multiply(PI_HI, w), w), w), w)
        val y2 = y.multiply(y, w)
        var yPow = y
        for (k in 1..STIRLING_TERMS) {
            s = s.add(BERNOULLI[2 * k].divide(BigDecimal(2L * k * (2 * k - 1)).multiply(yPow, w), w), w)
            yPow = yPow.multiply(y2, w)
        }
        return s.subtract(ln(shiftProduct, w), w).round(mc)
    }

    fun gamma(x: BigDecimal, mc: MathContext = MC): BigDecimal = exp(lnGamma(x, guard(mc)), mc)

    /** Reference value of E_beta(z), the condition number kappa of the exp-of-difference term formula, the term count. */
    data class MittagLefflerRef(val value: BigDecimal, val kappa: Double, val terms: Int)

    /**
     * E_beta(z) = Σ z^k/Gamma(beta k + 1) at the exact binary beta and z (beta = 1/3 in Double is not 1/3, and
     * dE/dbeta is large: the oracle must not use a rational recurrence in beta). Gamma from [lnGamma], 40 digits.
     * kappa = Σ t_k A_k/Σ t_k, A_k = k|ln z| + |ln Gamma(beta k + 1)| + (beta k + 1): the last summand bounds the
     * cancellation inside (x - 1/2) ln x - x of a Stirling-based lnGamma (see [mittagLefflerGammaFree]).
     */
    fun mittagLeffler(beta: Double, z: Double): MittagLefflerRef {
        val w = MathContext(40, RoundingMode.HALF_EVEN)
        if (z == 0.0) return MittagLefflerRef(BigDecimal.ONE, 0.0, 1)
        val b = bd(beta)
        val lnZ = ln(bd(z), w)
        var sum = BigDecimal.ONE
        var weighted = BigDecimal.ZERO
        var prev = BigDecimal.ONE
        var decreasing = false
        var k = 1
        val stop = BigDecimal.ONE.movePointLeft(25)
        while (true) {
            val lg = lnGamma(b.multiply(BigDecimal(k)).add(BigDecimal.ONE), w)
            val term = exp(lnZ.multiply(BigDecimal(k)).subtract(lg, w), w)
            sum = sum.add(term, w)
            val a = k * kotlin.math.abs(lnZ.toDouble()) + kotlin.math.abs(lg.toDouble()) + beta * k + 1.0
            weighted = weighted.add(term.multiply(BigDecimal(a), w), w)
            if (term < prev) decreasing = true
            prev = term
            if (decreasing && term < stop.multiply(sum)) break
            k++
        }
        return MittagLefflerRef(sum, weighted.divide(sum, w).toDouble(), k + 1)
    }

    /**
     * Cox–de Boor recursion for the quadratic B-splines on the knots y_i, i = -2..n+2 (array index i + 2), at y in the
     * cell [y_k, y_{k+1}], 0 <= k <= n - 1: returns (B_{k-2}, B_{k-1}, B_k) and their derivatives d/dy. The
     * denominators y_{k+1} - y_{k-1} and y_{k+2} - y_k are positive for every cell of a strictly increasing interior.
     */
    fun quadraticBSplines(y: Array<BigDecimal>, k: Int, at: BigDecimal, mc: MathContext): Pair<Array<BigDecimal>, Array<BigDecimal>> {
        fun knot(i: Int) = y[i + 2]
        val yk = knot(k); val yk1 = knot(k + 1)
        val b1Right = at.subtract(yk).divide(yk1.subtract(yk), mc) // B_{k,1}
        val b1Left = yk1.subtract(at).divide(yk1.subtract(yk), mc) // B_{k-1,1}
        val dLeft = yk1.subtract(knot(k - 1)) // y_{k+1} - y_{k-1}
        val dRight = knot(k + 2).subtract(yk) // y_{k+2} - y_k
        val values = arrayOf(
            yk1.subtract(at).divide(dLeft, mc).multiply(b1Left, mc),
            at.subtract(knot(k - 1)).divide(dLeft, mc).multiply(b1Left, mc)
                .add(knot(k + 2).subtract(at).divide(dRight, mc).multiply(b1Right, mc), mc),
            at.subtract(yk).divide(dRight, mc).multiply(b1Right, mc),
        )
        val derivs = arrayOf(
            TWO.multiply(b1Left).divide(dLeft, mc).negate(),
            TWO.multiply(b1Left.divide(dLeft, mc).subtract(b1Right.divide(dRight, mc), mc)),
            TWO.multiply(b1Right).divide(dRight, mc),
        )
        return values to derivs
    }

    // ==================== validation items 6 and 9 (round V2) ====================

    /**
     * Values (B_{k-2}, B_{k-1}, B_k) of [quadraticBSplines] in binary64 at `at` in the cell [y_k, y_{k+1}] (same
     * recursion, same knot layout y[i + 2] = y_i); used for the bulk assembly of the reference scheme (item 9).
     */
    fun quadraticBSplinesD(y: DoubleArray, k: Int, at: Double): DoubleArray {
        val ym = y[k + 1]; val yk = y[k + 2]; val yk1 = y[k + 3]; val yk2 = y[k + 4]
        val b1Right = (at - yk) / (yk1 - yk)
        val b1Left = (yk1 - at) / (yk1 - yk)
        val dLeft = yk1 - ym
        val dRight = yk2 - yk
        return doubleArrayOf(
            (yk1 - at) / dLeft * b1Left,
            (at - ym) / dLeft * b1Left + (yk2 - at) / dRight * b1Right,
            (at - yk) / dRight * b1Right,
        )
    }

    /** Gaussian elimination with partial pivoting in binary64 (own LU, item 9): solves a x = rhs; inputs are copied. */
    fun solveDense(a: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
        val m = rhs.size
        val lu = Array(m) { a[it].copyOf() }
        val x = rhs.copyOf()
        for (c in 0 until m) {
            var p = c
            for (r in c + 1 until m) if (Math.abs(lu[r][c]) > Math.abs(lu[p][c])) p = r
            check(lu[p][c] != 0.0) { "solveDense: singular matrix at column $c" }
            if (p != c) { val t = lu[p]; lu[p] = lu[c]; lu[c] = t; val tx = x[p]; x[p] = x[c]; x[c] = tx }
            for (r in c + 1 until m) {
                val f = lu[r][c] / lu[c][c]
                if (f != 0.0) { for (q in c until m) lu[r][q] -= f * lu[c][q]; x[r] -= f * x[c] }
            }
        }
        for (r in m - 1 downTo 0) {
            var s = x[r]
            for (q in r + 1 until m) s -= lu[r][q] * x[q]
            x[r] = s / lu[r][r]
        }
        return x
    }

    /** Gaussian elimination with partial pivoting in BigDecimal (item 6): solves a x = rhs; inputs are copied. */
    fun solveBig(a: Array<Array<BigDecimal>>, rhs: Array<BigDecimal>, mc: MathContext): Array<BigDecimal> {
        val m = rhs.size
        val lu = Array(m) { a[it].copyOf() }
        val x = rhs.copyOf()
        for (c in 0 until m) {
            var p = c
            for (r in c + 1 until m) if (lu[r][c].abs() > lu[p][c].abs()) p = r
            check(lu[p][c].signum() != 0) { "solveBig: singular matrix at column $c" }
            if (p != c) { val t = lu[p]; lu[p] = lu[c]; lu[c] = t; val tx = x[p]; x[p] = x[c]; x[c] = tx }
            for (r in c + 1 until m) {
                val f = lu[r][c].divide(lu[c][c], mc)
                if (f.signum() != 0) {
                    for (q in c until m) lu[r][q] = lu[r][q].subtract(f.multiply(lu[c][q], mc), mc)
                    x[r] = x[r].subtract(f.multiply(x[c], mc), mc)
                }
            }
        }
        for (r in m - 1 downTo 0) {
            var s = x[r]
            for (q in r + 1 until m) s = s.subtract(lu[r][q].multiply(x[q], mc), mc)
            x[r] = s.divide(lu[r][r], mc)
        }
        return x
    }

    /**
     * Tanh-sinh rule on [-1, 1] with step [h], |kh| <= 4 (weights below 1e-36 beyond): x_k = tanh(u_k),
     * u_k = (pi/2) sinh(kh), w_k = h (pi/2) cosh(kh)/cosh^2(u_k). The distance to the nearer end,
     * 1 - |x_k| = 2/(1 + e^(2|u_k|)), is kept separately, so nodes next to an end carry no cancellation.
     */
    class TanhSinh(h: Double) {
        private val w: DoubleArray
        private val gap: DoubleArray
        private val neg: BooleanArray

        init {
            val kMax = Math.ceil(4.0 / h).toInt()
            val m = 2 * kMax + 1
            w = DoubleArray(m); gap = DoubleArray(m); neg = BooleanArray(m)
            for (i in 0 until m) {
                val t = (i - kMax) * h
                val u = 0.5 * Math.PI * Math.sinh(t)
                val ch = Math.cosh(u)
                w[i] = h * 0.5 * Math.PI * Math.cosh(t) / (ch * ch)
                gap[i] = 2.0 / (1.0 + Math.exp(2.0 * Math.abs(u)))
                neg[i] = t < 0
            }
        }

        /** Calls [action] with (node s, s - lo, hi - s, weight) of the rule mapped to [lo, hi]. */
        fun forEach(lo: Double, hi: Double, action: (Double, Double, Double, Double) -> Unit) {
            val half = 0.5 * (hi - lo)
            val len = hi - lo
            for (i in w.indices) {
                val d = half * gap[i]
                if (neg[i]) action(lo + d, d, len - d, half * w[i]) else action(hi - d, len - d, d, half * w[i])
            }
        }
    }

    /** Gauss-Legendre nodes on [-1, 1] by Newton on P_m, ascending (the probe control set uses m = 8). */
    fun gaussLegendreNodes(m: Int): DoubleArray = DoubleArray(m) { i ->
        var x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5))
        repeat(60) {
            var p0 = 1.0
            var p1 = x
            for (k in 2..m) { val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2 }
            x -= p1 / (m * (x * p1 - p0) / (x * x - 1.0))
        }
        x
    }.sortedArray()

    /**
     * Item 6 oracle: quadratic B-splines in g(t) on the knots g(x_i) with triple end knots (own Cox-de Boor,
     * [quadraticBSplines]), g(t) = t^beta (beta = null: g = id, the space B), and the weights of theta_j from the
     * local biorthogonality system sum_p w_p omega_i(points_p) = delta_ij, i in indices, solved by [solveBig].
     */
    class ThetaOracle(private val x: DoubleArray, private val beta: BigDecimal?, private val mc: MathContext) {
        private val n = x.size - 1
        private val y: Array<BigDecimal> = Array(n + 5) { i -> g(bd(x[(i - 2).coerceIn(0, n)])) }

        fun g(t: BigDecimal): BigDecimal = if (beta == null) t else WsieOracles.pow(t, beta, mc)

        /** Cell k with x_k <= t (< x_{k+1} unless k = n - 1) and (omega_{k-2}, omega_{k-1}, omega_k)(t). */
        fun omegas(t: BigDecimal): Pair<Int, Array<BigDecimal>> {
            var k = 0
            while (k < n - 1 && t >= bd(x[k + 1])) k++
            return k to quadraticBSplines(y, k, g(t), mc).first
        }

        fun weights(j: Int, points: Array<BigDecimal>, indices: IntArray): Array<BigDecimal> {
            val m = points.size
            val cols = points.map { omegas(it) }
            val a = Array(m) { r ->
                Array(m) { c ->
                    val (k, v) = cols[c]
                    val i = indices[r]
                    if (i < k - 2 || i > k) BigDecimal.ZERO else v[i - k + 2]
                }
            }
            val rhs = Array(m) { if (indices[it] == j) BigDecimal.ONE else BigDecimal.ZERO }
            return solveBig(a, rhs, mc)
        }

        /** Sampling point of [l, r]: the arithmetic midpoint, or (tau) g^{-1}((g(l) + g(r))/2) with g = t^beta. */
        fun midpoint(l: Double, r: Double, tau: Boolean): BigDecimal {
            val lb = bd(l); val rb = bd(r)
            if (!tau || beta == null) return lb.add(rb).divide(BigDecimal(2), mc)
            val half = g(lb).add(g(rb)).divide(BigDecimal(2), mc)
            return WsieOracles.pow(half, BigDecimal.ONE.divide(beta, mc), mc)
        }

        /**
         * theta_j as (points, weights): j = -2 and j = n - 1 are the values at x_0 and x_n; j = -1 and j = n - 2 use
         * the 3 points (end knot, sampling point, next knot); interior j the 5 points x_j, three sampling points, x_{j+3}.
         */
        fun theta(j: Int, tau: Boolean): Pair<Array<BigDecimal>, Array<BigDecimal>> {
            if (j == -2) return arrayOf(bd(x[0])) to arrayOf(BigDecimal.ONE)
            if (j == n - 1) return arrayOf(bd(x[n])) to arrayOf(BigDecimal.ONE)
            val (pts, idx) = when (j) {
                -1 -> arrayOf(bd(x[0]), midpoint(x[0], x[1], tau), bd(x[1])) to intArrayOf(-2, -1, 0)
                n - 2 -> arrayOf(bd(x[n - 1]), midpoint(x[n - 1], x[n], tau), bd(x[n])) to intArrayOf(n - 3, n - 2, n - 1)
                else -> arrayOf(
                    bd(x[j]), midpoint(x[j], x[j + 1], tau), midpoint(x[j + 1], x[j + 2], tau),
                    midpoint(x[j + 2], x[j + 3], tau), bd(x[j + 3]),
                ) to IntArray(5) { j - 2 + it }
            }
            return pts to weights(j, pts, idx)
        }

        /** max over ALL i = -2..n-1 of |sum_p w_p omega_i(points_p) - delta_ij| (not only the local indices). */
        fun biorthogonalityDefect(j: Int, points: Array<BigDecimal>, w: Array<BigDecimal>): Double {
            val acc = Array(n + 2) { BigDecimal.ZERO }
            for (p in points.indices) {
                val (k, v) = omegas(points[p])
                for (q in 0..2) acc[k + q] = acc[k + q].add(w[p].multiply(v[q], mc), mc)
            }
            acc[j + 2] = acc[j + 2].subtract(BigDecimal.ONE, mc)
            return acc.maxOf { it.abs().toDouble() }
        }

        /** s(t) = sum_i c_{i+2} omega_i(t) evaluated in BigDecimal at the exact binary64 value t. */
        fun spline(c: DoubleArray, t: Double): Double {
            val (k, v) = omegas(bd(t))
            var s = BigDecimal.ZERO
            for (q in 0..2) s = s.add(bd(c[k + q]).multiply(v[q], mc), mc)
            return s.toDouble()
        }
    }
}
