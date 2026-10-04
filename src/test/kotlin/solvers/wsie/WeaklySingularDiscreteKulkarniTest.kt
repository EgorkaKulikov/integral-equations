package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import numerics.SpecialFunctions
import org.junit.jupiter.api.Tag
import problems.wsie.WeaklySingularProblem
import solvers.core.SolutionFunc
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests of the product-integration operator [ProductIntegrationOperator] (𝓛_m) and of the discrete Kulkarni scheme
 * [WeaklySingularSecondKindSolver.discreteKulkarni] (M-DK). Time tags are on the methods. The convergence of 𝓛_m
 * is checked pointwise only: norm convergence 𝓛_m → 𝓛 does not hold (collectively compact setting), and the
 * comparison with the classical Kulkarni scheme is a numerical observation printed with the prefix `CDK`.
 */
class WeaklySingularDiscreteKulkarniTest {

    private val alpha = 0.5

    private fun binom(d: Int, k: Int): Double {
        var c = 1.0
        for (i in 1..k) c = c * (d - k + i) / i
        return c
    }

    /** ∫_0^t (t − s)^(−alpha) s^d ds = t^(d+1−alpha) B(d+1, 1−alpha). */
    private fun volterraMoment(t: Double, d: Int): Double =
        if (t <= 0.0) 0.0 else t.pow(d + 1 - alpha) * SpecialFunctions.beta(d + 1.0, 1.0 - alpha)

    /** ∫_0^1 |t − s|^(−alpha) s^d ds: the Volterra part plus Σ_k C(d,k) t^(d−k) (1−t)^(k+1−alpha)/(k+1−alpha). */
    private fun fredholmMoment(t: Double, d: Int): Double {
        var right = 0.0
        for (k in 0..d) right += binom(d, k) * t.pow(d - k) * (1.0 - t).pow(k + 1 - alpha) / (k + 1 - alpha)
        return volterraMoment(t, d) + right
    }

    // ---- (1) exactness of L_m for polynomial k̃·v of degree < q ----

    @Test
    @Tag("fast")
    fun `L_m is exact for polynomial densities of degree below q`() {
        val ts = doubleArrayOf(0.0, 0.03, 0.25, 0.5, 0.61, 0.9999, 1.0)
        var worst = 0.0
        for (fine in listOf(Grid.uniform(4), Grid.power(5, r = 2.0))) for (q in intArrayOf(3, 4, 8)) {
            val vol = ProductIntegrationOperator(KernelWS(alpha), fine, q, volterra = true)
            val fre = ProductIntegrationOperator(KernelWS(alpha), fine, q, volterra = false)
            // k(t, s) = 1 + t s: k̃ v = s^d + t s^(d+1) is a polynomial of degree d + 1 < q in s (Fredholm only).
            val freK = ProductIntegrationOperator(KernelWS(alpha) { t, s -> 1.0 + t * s }, fine, q, volterra = false)
            for (t in ts) for (d in 0 until q) {
                val v = { s: Double -> s.pow(d) }
                val ev = abs(vol.apply(t, v) - volterraMoment(t, d))
                val ef = abs(fre.apply(t, v) - fredholmMoment(t, d))
                val ek = if (d + 1 < q) abs(freK.apply(t, v) - fredholmMoment(t, d) - t * fredholmMoment(t, d + 1)) else 0.0
                worst = maxOf(worst, ev, ef, ek)
                assertTrue(ev <= 1e-13 && ef <= 1e-13 && ek <= 1e-13, "q=$q d=$d t=$t: errors $ev $ef $ek")
            }
        }
        println("CDK exactness: max |L_m s^d - closed form| = %.2e".format(worst))
    }

    @Test
    @Tag("fast")
    fun `nodes lie strictly inside the cells and a non-refining fine grid is rejected`() {
        val fine = Grid.power(8, r = 3.0)
        val lm = ProductIntegrationOperator(KernelWS(alpha), fine, 4, volterra = false)
        val s = lm.nodes()
        assertTrue(s.size == 32)
        for (p in s.indices) {
            val l = p / 4
            assertTrue(s[p] > fine.x(l) && s[p] < fine.x(l + 1), "node $p outside cell $l")
        }
        val coarse = solver(Grid.power(4, r = 3.0), fredholm, fa.cL, fa.rhs)
        assertFailsWith<IllegalArgumentException> {
            coarse.discreteKulkarni(ProductIntegrationOperator(KernelWS(alpha), Grid.uniform(5), 4, volterra = false))
        }
        assertFailsWith<IllegalArgumentException> { coarse.discreteKulkarni(ProductIntegrationOperator(KernelWS(alpha), Grid.power(8, r = 3.0), 4, volterra = true)) }
    }

    // ---- (2) pointwise convergence L_m u → L u for u = √t, cos t (recorded) ----

    @Test
    @Tag("slow")
    fun `L_m u converges to L u for sqrt and cos as m grows`() {
        val ref = Grid.uniform(16)
        val refQuad = AlgebraicSingularQuadrature(alpha, 20)
        val refOps = listOf(
            "V" to WeaklySingularVolterraOperator(KernelWS(alpha), ref, refQuad, 40),
            "F" to WeaklySingularFredholmOperator(KernelWS(alpha), ref, refQuad, 40),
        )
        val ts = DoubleArray(101) { it / 100.0 }
        for ((uName, u) in listOf("sqrt" to { s: Double -> sqrt(s) }, "cos" to { s: Double -> cos(s) })) {
            for ((kind, refOp) in refOps) {
                val exact = DoubleArray(ts.size) { refOp.apply(ts[it], u) }
                val errs = intArrayOf(4, 8, 16, 32, 64).map { m ->
                    val lm = ProductIntegrationOperator(KernelWS(alpha), Grid.uniform(m), 4, volterra = kind == "V")
                    ts.indices.maxOf { abs(lm.apply(ts[it], u) - exact[it]) }
                }
                println("CDK conv $kind u=$uName q=4 m=4..64: " + errs.joinToString(" ") { "%.2e".format(it) })
                assertTrue(errs.all { it.isFinite() } && errs.last() < errs.first(), "$kind $uName: $errs")
            }
        }
    }

    // ---- (3) discrete Kulkarni versus classical Kulkarni, F-a and V-b, theta, B, power r = 3 ----

    private val fa = WeaklySingularProblem.F_A
    private val vb = WeaklySingularProblem.V_B

    private val fredholm = { g: Grid -> WeaklySingularFredholmOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5), 10) }
    private val volterra = { g: Grid -> WeaklySingularVolterraOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5)) }

    private fun solver(
        grid: Grid,
        makeOp: (Grid) -> WeaklySingularOperator,
        cL: Double,
        f: (Double) -> Double,
    ): WeaklySingularSecondKindSolver {
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        return WeaklySingularSecondKindSolver(basis, ProjFunctionals(basis), makeOp(grid), cL, f)
    }

    /** E_h = max |u − u_h| on 10 equally spaced points per cell (cell starts) plus b — as in the solver test. */
    private fun errorEh(grid: Grid, exact: (Double) -> Double, sol: SolutionFunc): Double {
        val bp = grid.breakpoints
        var e = abs(exact(bp.last()) - sol.eval(bp.last()))
        for (k in 0 until bp.size - 1) for (m in 0 until 10) {
            val t = bp[k] + (bp[k + 1] - bp[k]) * m / 10.0
            e = max(e, abs(exact(t) - sol.eval(t)))
        }
        return e
    }

    private fun compare(title: String, makeOp: (Grid) -> WeaklySingularOperator, prob: WeaklySingularProblem) {
        for (n in intArrayOf(8, 16)) {
            val grid = Grid.power(n, r = 3.0)
            val s = solver(grid, makeOp, prob.cL, prob.rhs)
            val t0 = System.nanoTime()
            val eK = errorEh(grid, prob.exact, s.kulkarni())
            val tK = (System.nanoTime() - t0) * 1e-9
            for (q in intArrayOf(4, 8)) for (p in intArrayOf(1, 2, 4)) {
                val lm = ProductIntegrationOperator(KernelWS(prob.alpha), Grid.power(n * p, r = 3.0), q, makeOp === volterra)
                val t1 = System.nanoTime()
                val eD = errorEh(grid, prob.exact, s.discreteKulkarni(lm))
                val tD = (System.nanoTime() - t1) * 1e-9
                println(
                    "CDK $title pow3 n=$n q=$q p=$p m=${n * p}: E_h discrete %.3e (%.2f s)  kulkarni %.3e (%.2f s)  ratio %.3f"
                        .format(eD, tD, eK, tK, eD / eK),
                )
                assertTrue(eD.isFinite() && eD < 1.0, "$title n=$n q=$q p=$p: E_h = $eD")
            }
        }
    }

    @Test
    @Tag("slow")
    fun `F-a discrete Kulkarni versus classical Kulkarni on power grid r=3`() = compare("F-a", fredholm, fa)

    @Test
    @Tag("slow")
    fun `V-b discrete Kulkarni versus classical Kulkarni on power grid r=3`() = compare("V-b", volterra, vb)
}
