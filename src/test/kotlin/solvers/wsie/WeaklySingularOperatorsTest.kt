package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import numerics.SpecialFunctions
import org.junit.jupiter.api.Tag
import splines.Grid
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Closed-form checks of [WeaklySingularVolterraOperator] and [WeaklySingularFredholmOperator].
 *
 * The reference values are exact integrals of polynomial densities against the Abel kernel (Beta functions), so
 * the tolerance 1e-13 measures the partition logic plus the library quadrature, not a discretization error.
 * The maxima are printed so that the reports can quote them.
 */
@Tag("fast")
class WeaklySingularOperatorsTest {

    private val alphas = doubleArrayOf(1.0 / 3.0, 0.5, 2.0 / 3.0)
    private val grids = listOf("uniform(16)" to Grid.uniform(16), "power(16, r=3)" to Grid.power(16, r = 3.0))

    private fun relErr(value: Double, exact: Double): Double =
        if (exact == 0.0) abs(value) else abs(value - exact) / abs(exact)

    @Test
    fun volterraMomentsMatchBetaFunction() {
        var worst = 0.0
        for ((name, grid) in grids) for (alpha in alphas) {
            val op = WeaklySingularVolterraOperator(KernelWS(alpha), grid, AlgebraicSingularQuadrature(alpha))
            for (t in doubleArrayOf(0.0, 1e-9, grid.x(3), 0.37, 1.0)) for (j in 0..3) {
                val exact = t.pow(j + 1 - alpha) * SpecialFunctions.beta(1.0 - alpha, j + 1.0)
                val err = relErr(op.apply(t) { s -> s.pow(j) }, exact)
                worst = max(worst, err)
                assertTrue(err <= 1e-13, "Volterra moment $name alpha=$alpha j=$j t=$t: rel err $err")
            }
        }
        println("WSIE-MAX volterraMoments rel=$worst")
    }

    @Test
    fun volterraWithSmoothFactorMatchesClosedForm() {
        var worst = 0.0
        for ((name, grid) in grids) for (alpha in alphas) {
            val op = WeaklySingularVolterraOperator(
                KernelWS(alpha) { t, s -> 1.0 + t * s }, grid, AlgebraicSingularQuadrature(alpha),
            )
            for (t in doubleArrayOf(1e-9, grid.x(3), 0.37, 1.0)) {
                // ∫_0^t (t − s)^(−α)(1 + ts) ds = t^(1−α)/(1−α) + t · t^(2−α) B(1−α, 2)
                val exact = t.pow(1 - alpha) / (1 - alpha) + t * t.pow(2 - alpha) * SpecialFunctions.beta(1 - alpha, 2.0)
                val err = relErr(op.apply(t) { 1.0 }, exact)
                worst = max(worst, err)
                assertTrue(err <= 1e-13, "Volterra k=1+ts $name alpha=$alpha t=$t: rel err $err")
            }
        }
        println("WSIE-MAX volterraSmoothFactor rel=$worst")
    }

    @Test
    fun fredholmAbelKernelMatchesClosedForm() {
        var worst = 0.0
        for ((name, grid) in grids) {
            val op = WeaklySingularFredholmOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(0.5))
            for (t in doubleArrayOf(0.0, 1e-9, 0.25, 0.5, 1.0)) {
                val exact0 = 2.0 * (sqrt(t) + sqrt(1.0 - t))
                val exact1 = 4.0 / 3.0 * t.pow(1.5) + 2.0 / 3.0 * (1.0 - t).pow(1.5) + 2.0 * t * sqrt(1.0 - t)
                val err0 = relErr(op.apply(t) { 1.0 }, exact0)
                val err1 = relErr(op.apply(t) { s -> s }, exact1)
                worst = max(worst, max(err0, err1))
                assertTrue(err0 <= 1e-13, "Fredholm u=1 $name t=$t: rel err $err0")
                assertTrue(err1 <= 1e-13, "Fredholm u=s $name t=$t: rel err $err1")
            }
        }
        println("WSIE-MAX fredholmAbel rel=$worst")
    }

    @Test
    fun applyOnSupportAgreesWithDirectQuadratureAndIndicator() {
        val grid = Grid.power(16, r = 3.0)
        val alpha = 1.0 / 3.0
        val quad = AlgebraicSingularQuadrature(alpha)
        val kernel = KernelWS(alpha) { t, s -> 1.0 + t * s }
        val vol = WeaklySingularVolterraOperator(kernel, grid, quad)
        val fred = WeaklySingularFredholmOperator(kernel, grid, quad)
        val lo = grid.x(2)
        val hi = grid.x(5)
        val u = { s: Double -> 1.0 + s + s * s }
        val masked = { s: Double -> if (s in lo..hi) u(s) else 0.0 }
        val inside = 0.5 * (grid.x(3) + grid.x(4))
        var worst = 0.0
        for (t in doubleArrayOf(0.5 * grid.x(2), lo, inside, grid.x(4), hi, 0.9)) {
            val volDirect = if (t <= lo) 0.0 else
                quad.integrate(grid.breakpoints.filter { it in lo..min(hi, t) }.let { pts ->
                    (pts + min(hi, t)).distinct().toDoubleArray()
                }, t) { s -> kernel.k(t, s) * u(s) }
            val fredDirect = quad.integrate(doubleArrayOf(lo, grid.x(3), grid.x(4), hi), t) { s -> kernel.k(t, s) * u(s) }
            val pairs = listOf(
                vol.applyOnSupport(t, lo, hi, u) to volDirect,
                vol.apply(t, masked) to volDirect,
                fred.applyOnSupport(t, lo, hi, u) to fredDirect,
                fred.apply(t, masked) to fredDirect,
            )
            for ((value, reference) in pairs) {
                val err = abs(value - reference) / max(1.0, abs(reference))
                worst = max(worst, err)
                assertTrue(err <= 1e-14, "applyOnSupport t=$t: $value vs $reference (err $err)")
            }
        }
        assertEquals(0.0, vol.applyOnSupport(lo, lo, hi, u))
        assertEquals(0.0, fred.applyOnSupport(0.3, hi, lo, u))
        println("WSIE-MAX applyOnSupport err=$worst")
    }

    @Test
    fun endpointRefinementResolvesSquareRootDensity() {
        // (Vu)(1) for u(s) = s^(1/2), alpha = 1/2: B(1/2, 3/2) = π/2.
        val grid = Grid.uniform(8)
        val quad = AlgebraicSingularQuadrature(0.5)
        val exact = PI / 2.0
        val err0 = relErr(WeaklySingularVolterraOperator(KernelWS(0.5), grid, quad, 0).apply(1.0) { sqrt(it) }, exact)
        val err40 = relErr(WeaklySingularVolterraOperator(KernelWS(0.5), grid, quad, 40).apply(1.0) { sqrt(it) }, exact)
        // The Fredholm operator refines at both ends: ∫_0^1 |1/2 − s|^(−1/2) s^(1/2) (1 − s)^(1/2) ds is symmetric,
        // so only consistency between m = 40 and a further refinement is checked here.
        val f40 = WeaklySingularFredholmOperator(KernelWS(0.5), grid, quad, 40).apply(0.5) { sqrt(it * (1 - it)) }
        val f60 = WeaklySingularFredholmOperator(KernelWS(0.5), grid, quad, 60).apply(0.5) { sqrt(it * (1 - it)) }
        println("WSIE-MAX endpointRefinement m0=$err0 m40=$err40 fredholm|m40-m60|=${abs(f40 - f60)}")
        assertTrue(err40 <= 1e-12, "endpointRefinement m=40: rel err $err40")
        assertTrue(err0 > err40, "refinement must reduce the error: m=0 $err0, m=40 $err40")
        assertTrue(abs(f40 - f60) <= 1e-12 * abs(f60), "Fredholm refinement m=40 vs m=60: $f40 vs $f60")
    }

    @Test
    fun invalidArgumentsAreRejected() {
        for (bad in doubleArrayOf(0.0, 1.0, -0.1, 1.5, Double.NaN)) {
            assertFailsWith<IllegalArgumentException>("alpha=$bad") { KernelWS(bad) }
        }
        val grid = Grid.uniform(4)
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularVolterraOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(1.0 / 3.0))
        }
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularFredholmOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(1.0 / 3.0))
        }
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularVolterraOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(0.5), -1)
        }
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularFredholmOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(0.5), -1)
        }
    }

    @Test
    fun nanArgumentPropagates() {
        val grid = Grid.uniform(4)
        val quad = AlgebraicSingularQuadrature(0.5)
        val ops = listOf<WeaklySingularOperator>(
            WeaklySingularVolterraOperator(KernelWS(0.5), grid, quad),
            WeaklySingularFredholmOperator(KernelWS(0.5), grid, quad),
        )
        for (op in ops) {
            assertTrue(op.apply(Double.NaN) { 1.0 }.isNaN(), "${op::class.simpleName}.apply(NaN)")
            assertTrue(op.applyOnSupport(Double.NaN, 0.0, 0.5) { 1.0 }.isNaN(), "${op::class.simpleName}.applyOnSupport(NaN)")
        }
    }
}
