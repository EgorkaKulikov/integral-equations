package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import numerics.LinearAlgebra
import org.junit.jupiter.api.Tag
import problems.wsie.WeaklySingularProblem
import solvers.core.SolutionFunc
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests of the two-grid Kulkarni scheme [WeaklySingularSecondKindSolver.twoGridKulkarni] (M-DS).
 *
 * The time tags are placed on the methods: the nesting checks need no quadrature and run in `fastTest`,
 * the solves on F-a and V-b run in `slowTest`. The comparison with the classical Kulkarni scheme is a
 * numerical observation (no order is proven), so the numbers are printed with the prefix `C2G`.
 */
class WeaklySingularTwoGridKulkarniTest {

    // ---- (1) nesting of the power grids: x^(np)_{pk} = x^(n)_k exactly ----

    @Test
    @Tag("fast")
    fun `power grids are nested for m = n p`() {
        for (r in doubleArrayOf(1.0, 2.0, 3.0)) for (n in intArrayOf(4, 8, 16, 32)) for (p in 2..4) {
            assertNested(Grid.power(n, r = r), Grid.power(n * p, r = r), p, "power r=$r n=$n p=$p")
        }
    }

    @Test
    @Tag("fast")
    fun `symmetric power grids are nested for even n and m = n p`() {
        for (r in doubleArrayOf(1.0, 2.0, 3.0)) for (n in intArrayOf(4, 8, 16, 32)) for (p in 2..4) {
            assertNested(Grid.symmetricPower(n, r = r), Grid.symmetricPower(n * p, r = r), p, "symPow r=$r n=$n p=$p")
        }
    }

    /** Bitwise equality: both nodes are `pow` of the same correctly rounded quotient (pk)/(np) = k/n. */
    private fun assertNested(coarse: Grid, fine: Grid, p: Int, label: String) {
        assertEquals(coarse.n * p, fine.n, label)
        for (k in 0..coarse.n) assertEquals(coarse.x(k), fine.x(p * k), "$label: node k=$k")
    }

    @Test
    @Tag("fast")
    fun `non-nested or mismatched fine level is rejected`() {
        val coarse = solver(Grid.power(8, r = 3.0), fredholm)
        assertFailsWith<IllegalArgumentException> { coarse.twoGridKulkarni(solver(Grid.uniform(16), fredholm), 2) }
        assertFailsWith<IllegalArgumentException> { coarse.twoGridKulkarni(solver(Grid.power(24, r = 3.0), fredholm), 2) }
        assertFailsWith<IllegalArgumentException> { coarse.twoGridKulkarni(solver(Grid.power(16, r = 3.0), volterra), 2) }
        assertFailsWith<IllegalArgumentException> { coarse.twoGridKulkarni(coarse, 0) }
    }

    // ---- nesting of the spaces (N) and the identity M_nm R = M ----

    @Test
    @Tag("slow")
    fun `refinement matrix reproduces coarse basis and M_nm R equals M`() {
        for ((title, makeOp) in listOf("F-a" to fredholm, "V-b" to volterra)) for (p in 2..3) {
            val coarse = solver(Grid.power(8, r = 3.0), makeOp)
            val fine = solver(Grid.power(8 * p, r = 3.0), makeOp)
            val (mNm, _, r) = coarse.twoGridBlocks(fine)
            // (N) for B: omega^(n)_i = sum_l R_{l,i} omega^(m)_l at 10 points per fine cell.
            var nest = 0.0
            val bp = fine.grid.breakpoints
            for (i in 0 until coarse.dim) {
                val col = DoubleArray(fine.dim) { r[it, i] }
                for (k in 0 until bp.size - 1) for (s in 0 until 10) {
                    val t = bp[k] + (bp[k + 1] - bp[k]) * s / 10.0
                    nest = max(nest, abs(coarse.basis.omega(i - 2, t) - fine.basis.evalSpline(col, t)))
                }
            }
            val mr = LinearAlgebra.matMat(mNm, r)
            val m = coarse.matrixM()
            var diff = 0.0
            for (a in 0 until coarse.dim) for (b in 0 until coarse.dim) diff = max(diff, abs(mr[a, b] - m[a, b]))
            println("C2G $title pow3 n=8 p=$p: nesting defect %.2e, max |M_nm R - M| = %.2e".format(nest, diff))
            assertTrue(nest <= 1e-12, "$title p=$p: R does not reproduce the coarse basis: $nest")
            assertTrue(diff <= 1e-10, "$title p=$p: M_nm R differs from M by $diff")
        }
    }

    // ---- solves ----

    private val fa = WeaklySingularProblem.F_A
    private val vb = WeaklySingularProblem.V_B

    private val fredholm = { g: Grid -> WeaklySingularFredholmOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5), 10) }
    private val volterra = { g: Grid -> WeaklySingularVolterraOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5)) }

    private fun solver(
        grid: Grid,
        makeOp: (Grid) -> WeaklySingularOperator,
        cL: Double = fa.cL,
        f: (Double) -> Double = fa.rhs,
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

    // ---- (2) two-grid p = 2 versus classical Kulkarni on the coarse grid, power grid r = 3 ----

    private fun compare(
        title: String,
        makeOp: (Grid) -> WeaklySingularOperator,
        cL: Double,
        f: (Double) -> Double,
        exact: (Double) -> Double,
    ) {
        val p = 2
        val ns = intArrayOf(8, 16)
        val eK = DoubleArray(ns.size)
        val e2 = DoubleArray(ns.size)
        for ((i, n) in ns.withIndex()) {
            val grid = Grid.power(n, r = 3.0)
            val coarse = solver(grid, makeOp, cL, f)
            val fine = solver(Grid.power(n * p, r = 3.0), makeOp, cL, f)
            val t0 = System.nanoTime()
            eK[i] = errorEh(grid, exact, coarse.kulkarni())
            val t1 = System.nanoTime()
            e2[i] = errorEh(grid, exact, coarse.twoGridKulkarni(fine, p))
            val t2 = System.nanoTime()
            // Context only (not asserted): GKV approximate the fine iterated collocation u_m, i.e. fine.sloan().
            val eSm = errorEh(grid, exact, fine.sloan())
            println(
                "C2G $title pow3 n=$n: E_h kulkarni %.3e (%.2f s)  twoGrid p=$p %.3e (%.2f s)  ratio %.3f  fine sloan %.3e"
                    .format(eK[i], (t1 - t0) * 1e-9, e2[i], (t2 - t1) * 1e-9, e2[i] / eK[i], eSm),
            )
        }
        for (i in ns.indices) assertTrue(e2[i].isFinite(), "$title n=${ns[i]}: two-grid E_h is not finite")
        assertTrue(e2[1] < e2[0], "$title: two-grid E_h does not decrease from n=8 to n=16: ${e2.toList()}")
    }

    @Test
    @Tag("slow")
    fun `F-a two-grid p=2 versus classical Kulkarni on power grid r=3`() =
        compare("F-a", fredholm, fa.cL, fa.rhs, fa.exact)

    @Test
    @Tag("slow")
    fun `V-b two-grid p=2 versus classical Kulkarni on power grid r=3`() =
        compare("V-b", volterra, vb.cL, vb.rhs, vb.exact)

    // ---- (3) p = 1: U = L P_n, so the scheme coincides with the Sloan iteration ----

    @Test
    @Tag("slow")
    fun `p=1 reproduces the Sloan iteration to roundoff`() {
        for ((title, makeOp, prob) in listOf(Triple("F-a", fredholm, fa), Triple("V-b", volterra, vb))) {
            val grid = Grid.power(8, r = 3.0)
            val s = solver(grid, makeOp, prob.cL, prob.rhs)
            val two = s.twoGridKulkarni(s, 1)
            val sloan = s.sloan()
            var diff = 0.0
            var scale = 0.0
            val bp = grid.breakpoints
            for (k in 0 until bp.size - 1) for (m in 0 until 10) {
                val t = bp[k] + (bp[k + 1] - bp[k]) * m / 10.0
                diff = max(diff, abs(two.eval(t) - sloan.eval(t)))
                scale = max(scale, abs(sloan.eval(t)))
            }
            println("C2G $title pow3 n=8 p=1: max |twoGrid - sloan| = %.2e (max |u_h| = %.2e)".format(diff, scale))
            assertTrue(diff <= 1e-12 * scale, "$title: p=1 differs from sloan() by $diff")
        }
    }
}
