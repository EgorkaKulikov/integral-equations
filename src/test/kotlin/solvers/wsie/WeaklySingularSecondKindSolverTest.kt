package solvers.wsie

import numerics.AlgebraicSingularQuadrature
import org.junit.jupiter.api.Tag
import problems.wsie.WeaklySingularProblem
import solvers.core.SolutionFunc
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Smoke and monotonicity tests of [WeaklySingularSecondKindSolver] on two model problems.
 *
 * Only solvability and the decrease of E_h from n = 8 to n = 32 are asserted: the observed orders are
 * numerical observations and are measured by the probe driver, not fixed here.
 */
@Tag("slow")
class WeaklySingularSecondKindSolverTest {

    private val schemes = listOf("base", "sloan", "kulkarni", "itKulkarni")

    private fun run(s: WeaklySingularSecondKindSolver, scheme: String): SolutionFunc = when (scheme) {
        "base" -> s.base()
        "sloan" -> s.sloan()
        "kulkarni" -> s.kulkarni()
        else -> s.iteratedKulkarni()
    }

    /** E_h = max |u − u_h| on 10 equally spaced points per cell (cell starts) plus b. */
    private fun errorEh(grid: Grid, exact: (Double) -> Double, sol: SolutionFunc): Double {
        val bp = grid.breakpoints
        var e = abs(exact(bp.last()) - sol.eval(bp.last()))
        for (k in 0 until bp.size - 1) for (m in 0 until 10) {
            val t = bp[k] + (bp[k + 1] - bp[k]) * m / 10.0
            e = max(e, abs(exact(t) - sol.eval(t)))
        }
        return e
    }

    // ---- Problem F-a: u − 0.2 ∫_0^1 |t − s|^(−1/2) u(s) ds = f, u = 1 + √t + √(1 − t) ([WeaklySingularProblem.F_A]) ----

    private val fa = WeaklySingularProblem.F_A

    @Test
    fun `F-a closed-form image agrees with the refined Fredholm operator`() {
        val op = WeaklySingularFredholmOperator(KernelWS(0.5), Grid.uniform(8), AlgebraicSingularQuadrature(0.5), 40)
        var worst = 0.0
        for (t in doubleArrayOf(0.0, 1e-6, 0.1, 0.25, 0.5, 0.7, 0.999, 1.0)) {
            worst = max(worst, abs(op.apply(t, fa.exact) - (fa.exact(t) - fa.rhs(t)) / fa.cL))
        }
        println("C3 F-a closed form vs operator (refinement 40): max diff = %.2e".format(worst))
        assertTrue(worst <= 1e-12, "F-a closed form differs from the operator by $worst")
    }

    // ---- Problem V-b: u − ∫_0^t (t − s)^(−1/2) u(s) ds = f, u = cos t ----

    private val vFine = WeaklySingularVolterraOperator(KernelWS(0.5), Grid.uniform(64), AlgebraicSingularQuadrature(0.5))

    @Test
    fun `V-b right-hand side is partition independent`() {
        val other = WeaklySingularVolterraOperator(KernelWS(0.5), Grid.uniform(48), AlgebraicSingularQuadrature(0.5))
        var worst = 0.0
        for (t in doubleArrayOf(0.0, 1e-3, 0.1, 0.33, 0.5, 0.77, 1.0)) {
            val a = vFine.apply(t) { s -> cos(s) }
            val b = other.apply(t) { s -> cos(s) }
            worst = max(worst, abs(a - b) / max(1.0, abs(a)))
        }
        println("C3 V-b rhs partition check: max rel diff = %.2e".format(worst))
        assertTrue(worst <= 1e-13, "V-b right-hand side is partition dependent: $worst")
    }

    private fun table(
        title: String,
        grids: List<Pair<String, (Int) -> Grid>>,
        makeOp: (Grid) -> WeaklySingularOperator,
        cL: Double,
        f: (Double) -> Double,
        exact: (Double) -> Double,
    ) {
        val ns = intArrayOf(8, 16, 32)
        for ((gName, gridOf) in grids) {
            val eh = HashMap<String, DoubleArray>()
            for ((k, n) in ns.withIndex()) {
                val grid = gridOf(n)
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                val solver = WeaklySingularSecondKindSolver(basis, ProjFunctionals(basis), makeOp(grid), cL, f)
                for (sc in schemes) {
                    val t0 = System.nanoTime()
                    val sol = run(solver, sc)
                    val t1 = System.nanoTime()
                    val e = errorEh(grid, exact, sol)
                    val t2 = System.nanoTime()
                    eh.getOrPut(sc) { DoubleArray(ns.size) }[k] = e
                    println("C3 time $title $gName n=$n $sc: solve %.2f s, E_h eval %.2f s"
                        .format((t1 - t0) * 1e-9, (t2 - t1) * 1e-9))
                }
            }
            for (sc in schemes) {
                val v = eh.getValue(sc)
                println("C3 E_h $title %-10s %-11s n=8 %.3e  n=16 %.3e  n=32 %.3e".format(gName, sc, v[0], v[1], v[2]))
                assertTrue(v[2] < v[0], "$title $gName $sc: E_h does not decrease from n=8 to n=32: ${v.toList()}")
            }
        }
    }

    @Test
    fun `F-a all schemes with theta functionals`() = table(
        "F-a",
        listOf("uniform" to { n: Int -> Grid.uniform(n) }, "symPow3" to { n: Int -> Grid.symmetricPower(n, r = 3.0) }),
        { g -> WeaklySingularFredholmOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5), 10) },
        fa.cL, fa.rhs, fa.exact,
    )

    @Test
    fun `V-b all schemes with theta functionals`() = table(
        "V-b",
        listOf("uniform" to { n: Int -> Grid.uniform(n) }, "pow2" to { n: Int -> Grid.power(n, r = 2.0) }),
        { g -> WeaklySingularVolterraOperator(KernelWS(0.5), g, AlgebraicSingularQuadrature(0.5)) },
        WeaklySingularProblem.V_B.cL, WeaklySingularProblem.V_B.rhs, WeaklySingularProblem.V_B.exact,
    )

    @Test
    fun `derivative-based family is rejected`() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = WeaklySingularFredholmOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(0.5))
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularSecondKindSolver(basis, DeBoorFixFunctionals(basis), op, 0.2, fa.rhs)
        }
        val other = WeaklySingularFredholmOperator(KernelWS(0.5), Grid.uniform(8), AlgebraicSingularQuadrature(0.5))
        assertFailsWith<IllegalArgumentException> {
            WeaklySingularSecondKindSolver(basis, ProjFunctionals(basis), other, 0.2, fa.rhs)
        }
    }

    @Test
    fun `three-point quasi-interpolant family runs the base scheme`() {
        val e = DoubleArray(3)
        for ((k, n) in intArrayOf(8, 16, 32).withIndex()) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = WeaklySingularFredholmOperator(KernelWS(0.5), grid, AlgebraicSingularQuadrature(0.5), 10)
            val solver = WeaklySingularSecondKindSolver(basis, ThreePointFunctionals(basis), op, 0.2, fa.rhs)
            e[k] = errorEh(grid, fa.exact, solver.base())
        }
        println("C3 E_h F-a uniform    lambda-base n=8 %.3e  n=16 %.3e  n=32 %.3e".format(e[0], e[1], e[2]))
        assertTrue(e[2] < e[0], "three-point base: E_h does not decrease: ${e.toList()}")
    }
}
