package solvers.fredholm

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue
import problems.fredholm.FredholmProblem

/**
 * An FD cross-check of the second derivative of the Fredholm image (K u)''(t) = int K_tt u ds
 * against a central finite difference. The points are taken
 * INSIDE the smooth pieces of the spline (not at the knots), since omega_i'' is discontinuous at the knots.
 */
@Tag("fast")
class FredholmDeriv2FDTest {
    private val quad = GaussLegendre(8)

    @Test fun secondDerivOfImageMatchesFiniteDifference() {
        val problems = listOf(FredholmProblem.F2, FredholmProblem.F2exp)
        val h = 1e-4
        for (p in problems) {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = FredholmOperator(p.kernel, grid, quad)
            // Interior points of the smooth pieces: the midpoints of the grid intervals.
            val innerTs = (0 until grid.n).map { 0.5 * (grid.x(it) + grid.x(it + 1)) }
            for (i in -2 until basis.n) {
                val u = { s: Double -> basis.omega(i, s) }
                for (t in innerTs) {
                    if (t - 2 * h < grid.a || t + 2 * h > grid.b) continue
                    val analytic = op.applyDeriv2(t, u)
                    val fd = (op.apply(t + h, u) - 2.0 * op.apply(t, u) + op.apply(t - h, u)) / (h * h)
                    val err = kotlin.math.abs(analytic - fd)
                    assertTrue(
                        err < 1e-5,
                        "F[${p.name}] (K omega_$i)'' at t=$t: analytic=$analytic fd=$fd err=$err",
                    )
                }
            }
        }
    }
}
