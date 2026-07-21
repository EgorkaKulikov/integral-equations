package solvers.fredholm

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * FD-сверка второй производной образа Фредгольма (K u)''(t) = int K_tt u ds
 * с центральной конечной разностью (deboorfix-spec.md §4(в)). Точки берутся
 * ВНУТРИ гладких кусков сплайна (не в узлах), т.к. omega_i'' разрывна в узлах.
 */
class FredholmDeriv2FDTest {
    private val quad = GaussLegendre(8)

    @Test fun secondDerivOfImageMatchesFiniteDifference() {
        val problems = listOf(ModelProblem.F2, ModelProblem.F2exp)
        val h = 1e-4
        for (p in problems) {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = FredholmOperator(p.kernel, grid, quad)
            // Внутренние точки гладких кусков: середины интервалов сетки.
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
