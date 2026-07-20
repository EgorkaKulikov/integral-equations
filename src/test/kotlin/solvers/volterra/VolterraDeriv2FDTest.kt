package solvers.volterra

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * FD-сверка второй производной образа Вольтерра (V u)''(t) по формуле (V2'')
 * (deboorfix-spec.md §4(в)) с центральной конечной разностью первой производной
 * (V u)'(t) (Лейбниц). Точки берутся ВНУТРИ гладких кусков сплайна (не в узлах),
 * т.к. omega_i'' разрывна в узлах.
 *
 * КРИТИЧНО: включены V2-ядра с K(t,t)!=0 (V2: K=1/(1+t+s), V2exp: e^{-(t-s)^2}),
 * где проявляется член K(t,t) u'(t): без него FD-сверка провалится.
 */
class VolterraDeriv2FDTest {
    private val quad = GaussLegendre(8)

    @Test fun secondDerivOfImageMatchesFiniteDifference() {
        // V2, V2exp: K(t,t)!=0 (проверяют член K(t,t)u'); V2win: K(t,t)=0 (полнота диагонали).
        val problems = listOf(ModelProblem.V2, ModelProblem.V2exp, ModelProblem.V2win)
        val h = 1e-5
        for (p in problems) {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = VolterraOperator(p.kernel, grid, quad)
            val innerTs = (0 until grid.n).map { 0.5 * (grid.x(it) + grid.x(it + 1)) }
            for (i in -2 until basis.n) {
                val u = { s: Double -> basis.omega(i, s) }
                val uD = { s: Double -> basis.omegaDeriv(i, s) }
                for (t in innerTs) {
                    if (t - h < grid.a || t + h > grid.b) continue
                    val analytic = op.applyDeriv2(t, u, uD)
                    // Центральная разность ПЕРВОЙ производной (Лейбниц), гладкой в t внутри куска.
                    val fd = (op.applyDeriv(t + h, u) - op.applyDeriv(t - h, u)) / (2.0 * h)
                    val err = kotlin.math.abs(analytic - fd)
                    assertTrue(
                        err < 1e-5,
                        "V[${p.name}] (V omega_$i)'' at t=$t: analytic=$analytic fd=$fd err=$err",
                    )
                }
            }
        }
    }

    /**
     * Прямой контроль важности члена K(t,t)u'(t): на V2exp (K(t,t)=1) убираем этот член
     * вручную и убеждаемся, что расхождение с FD становится значимым хотя бы где-то.
     */
    @Test fun boundaryTermIsNecessaryForKttNonzero() {
        val p = ModelProblem.V2exp
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = VolterraOperator(p.kernel, grid, quad)
        val h = 1e-5
        var maxMismatchWithout = 0.0
        val innerTs = (0 until grid.n).map { 0.5 * (grid.x(it) + grid.x(it + 1)) }
        for (i in -2 until basis.n) {
            val u = { s: Double -> basis.omega(i, s) }
            val uD = { s: Double -> basis.omegaDeriv(i, s) }
            for (t in innerTs) {
                if (t - h < grid.a || t + h > grid.b) continue
                val fd = (op.applyDeriv(t + h, u) - op.applyDeriv(t - h, u)) / (2.0 * h)
                // "Неправильная" версия: без члена K(t,t) u'(t).
                val withoutTerm = op.applyDeriv2(t, u, uD) - p.kernel.k(t, t) * uD(t)
                maxMismatchWithout = maxOf(maxMismatchWithout, kotlin.math.abs(withoutTerm - fd))
            }
        }
        assertTrue(
            maxMismatchWithout > 1e-3,
            "Ожидалось заметное расхождение без члена K(t,t)u', получено $maxMismatchWithout",
        )
    }
}
