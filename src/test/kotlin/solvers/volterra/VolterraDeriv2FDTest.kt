package solvers.volterra

import org.junit.jupiter.api.Tag
import problems.volterra.VolterraProblem
import problems.volterra.firstKindSolver
import problems.volterra.secondKindSolver
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * An FD cross-check of the second derivative of the Volterra image (V u)''(t) by the formula (V2'')
 * against a central finite difference of the first derivative
 * (V u)'(t) (Leibniz). The points are taken INSIDE the smooth pieces of the spline (not at the knots),
 * since omega_i'' is discontinuous at the knots.
 *
 * CRITICAL: the V2 kernels with K(t,t)!=0 are included (V2: K=1/(1+t+s), V2exp: e^{-(t-s)^2}),
 * where the term K(t,t) u'(t) shows up: without it the FD cross-check would fail.
 */
@Tag("fast")
class VolterraDeriv2FDTest {
    private val quad = GaussLegendre(8)

    @Test fun secondDerivOfImageMatchesFiniteDifference() {
        // V2, V2exp: K(t,t)!=0 (they check the term K(t,t)u'); V2win: K(t,t)=0 (the completeness of the diagonal).
        val problems = listOf(VolterraProblem.V2, VolterraProblem.V2exp, VolterraProblem.V2win)
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
                    // A central difference of the FIRST derivative (Leibniz), smooth in t inside a piece.
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
     * A direct control of the importance of the term K(t,t)u'(t): on V2exp (K(t,t)=1) we remove this term
     * by hand and make sure that the discrepancy with the FD becomes significant at least somewhere.
     */
    @Test fun boundaryTermIsNecessaryForKttNonzero() {
        val p = VolterraProblem.V2exp
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
                // The "wrong" version: without the term K(t,t) u'(t).
                val withoutTerm = op.applyDeriv2(t, u, uD) - p.kernel.k(t, t) * uD(t)
                maxMismatchWithout = maxOf(maxMismatchWithout, kotlin.math.abs(withoutTerm - fd))
            }
        }
        assertTrue(
            maxMismatchWithout > 1e-3,
            "A noticeable discrepancy without the term K(t,t)u' was expected, got $maxMismatchWithout",
        )
    }
}
