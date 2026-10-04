package problems.wsie

import numerics.AlgebraicSingularQuadrature
import org.junit.jupiter.api.Tag
import solvers.wsie.WeaklySingularFredholmOperator
import solvers.wsie.WeaklySingularOperator
import solvers.wsie.WeaklySingularVolterraOperator
import splines.Grid
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Consistency of the weakly singular model problems: the exact solution, the right-hand side and the kernel of
 * every problem satisfy the equation up to the error of a refined product-integration rule.
 *
 * The operator is applied on `Grid.uniform(8)` with 40 levels of geometric refinement at the ends, because the
 * exact solutions behave like powers of `t` and `1 − t` there; the residual is therefore a check of the closed
 * forms and series of [WeaklySingularProblem], not of the operators.
 */
@Tag("fast")
class WeaklySingularProblemsTest {

    private val points = doubleArrayOf(0.0, 1e-10, 1e-6, 0.01, 0.1, 0.37, 0.5, 0.9, 1.0)

    private fun operatorFor(p: WeaklySingularProblem): WeaklySingularOperator {
        val quad = AlgebraicSingularQuadrature(p.alpha, 16)
        return when (p.type) {
            WeaklySingularType.VOLTERRA -> WeaklySingularVolterraOperator(p.kernel, Grid.uniform(8), quad, 40)
            WeaklySingularType.FREDHOLM -> WeaklySingularFredholmOperator(p.kernel, Grid.uniform(8), quad, 40)
        }
    }

    private fun maxResidual(p: WeaklySingularProblem): Double {
        val op = operatorFor(p)
        var worst = 0.0
        for (t in points) {
            val r = p.exact(t) - p.cL * op.apply(t, p.exact) - p.rhs(t)
            worst = max(worst, abs(r))
        }
        return worst
    }

    @Test
    fun `every problem satisfies its equation`() {
        for (p in WeaklySingularProblem.ALL) {
            val r = maxResidual(p)
            println("C4 residual ${p.name} (alpha = %.4f, refinement 40, 16 nodes): max = %.2e".format(p.alpha, r))
            assertTrue(r <= 1e-12, "${p.name}: residual $r exceeds 1e-12")
        }
    }

    @Test
    fun `V-a Mittag-Leffler form agrees with the erfc form`() {
        var worst = 0.0
        for (t in points) {
            val ml = WeaklySingularProblem.V_A.exact(t)
            val erfcForm = WeaklySingularProblem.abelHalfErfcForm(t)
            worst = max(worst, abs(ml - erfcForm) / abs(erfcForm))
        }
        println("C4 V-a Mittag-Leffler vs erfc form: max rel diff = %.2e".format(worst))
        assertTrue(worst <= 1e-14, "V-a: Mittag-Leffler and erfc forms differ by $worst (relative)")
    }

    @Test
    fun `catalogue lists four problems with distinct names`() {
        val names = WeaklySingularProblem.ALL.map { it.name }
        assertEquals(listOf("V-a", "V-b", "V-c", "F-a"), names)
        assertEquals(WeaklySingularType.FREDHOLM, WeaklySingularProblem.F_A.type)
    }
}
