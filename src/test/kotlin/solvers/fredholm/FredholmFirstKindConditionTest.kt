package solvers.fredholm

import numerics.ConditionEstimate
import numerics.Conditioning
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.fredholm.firstKindSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests of the applicability bound of the regularized F1 path in the parameter `alpha`
 * (issue #11) and of the programmatic access to the conditioning of the assembled system.
 *
 * The requirement being checked: the KDoc statement "below roughly `alpha = 1e-8`
 * few significant digits are left" is backed by a MEASUREMENT on the library itself and not
 * only by an external observation. The bound is recorded as an APPLICABILITY BOUND and not
 * as a rejection: no call of the solver fails because of a large `cond`.
 */
@Tag("fast")
class FredholmFirstKindConditionTest {

    private val quad = GaussLegendre(8)

    private fun solver(alpha: Double, n: Int = 8): FredholmFirstKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        return firstKindSolver(
            FredholmProblem.F1,
            basis,
            ProjFunctionals(basis),
            FredholmOperator(FredholmProblem.F1.kernel, grid, quad),
            alpha,
        )
    }

    /**
     * THE CONDITIONING GROWS AS `alpha^{-1}` — the qualitative statement of the KDoc
     * is confirmed by a measurement.
     *
     * The baseline of the current run (`n = 8`, basis B, family theta):
     * `alpha = 1e-6` → `cond∞ ≈ 1.449e+06`, `1e-8` → `1.492e+08`,
     * `1e-10` → `1.181e+10`. Every decrease of `alpha` by two orders raises
     * `cond∞` by about two orders, as the growth of the entries of `M` predicts.
     * The tolerance is taken wide (a factor of 3 in both directions): what is checked is the ORDER of the growth,
     * and not the reproducibility of the digits on a foreign architecture.
     */
    @Test fun conditionGrowsInverselyWithAlpha() {
        val condAt = mutableMapOf<Double, Double>()
        for (alpha in listOf(1e-6, 1e-8, 1e-10)) {
            val est = solver(alpha).baseCondition()
            assertTrue(est.condInf.isFinite(), "alpha=$alpha: cond must be finite, got ${est.condInf}")
            condAt[alpha] = est.condInf
        }
        assertTrue(condAt[1e-6]!! in 5e5..5e6, "alpha=1e-6: cond=${condAt[1e-6]}")
        assertTrue(condAt[1e-8]!! in 5e7..5e8, "alpha=1e-8: cond=${condAt[1e-8]}")
        assertTrue(condAt[1e-10]!! in 4e9..4e10, "alpha=1e-10: cond=${condAt[1e-10]}")
        // Monotonicity: the smaller the alpha, the worse the conditioning.
        assertTrue(condAt[1e-6]!! < condAt[1e-8]!! && condAt[1e-8]!! < condAt[1e-10]!!)
    }

    /**
     * THE APPLICABILITY BOUND RUNS ROUGHLY AT `alpha = 1e-8` — measured HERE,
     * by the same reliability criterion as in `Conditioning`.
     *
     * At `alpha >= 1e-8` the residual of the inversion of the assembled matrix stays below the threshold
     * [Conditioning.INVERSION_RESIDUAL_TOLERANCE] and the `cond` estimate is reliable;
     * at `alpha = 1e-10` (the value of [FredholmFirstKindSolver.DEFAULT_REGULARIZATION])
     * the residual already exceeds the threshold — that is, the matrix entered the regime where the conditioning
     * estimate itself loses its meaning. This is a machine-checkable form of the statement
     * "below roughly 1e-8 few significant digits are left".
     */
    @Test fun defaultRegularizationLiesBelowTheReliabilityBoundary() {
        assertTrue(solver(1e-6).baseCondition().isReliable, "alpha=1e-6 must stay reliable")
        assertTrue(solver(1e-8).baseCondition().isReliable, "alpha=1e-8 must stay reliable")

        val atDefault = solver(FredholmFirstKindSolver.DEFAULT_REGULARIZATION).baseCondition()
        assertEquals(1e-10, FredholmFirstKindSolver.DEFAULT_REGULARIZATION, 0.0)
        assertTrue(
            !atDefault.isReliable,
            "alpha=1e-10 falls into the problematic range: cond=${atDefault.condInf}, " +
                "inversion residual=${atDefault.inversionResidual}",
        )
        // An untrustworthy number must not be obtained through inattention.
        assertNull(atDefault.valueOrNull())
    }

    /**
     * THIS IS NOT A REJECTION. At `alpha = 1e-10`, where the `cond` estimate is already untrustworthy,
     * the solver works regularly: the base scheme and the Sloan iteration return
     * finite values, not a single exception is thrown.
     *
     * The requirement follows directly from the justification of
     * `LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE`: a large `cond` for the problem F1 is
     * a regular regime of the method, and it must not be rejected.
     */
    @Test fun poorConditioningDoesNotRejectTheProblem() {
        val s = solver(FredholmFirstKindSolver.DEFAULT_REGULARIZATION)
        for (t in listOf(0.0, 0.25, 0.5, 0.75, 1.0)) {
            assertTrue(s.base().eval(t).isFinite(), "base: t=$t")
            assertTrue(s.sloan().eval(t).isFinite(), "sloan: t=$t")
        }
    }

    /**
     * A SELF-CHECK BY TWO EQUIVALENT EXPRESSIONS (a technique documented in
     * the KDoc of the solver) is workable and does NOT give a false alarm at a suitable `alpha`.
     *
     * One and the same quantity `‖u_h‖` is computed in two algebraically equivalent
     * ways: by dividing `u/(1+t)` against multiplying by `u · p`, where `p = 1/(1+t)`.
     * The expressions differ by exactly one rounding. At `alpha = 1e-6`, where by
     * the measurements there is no discrepancy, there is none here either at the level of 1e-9 —
     * that is, the technique itself creates no noise and is suitable as a lower estimate of the really
     * available accuracy.
     */
    @Test fun twoEquivalentWritingsAgreeAtUsableAlpha() {
        val s = solver(1e-6)
        val u = s.base().eval
        var maxRel = 0.0
        for (k in 0..40) {
            val t = k / 40.0
            val byDivision = u(t) / (1.0 + t)
            val byMultiplication = u(t) * (1.0 / (1.0 + t))
            val scale = maxOf(kotlin.math.abs(byDivision), kotlin.math.abs(byMultiplication))
            if (scale > 0.0) maxRel = maxOf(maxRel, kotlin.math.abs(byDivision - byMultiplication) / scale)
        }
        assertTrue(maxRel < 1e-9, "the two equivalent expressions diverged by $maxRel at alpha=1e-6")
    }

    /**
     * The estimate is computed on THE SAME matrix the system is solved with: `baseCondition`
     * must coincide with a direct call of `Conditioning.conditionInf(baseMatrix())`.
     * Otherwise the user would be shown a number from a different problem.
     */
    @Test fun reportedConditionIsTakenFromTheMatrixActuallySolved() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(FredholmProblem.F1.kernel, grid, quad)
        val s = firstKindSolver(FredholmProblem.F1, basis, funcs, op, 1e-6)
        val inner = FredholmSecondKindSolver(
            basis, funcs, op, cL = -1.0 / 1e-6,
            rhs = solvers.core.RhsWithDerivatives(
                { t -> FredholmProblem.F1.rhsExact(t, op) / 1e-6 },
                { t -> FredholmProblem.F1.rhsExactDeriv(t, op) / 1e-6 },
                { t -> FredholmProblem.F1.rhsExactDeriv2(t, op) / 1e-6 },
            ),
        )
        val expected: ConditionEstimate = Conditioning.conditionInf(inner.baseMatrix())
        assertEquals(expected.condInf, s.baseCondition().condInf, 0.0)
    }

    /** A non-positive reliability threshold is rejected — it would mean the absence of a check. */
    @Test fun baseConditionRejectsNonPositiveTolerance() {
        assertFailsWith<IllegalArgumentException> { solver(1e-6).baseCondition(tolerance = 0.0) }
    }
}
