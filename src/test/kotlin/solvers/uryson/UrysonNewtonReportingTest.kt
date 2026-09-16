package solvers.uryson

import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * THE REPORTING CONTRACT of the Newton iterations of the Uryson solver (item 8.4 of the spec).
 *
 * Three statements are checked that before the fix held in NONE of the three
 * iterative schemes (`solveBase`, `kulkarni`, `nystrom`):
 *
 *  1. `iterations` is the number of Newton steps ACTUALLY performed, and not the number of checks
 *     of the criterion. Previously the counter was incremented BEFORE the residual check, so on
 *     an instant convergence a one went out at zero steps (measurement: `cL = 0`
 *     gave `iterations = 1` with an unchanged coefficient vector).
 *  2. `residual` refers to the RETURNED point. Previously, on exiting by the step criterion,
 *     the residual at the point BEFORE the step was returned, that is, a systematically OVERSTATED one.
 *     Both actual cases of exiting by the step (the base scheme, problem B, n=8):
 *     at `tol = 1e-2` `2.1409e-2` was reported instead of `4.9945e-5` (429 times worse),
 *     at `tol = 1e-10` `2.7402e-10` instead of `4.4409e-15` (6.2e4 times worse).
 *  3. `converged` is trustworthy: it is set by the ACTUAL residual, and not by which
 *     criterion interrupted the loop. Previously a small step was declared a success unconditionally,
 *     so a scheme reported `converged = true` with a residual ABOVE the requested `tol`.
 *
 * PROTECTION AGAINST DEGENERATION. The three tests below iterate over configurations and check the condition
 * not in each of them (for example, only where the scheme converged, or only where
 * the exit happened by the step). Such an iteration is SILENTLY zeroed if the behaviour changes
 * and no suitable configurations are left: the test will keep being green while checking
 * nothing. Therefore each of them counts the cases ACTUALLY CHECKED and requires
 * that there be more than zero of them.
 */
@Tag("fast")
class UrysonNewtonReportingTest {

    private val quad = GaussLegendre(8)

    /**
     * The tolerance of the comparison of the reported residual with an independently recomputed one.
     *
     * Both numbers are computed by one and the same code (`CollocationCore.xiVector`) at one
     * and the same point, so in fact they coincide bitwise; the tolerance is left for the case of
     * a different summation order on a change of backend. It is many orders smaller than
     * the discrepancies recorded above (429 and 6.2e4 times), so the old behaviour
     * would certainly not have passed this test.
     */
    private val residualMatchTolerance = 1e-12

    /** The lower bound of the tolerance of the base scheme — `UrysonSecondKindSolver.NEWTON_TOLERANCE_FLOOR`. */
    private val newtonToleranceFloor = 1e-13

    private fun solverFor(
        problem: UrysonProblem,
        n: Int,
        tol: Double,
        lambdaOverride: Double? = null,
    ): UrysonSecondKindSolver {
        val basis = MinimalSplineBasis(GeneratingSystem.B, Grid.uniform(n))
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, basis.grid, quad)
        return UrysonSecondKindSolver(
            basis = basis,
            funcs = funcs,
            space = space,
            op = op,
            cL = lambdaOverride ?: problem.lambda,
            rhs = { t -> problem.rhsExact(t, op) },
            tol = tol,
        )
    }

    /** The vector `theta_j(f)` — also the initial approximation of the base scheme. */
    private fun thetaOf(solver: UrysonSecondKindSolver): DoubleArray {
        val n = solver.grid.n
        return DoubleArray(n + 2) { solver.funcs.chi(it - 2).apply(solver.rhs, { 0.0 }, { 0.0 }) }
    }

    /**
     * An INDEPENDENT recomputation of the residual of the base scheme `F(c) = c - theta(f) - cL Xi(c)`
     * at a given point — the baseline for checking the field `residual`.
     */
    private fun baseResidualAt(solver: UrysonSecondKindSolver, coeffs: DoubleArray): Double {
        val core = CollocationCore(solver.basis, solver.funcs, solver.op)
        val thetaF = thetaOf(solver)
        val xi = core.xiVector(coeffs)
        return LinearAlgebra.normInf(
            DoubleArray(coeffs.size) { coeffs[it] - thetaF[it] - solver.cL * xi[it] },
        )
    }

    /** The outcome of reproducing the base scheme: the same as what the solver returned, plus OBSERVATIONS of the loop. */
    private class Replay(
        val steps: Int,
        val residual: Double,
        val converged: Boolean,
        /** How many times the step computation was ACTUALLY called — an independent step counter. */
        val stepCalls: Int,
        /** The norm of the last step: if it is below the tolerance, the loop exited BY THE STEP. */
        val lastStepNorm: Double,
        val coeffs: DoubleArray,
    )

    /**
     * Reproduces the base scheme THROUGH THE SAME helper [runNewtonIterations], but with
     * wrappers observing the loop: it counts the calls of the step computation and remembers
     * the norm of the last step.
     *
     * Needed for two things unavailable from the outside through the public API:
     *  - to find out by which of the two criteria the loop exited (without this one cannot claim
     *    that the subject of the fix — the exit BY THE STEP — was touched at all);
     *  - to get a counter of the performed steps INDEPENDENT of the field `performedSteps`.
     *
     * The correctness of the reproduction is not postulated but CHECKED: every test
     * using this method compares `steps` and `residual` with what the real
     * `solveBase()` returned.
     */
    private fun replayBase(solver: UrysonSecondKindSolver, tol: Double): Replay {
        val n = solver.grid.n
        val core = CollocationCore(solver.basis, solver.funcs, solver.op)
        val thetaF = thetaOf(solver)
        val c = thetaF.copyOf()
        var stepCalls = 0
        var lastStepNorm = Double.NaN
        val run = runNewtonIterations(
            x = c,
            maxSteps = UrysonSecondKindSolver.DEFAULT_MAX_ITERATIONS,
            tolerance = maxOf(tol, newtonToleranceFloor),
            residualAt = { current ->
                val xi = core.xiVector(current)
                DoubleArray(n + 2) { current[it] - thetaF[it] - solver.cL * xi[it] }
            },
            stepAt = { current, f ->
                stepCalls++
                val b = core.bMatrix(current)
                val jacobian = DenseMatrix.build(n + 2, n + 2) { r, col ->
                    val value = -solver.cL * b[r, col]
                    if (r == col) value + 1.0 else value
                }
                val delta = LinearAlgebra.solve(jacobian, DoubleArray(n + 2) { -f[it] })
                lastStepNorm = LinearAlgebra.normInf(delta)
                delta
            },
        )
        return Replay(run.performedSteps, run.residual, run.converged, stepCalls, lastStepNorm, c)
    }

    /**
     * INSTANT CONVERGENCE: at `cL = 0` the equation turns into `x = f`, and
     * the initial approximation of the base scheme `c_0 = theta(f)` is ALREADY the solution.
     * Hence not a single Newton step is needed, and an honest counter must give 0.
     *
     * This is not an artificial configuration for the sake of the test but a degenerate case of the
     * equation itself (a zero factor in front of the integral operator), reachable through
     * the regular public constructor.
     *
     * The statement is strengthened by a check that the coefficient vector did NOT SHIFT bitwise:
     * otherwise "zero steps" could also be reported after a step was actually made.
     */
    @Test
    fun instantConvergenceReportsZeroNewtonSteps() {
        val solver = solverFor(UrysonProblem.A, n = 8, tol = 1e-12, lambdaOverride = 0.0)
        val expectedCoeffs = thetaOf(solver)

        val newton = solver.solveBase()
        assertTrue(newton.converged, "At cL = 0 the base scheme must converge")
        assertEquals(
            0,
            newton.iterations,
            "The initial approximation is ALREADY the solution: zero Newton steps, got ${newton.iterations}",
        )
        assertEquals(0.0, newton.residual, "The residual at the exact solution must be zero")
        for (i in expectedCoeffs.indices) {
            assertEquals(
                expectedCoeffs[i].toRawBits(),
                newton.coeffs[i].toRawBits(),
                "Coefficient $i changed, hence a step was performed after all",
            )
        }

        // The Kulkarni scheme at cL = 0 comes to the same: G_K(c) = theta(f).
        val kulkarni = solver.kulkarni()
        assertEquals(
            0,
            kulkarni.iterations,
            "Kulkarni at cL = 0 makes no steps, got ${kulkarni.iterations}",
        )
        assertEquals(0.0, kulkarni.residual)

        // Nyström DELIBERATELY starts not from theta(f) but from the projection of a constant function,
        // so one step here MUST happen — and it is exactly one, not two.
        val nystrom = solver.nystrom()
        assertEquals(
            1,
            nystrom.iterations,
            "Nyström starts from the projection of one: exactly one step is needed, got ${nystrom.iterations}",
        )
    }

    /**
     * EXIT BY THE STEP CRITERION: the reported residual must refer to the RETURNED
     * point. This is checked by an independent recomputation.
     *
     * WHY THE COUNTER `stepExits` IS NEEDED. For the configurations that exited BY THE RESIDUAL
     * the statement is TAUTOLOGICAL: the reported number is that very residual, computed
     * by the same code at the same point — a match is guaranteed regardless of the fix.
     * The test becomes substantial ONLY on the configurations that exited BY THE STEP:
     * it is there that the old code returned the residual of the previous point. Therefore the test
     * counts such configurations and requires that there be more than zero of them — otherwise
     * it would silently degenerate into a check of an identity.
     *
     * The tolerances are chosen by measurement: at `tol = 1e-2` and `tol = 1e-10` the base scheme for
     * problem B exits exactly by the step norm (a cubic nonlinearity at `cL = 1`).
     */
    @Test
    fun reportedResidualIsMeasuredAtReturnedPoint() {
        var checked = 0
        var stepExits = 0
        for (tol in listOf(1e-2, 1e-5, 1e-10, 1e-12)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val label = "problem ${problem.name}, tol=$tol"
                val newton = solverFor(problem, n = 8, tol = tol).solveBase()
                val solver = solverFor(problem, n = 8, tol = tol)
                val recomputed = baseResidualAt(solver, newton.coeffs)
                val scale = maxOf(abs(recomputed), abs(newton.residual), 1e-300)
                assertTrue(
                    abs(newton.residual - recomputed) / scale < residualMatchTolerance,
                    "$label: the reported residual is ${newton.residual}, " +
                        "while at the RETURNED point it equals $recomputed",
                )
                checked++

                // We classify the exit criterion by reproducing the loop.
                val replay = replayBase(solverFor(problem, n = 8, tol = tol), tol)
                assertEquals(
                    newton.iterations,
                    replay.steps,
                    "$label: the reproduction of the loop diverged from solveBase in the number of steps",
                )
                if (replay.steps > 0 && replay.lastStepNorm < maxOf(tol, newtonToleranceFloor)) {
                    stepExits++
                }
            }
        }
        assertTrue(checked > 0, "The test degenerated: not a single configuration was checked")
        assertTrue(
            stepExits > 0,
            "The test degenerated: NOT A SINGLE configuration exited by the STEP criterion, and it is exactly this " +
                "case that is the subject of the fix — on an exit by the residual the statement is tautological. " +
                "Choose the tolerances anew (tol=1e-2 and tol=1e-10 on problem B are confirmed by measurement)",
        )
    }

    /**
     * THE TRUSTWORTHINESS OF THE `converged` FLAG: if convergence is declared, the residual MUST be
     * below the requested tolerance.
     *
     * This is not a consequence but a standalone contract ensured by the code: on an exit by
     * the step norm the residual is recomputed at the new point and COMPARED with the tolerance.
     * A small step by itself is not considered a success — it regularly arises with a poorly
     * conditioned Jacobian and with the approximate difference Jacobian of the `nystrom` scheme.
     *
     * Before the fix the contradiction was reproduced in fact on all three schemes (measurement,
     * n=8, problem B): the base one at `tol = 1e-2` reported `2.14e-2 > 1e-2`, the Kulkarni
     * scheme at `tol = 1e-5` `3.19e-5 > 1e-5`, Nyström at `tol = 1e-8`
     * `2.56e-8 > 1e-8`. All three cases are included in the iteration below.
     *
     * The counter `checked` is mandatory: without it a refusal of the schemes to converge (a `continue` on
     * every configuration) would zero the test, leaving it green.
     */
    @Test
    fun convergedResultNeverReportsResidualAboveTolerance() {
        var checked = 0
        for (tol in listOf(1e-2, 1e-5, 1e-8, 1e-10)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val solver = solverFor(problem, n = 8, tol = tol)
                val checks = listOf(
                    "base" to solver.base(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                )
                for ((name, solution) in checks) {
                    if (!solution.converged) continue
                    assertTrue(
                        solution.residual < tol,
                        "Problem ${problem.name}, tol=$tol, scheme $name: convergence is declared, " +
                            "but the reported residual ${solution.residual} is above the tolerance",
                    )
                    checked++
                }
            }
        }
        assertTrue(
            checked > 0,
            "The test degenerated: no scheme reported convergence, so the statement " +
                "was never checked",
        )
    }

    /**
     * THE COUNTER EQUALS THE NUMBER OF STEPS ACTUALLY PERFORMED — a comparison with an INDEPENDENT count.
     *
     * The independent count is the number of calls of the step computation lambda: it is called exactly
     * once per performed step, and its counter lives in the test, not in the helper.
     * Therefore the comparison does not depend on how the helper counts itself, and catches a divergence
     * even by one.
     *
     * WHY NOT MONOTONICITY IN `tol`. The former edition of this test checked that
     * tightening the tolerance does not decrease the number of steps, and claimed that it "catches a shift
     * of the counter by one". That was WRONG: the old semantics differs from the new one by a
     * UNIFORM shift by one, and monotonicity is invariant to a constant shift,
     * so the test did not catch the mutation by construction. A comparison with an independent count is
     * not invariant to the shift and catches it at once.
     */
    @Test
    fun stepCountEqualsNumberOfPerformedSteps() {
        var checked = 0
        var withSteps = 0
        for (tol in listOf(1e-1, 1e-2, 1e-5, 1e-10, 1e-12)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val label = "problem ${problem.name}, tol=$tol"
                val reported = solverFor(problem, n = 8, tol = tol).solveBase()
                val replay = replayBase(solverFor(problem, n = 8, tol = tol), tol)

                assertEquals(
                    replay.stepCalls,
                    replay.steps,
                    "$label: the helper reported ${replay.steps} steps, while the step computation was called " +
                        "${replay.stepCalls} times",
                )
                assertEquals(
                    replay.stepCalls,
                    reported.iterations,
                    "$label: solveBase reported ${reported.iterations} steps, while the step was actually " +
                        "performed ${replay.stepCalls} times",
                )
                checked++
                if (replay.stepCalls > 0) withSteps++
            }
        }
        assertTrue(checked > 0, "The test degenerated: not a single configuration was checked")
        assertTrue(
            withSteps > 0,
            "The test degenerated: in all the configurations the number of steps turned out to be zero, so the comparison " +
                "of the counters does not distinguish the semantics",
        )
    }

    /**
     * STALLING: a small step at a large residual is NOT declared a convergence.
     *
     * The regular problems A and B do not produce this regime — an iteration over 216 combinations
     * (problem × basis × grid × tolerance) gave ZERO cases. Therefore the regime is checked by
     * a direct call of [runNewtonIterations] with degenerate `F` and step: the residual is kept
     * equal to `5.0`, and the step to `1e-15`. Such a substitution does not replace the physics of the problem
     * but isolates exactly the control decision that is checked here.
     *
     * It is exactly this case that separates a contract ensured by the code from an empirical
     * observation: without the comparison with the tolerance `converged` would be `true` at a residual of `5.0`.
     */
    @Test
    fun negligibleStepWithLargeResidualIsNotConvergence() {
        val x = doubleArrayOf(1.0)
        val run = runNewtonIterations(
            x = x,
            maxSteps = 100,
            tolerance = 1e-8,
            residualAt = { doubleArrayOf(5.0) },
            stepAt = { _, _ -> doubleArrayOf(1e-15) },
        )
        assertFalse(
            run.converged,
            "The step is negligibly small, but the residual 5.0 is above the tolerance 1e-8: this is NOT a convergence",
        )
        assertTrue(run.stalled, "The case must be marked as STALLING, and not as an exhaustion of the limit")
        assertEquals(1, run.performedSteps, "Stalling is recognized right after the very first step")
        assertEquals(5.0, run.residual, "The residual reported must be the actual one")
        assertTrue(run.performedSteps < 100, "The iteration limit is NOT exhausted — the run was interrupted early")

        // A control in the opposite direction: the same exit by the step, but the residual is SMALL — a success.
        val converging = doubleArrayOf(10.0)
        val ok = runNewtonIterations(
            x = converging,
            maxSteps = 100,
            tolerance = 1e-8,
            residualAt = { current -> doubleArrayOf(abs(current[0])) },
            stepAt = { current, _ -> doubleArrayOf(-current[0]) },
        )
        assertTrue(ok.converged, "A small step at a ZERO residual must be considered a convergence")
        assertFalse(ok.stalled, "A successful outcome is not a stalling")
        assertEquals(0.0, ok.residual, "The residual must be measured at the returned point")
    }
}
