package solvers.core

import numerics.GaussLegendre
import numerics.NumericsContext
import numerics.backend.Backends
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.uryson.UrysonProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * WIRING of [NumericsContext] through the solvers.
 *
 * Motivation. The context passes through five classes ([solvers.core.SecondKindSolverCore],
 * both second-kind solvers, `UrysonSolver`/`CollocationCore`/`SplineSpace`,
 * [splines.functionals.FunctionalFamily]), and before these tests NO test
 * built a solver with a NON-default context. Hence a wiring error — for example, if
 * a `NumericsContext.default()` were left somewhere inside instead of the passed value —
 * would not show up at all: every run used one and the same default backend.
 *
 * The problem is chosen DELIBERATELY well-conditioned (Fredholm of the second kind, F2:
 * `K(t,s) = 1/(1+t+s)`, smooth solution). On equations of the FIRST kind a backend change
 * routinely gives a discrepancy up to 5.7e-2 because of poor conditioning — there a comparison
 * of backends would test the conditioning of the problem rather than the context wiring.
 *
 * The contract of [NumericsContext] itself as a value (a shared default instance,
 * equality by value) is checked in the `numerical-core` library (`NumericsContextTest`).
 */
@Tag("fast")
class NumericsContextWiringTest {

    private val problem = FredholmProblem.F2

    /** Builds the F2 solver entirely in ONE context (the functional family included). */
    private fun solver(ctx: NumericsContext, n: Int = 16): FredholmSecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis, ctx)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        return FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            ctx = ctx,
        )
    }

    /** Comparison points: nodes and interior points, including both ends of the interval. */
    private val samplePoints = doubleArrayOf(0.0, 0.13, 0.37, 0.5, 0.71, 0.99, 1.0)

    /**
     * The solver on the reference backend agrees with the solver on multik.
     *
     * This is the very check that the passed backend REACHES every place where a linear system is
     * solved: if the context were lost somewhere, both branches would run on one backend and the
     * test would pass trivially — hence it is checked separately below that the backends DIFFER
     * and that the results are then NOT bitwise equal (that is, a different LU path is really used).
     */
    @Test
    fun solverAgreesAcrossBackends() {
        val multik = solver(NumericsContext(backend = Backends.native())).base()
        val reference = solver(NumericsContext(backend = Backends.java())).base()
        for (t in samplePoints) {
            val a = multik.eval(t)
            val b = reference.eval(t)
            assertTrue(
                kotlin.math.abs(a - b) <= 1e-9,
                "t=$t: multik=$a, reference=$b, |difference|=${kotlin.math.abs(a - b)} > 1e-9",
            )
        }
    }

    /**
     * The context REALLY reaches the backend: two backends give a NUMERICALLY DIFFERENT (though
     * consistent) result in at least one point.
     *
     * Without this check the previous test would be self-confirming: if the wiring were
     * broken and both solvers ran on the default backend, the discrepancy would be EXACTLY
     * zero and the tolerance 1e-9 would hold automatically. Here the opposite is fixed:
     * the computation paths differ, hence the `backend` parameter is indeed used.
     */
    @Test
    fun differentBackendsTakeDifferentComputationPaths() {
        val multik = solver(NumericsContext(backend = Backends.native())).base()
        val reference = solver(NumericsContext(backend = Backends.java())).base()
        val anyBitwiseDifference = samplePoints.any { t -> multik.eval(t) != reference.eval(t) }
        assertTrue(
            anyBitwiseDifference,
            "Both backends produced a BITWISE identical result at every point. Either the context " +
                "wiring is broken (both solvers run on one backend), or the LU implementations coincide.",
        )
    }

    /**
     * `parallel = false` and `parallel = true` give a BITWISE identical solution.
     *
     * For [ParallelAssembly] itself this is already proved directly, but here the property
     * is checked THROUGH the solver: the assembly of the matrix M goes via `ctx.parallel`, and if
     * the parallel path changed the accumulation order, the numbers would differ in the low bits.
     */
    @Test
    fun parallelFlagDoesNotChangeResultBitwise() {
        val sequential = solver(NumericsContext(parallel = false)).base()
        val parallel = solver(NumericsContext(parallel = true)).base()
        for (t in samplePoints) {
            val s = sequential.eval(t)
            val p = parallel.eval(t)
            assertTrue(s == p, "t=$t: seq=$s, par=$p — the assembly must be bitwise identical")
        }
    }



    /**
     * NEGATIVE: a solver and a functional family with DIFFERENT contexts — a loud failure.
     *
     * Exactly the scenario the validation was added for: the family solves its tiny
     * linear systems on one backend and the solver on another, so parts of ONE problem are
     * computed by different LU implementations. Previously this passed silently.
     */
    @Test
    fun mismatchedFunctionalsContextFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        // The family is built on reference, the solver is asked to compute on multik.
        val funcs = ProjFunctionals(basis, NumericsContext(backend = Backends.java()))
        val ex = assertFailsWith<IllegalArgumentException> {
            FredholmSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives({ t -> problem.rhsExact(t, op) }, { t -> problem.rhsExactDeriv(t, op) }),
                ctx = NumericsContext(backend = Backends.native()),
            )
        }
        val message = ex.message!!
        assertTrue(message.contains("funcs"), "The message must name the dependency: $message")
        assertTrue(
            message.contains(Backends.java().name) && message.contains(Backends.native().name),
            "The message must name BOTH backends so that the discrepancy is visible: $message",
        )
    }

    /** A mismatch in the `parallel` flag is rejected too: the context is compared as a whole. */
    @Test
    fun mismatchedParallelFlagFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val funcs = ProjFunctionals(basis, NumericsContext(parallel = false))
        assertFailsWith<IllegalArgumentException> {
            FredholmSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives({ t -> problem.rhsExact(t, op) }, { t -> problem.rhsExactDeriv(t, op) }),
                ctx = NumericsContext(parallel = true),
            )
        }
    }

    /** Consistent contexts (including default ones for all participants) pass. */
    @Test
    fun matchingContextsAreAccepted() {
        for (ctx in listOf(
            NumericsContext.default(),
            NumericsContext(backend = Backends.java()),
            NumericsContext(backend = Backends.native(), parallel = false),
        )) {
            val solution = solver(ctx, n = 8).base()
            assertTrue(solution.eval(0.5).isFinite(), "ctx=${ctx.describe()}: the solution must be a number")
        }
    }

    // ==== Uryson: it has TWO dependencies carrying a context — funcs AND space =====

    /**
     * The Uryson second-kind solver is consistent across backends.
     *
     * Separately from Fredholm: Uryson has its own wiring branch — through `CollocationCore`
     * (the Jacobian and the Newton step) and `SplineSpace`.
     */
    @Test
    fun urysonSolverAgreesAcrossBackends() {
        fun solve(ctx: NumericsContext): (Double) -> Double {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = ProjFunctionals(basis, ctx)
            val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8), ctx)
            val op = solvers.uryson.UrysohnOperator(UrysonProblem.A.kernel, grid, GaussLegendre(8))
            return problems.uryson.secondKindSolver(UrysonProblem.A, basis, funcs, space, op, ctx = ctx)
                .base().eval
        }
        val multik = solve(NumericsContext(backend = Backends.native()))
        val reference = solve(NumericsContext(backend = Backends.java()))
        for (t in samplePoints) {
            val a = multik(t)
            val b = reference(t)
            assertTrue(
                kotlin.math.abs(a - b) <= 1e-9,
                "Uryson t=$t: multik=$a, reference=$b, |difference|=${kotlin.math.abs(a - b)} > 1e-9",
            )
        }
    }

    /**
     * NEGATIVE for `space`: exactly this discrepancy was found by the review.
     *
     * `UrysonFirstKindSolver.solveMorozov` computes the stabilizer `Omega` via
     * `space.ctx.backend`, and the Gauss-Newton system via its own `ctx.backend`.
     * Before the validation, two parts of one Morozov criterion could be computed by different LUs.
     */
    @Test
    fun mismatchedSplineSpaceContextFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val ctx = NumericsContext(backend = Backends.native())
        val funcs = ProjFunctionals(basis, ctx)
        // space is built on a DIFFERENT backend than the solver and the family.
        val space = solvers.uryson.SplineSpace(
            basis, GaussLegendre(8), NumericsContext(backend = Backends.java()),
        )
        val op = solvers.uryson.UrysohnOperator(UrysonProblem.A.kernel, grid, GaussLegendre(8))
        val ex = assertFailsWith<IllegalArgumentException> {
            solvers.uryson.UrysonFirstKindSolver(basis, funcs, space, op, ctx = ctx)
        }
        assertTrue(ex.message!!.contains("space"), "The message must name 'space': ${ex.message}")
    }
}
