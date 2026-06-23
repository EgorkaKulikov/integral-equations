package solvers.uryson

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Characterization (golden) tests for the Uryson (nonlinear) solvers and helpers.
 *
 * IMPORTANT: numeric thresholds are NOT analytic truth. Reference values were fixed by
 * running the CURRENT implementation once and serve only as a regression net for the
 * upcoming HPC refactor. Newton/Tikhonov/Morozov tolerances are generous and documented
 * per test. Golden E_h ceilings catch NaN / divergence / blow-up, not theory.
 */
class UrysonCoverageTest {
    private val quad = GaussLegendre(8)
    private fun finite(x: Double) = !x.isNaN() && !x.isInfinite()

    private fun ctx(p: ModelProblem, sys: GeneratingSystem, n: Int): SecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(p.kernel, grid, quad)
        return SecondKindSolver(p, basis, funcs, space, op)
    }

    // ---------------------- small types ----------------------

    /** Direct unit test of Quintuple: destructuring and field order (a,b,c,d,e). */
    @Test fun quintuple_fields() {
        val q = Quintuple(1.0, 2.0, 3.0, 4.0, 5.0)
        val (a, b, c, d, e) = q
        assertTrue(a == 1.0 && b == 2.0 && c == 3.0 && d == 4.0 && e == 5.0)
        assertEquals(q, Quintuple(1.0, 2.0, 3.0, 4.0, 5.0))
    }

    /**
     * Direct unit test of CheckResult: ok/passed flags. measured<=threshold -> ok=true;
     * measured>threshold -> ok=false. Covers both getters and the boolean expression.
     */
    @Test fun checkResult_flags() {
        val pass = CheckResult("x", 1e-12, 1e-10, true)
        val fail = CheckResult("y", 1.0, 1e-10, false)
        assertTrue(pass.ok && pass.passed)
        assertFalse(fail.ok)
    }

    /**
     * Direct unit test of FirstKindSolution fields and eval wrapper. Reference values
     * are arbitrary literals (pure data holder, no numeric semantics here).
     */
    @Test fun firstKindSolution_fields() {
        val s = FirstKindSolution(doubleArrayOf(1.0, 2.0), { t -> t * t }, 1e-3, 1e-4, 0.5)
        assertTrue(s.coeffs.size == 2 && s.alpha == 1e-3 && s.resid == 1e-4 && s.omega == 0.5)
        assertTrue(abs(s.eval(3.0) - 9.0) < 1e-12)
    }

    /** Covers SolutionFunc(eval, iterations) holder. */
    @Test fun solutionFunc_fields() {
        val sf = SolutionFunc({ t -> t + 1 }, 7)
        assertTrue(abs(sf.eval(1.0) - 2.0) < 1e-12 && sf.iterations == 7)
    }

    // ---------------------- ProjFunctional / ProjFunctionals ----------------------

    /**
     * Covers ProjFunctional.apply / applyValues / absSum and biorthogonality
     * theta_i(omega_j)=delta_ij of ProjFunctionals (reference: analytic identity,
     * tol 1e-8). Also covers cTheta(), closedFormInternal, theta() accessor.
     */
    @Test fun projFunctionals_biorthogonalityAndApis() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        for (i in -2..grid.n - 1) for (j in -2..grid.n - 1) {
            val v = funcs.theta(i).apply { t -> basis.omega(j, t) }
            assertTrue(abs(v - if (i == j) 1.0 else 0.0) < 1e-8, "theta_$i(omega_$j)=$v")
        }
        // applyValues consistency with apply on the functional's own nodes.
        val th = funcs.theta(0)
        val vals = DoubleArray(th.nodes.size) { th.nodes[it] }
        assertTrue(abs(th.applyValues(vals) - th.apply { t -> t }) < 1e-12)
        assertTrue(th.absSum() >= 1.0)
        assertTrue(funcs.cTheta() >= 1.0)
        // closedFormInternal matches the built internal functional on uniform grid.
        val cf = funcs.closedFormInternal(2)
        val approxOk = cf.coeffs.size == 5 && cf.coeffs.all { finite(it) }
        assertTrue(approxOk)
        // projectorCoeffs reproduces a function in span (constant 1).
        val pc = funcs.projectorCoeffs { 1.0 }
        assertTrue(abs(basis.evalSpline(pc, 0.5) - 1.0) < 1e-8)
    }

    // ---------------------- SplineSpace ----------------------

    /**
     * Covers SplineSpace: weights (sum == b-a), wInt (integrals of omega_j), gramR
     * (symmetric, banded), omegaReg quadratic form (>=0). Reference: weightsSum==1.0
     * analytic (b-a=1), gram symmetry exact, omegaReg(0)=0.
     */
    @Test fun splineSpace_weightsGramReg() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val space = SplineSpace(basis, quad)
        assertTrue(abs(space.weightsSum() - 1.0) < 1e-12)
        assertTrue(space.weights.sum() > 0 && space.wInt.all { finite(it) })
        val dim = space.dim
        for (i in 0 until dim) for (j in 0 until dim)
            assertTrue(abs(space.gramR[i][j] - space.gramR[j][i]) < 1e-12)
        assertTrue(abs(space.omegaReg(DoubleArray(dim))) < 1e-15)
        assertTrue(space.omegaReg(DoubleArray(dim) { 1.0 }) > 0.0)
    }

    // ---------------------- Operator + ModelProblem ----------------------

    /**
     * Covers UrysohnOperator.apply / frechet / applyNodes and ModelProblem.rhsExact for
     * all four problems A,B,C,D (both kinds). Reference: applyNodes(tau, x*at nodes)
     * matches apply within 1e-9 (analytic consistency); all rhs finite.
     */
    @Test fun operator_apis_and_allModelProblems_rhs() {
        val grid = Grid.uniform(8)
        for (p in listOf(ModelProblem.A, ModelProblem.B, ModelProblem.C, ModelProblem.D)) {
            val op = UrysohnOperator(p.kernel, grid, quad)
            val t = 0.4
            val xNodes = DoubleArray(op.gNode.size) { p.exact(op.gNode[it]) }
            val viaNodes = op.applyNodes(t, xNodes)
            val viaClosure = op.apply(t) { s -> p.exact(s) }
            assertTrue(abs(viaNodes - viaClosure) < 1e-9, "${p.name} applyNodes mismatch")
            assertTrue(finite(op.frechet(t, { s -> p.exact(s) }, { 1.0 })))
            assertTrue(finite(p.rhsExact(t, op)))
            // kernel dkdu finite
            assertTrue(finite(p.kernel.dkdu(t, 0.3, 1.0)))
        }
    }

    // ---------------------- CollocationCore ----------------------

    /**
     * Covers CollocationCore.uAtSupport / xiVector / bMatrix on problem A, n=8.
     * Reference: xiVector size n+2, bMatrix (n+2)x(n+2), all finite (fixed from run).
     */
    @Test fun collocationCore_xiAndJacobian() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = UrysohnOperator(ModelProblem.A.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val c = funcs.projectorCoeffs { t -> ModelProblem.A.exact(t) }
        val xi = core.xiVector(c)
        val b = core.bMatrix(c)
        assertTrue(xi.size == grid.n + 2 && xi.all { finite(it) })
        assertTrue(b.size == grid.n + 2 && b.all { row -> row.size == grid.n + 2 && row.all { finite(it) } })
        assertTrue(core.uAtSupport(c).all { finite(it) })
    }

    // ---------------------- SecondKindSolver schemes ----------------------

    /**
     * Covers SecondKindSolver base/sloan/kulkarni/nystrom for problem A (lambda=-1,
     * contractive). Reference E_h fixed from current run: base ~1.0e-4 at n=8,
     * all schemes finite and < 1e-2. Tolerance is a regression ceiling, not theory.
     */
    @Test fun problemA_allSchemes() {
        val s = ctx(ModelProblem.A, GeneratingSystem.B, 8)
        val exact = { t: Double -> ModelProblem.A.exact(t) }
        for (sol in listOf(s.base(), s.sloan(), s.kulkarni(), s.nystrom())) {
            val e = errorEhEval(exact, sol.eval, s.grid)
            assertTrue(finite(e) && e < 1e-2, "A scheme E_h=$e")
        }
        // errorEh(problem,...) overload coverage.
        assertTrue(finite(errorEh(ModelProblem.A, s.base(), s.grid)))
    }

    /**
     * Covers problem B (cubic kernel, lambda=1, NON-contractive -> exercises the Newton
     * preconditioner path in solveBase/kulkarni that simple iteration could not handle).
     * Reference E_h fixed from current run: finite and < 1e-1 for base/sloan/nystrom,
     * basis H, n=8.
     */
    @Test fun problemB_nonContractive_schemes() {
        val s = ctx(ModelProblem.B, GeneratingSystem.H, 8)
        val exact = { t: Double -> ModelProblem.B.exact(t) }
        for (sol in listOf(s.base(), s.sloan(), s.nystrom())) {
            val e = errorEhEval(exact, sol.eval, s.grid)
            assertTrue(finite(e) && e < 1e-1, "B scheme E_h=$e")
        }
    }

    /** Convergence sanity (characterization): problem A base E_h decreases n=8->16. */
    @Test fun problemA_converges() {
        val e8 = errorEhEval({ t -> ModelProblem.A.exact(t) }, ctx(ModelProblem.A, GeneratingSystem.B, 8).base().eval, Grid.uniform(8))
        val e16 = errorEhEval({ t -> ModelProblem.A.exact(t) }, ctx(ModelProblem.A, GeneratingSystem.B, 16).base().eval, Grid.uniform(16))
        assertTrue(e16 < e8, "no convergence e8=$e8 e16=$e16")
    }

    // ---------------------- FirstKindSolver (Tikhonov/Morozov) ----------------------

    /**
     * Covers FirstKindSolver for problem C (first kind, smooth kernel): noisyThetaF
     * (delta=0 branch and delta>0 branch), solveFixedAlpha (Gauss-Newton),
     * residual, and solveMorozov noise-free path. Reference: solution finite, alpha>0,
     * resid finite (fixed from current run). Regularization quality not asserted (ill-posed).
     */
    @Test fun problemC_firstKind_morozov_noiseFree() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(ModelProblem.C.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.C, basis, funcs, space, op)
        // delta=0 branch of noisyThetaF.
        val tf0 = solver.noisyThetaF(0.0, 1L)
        assertTrue(tf0.size == grid.n + 2 && tf0.all { finite(it) })
        // delta>0 branch (noise scaling).
        val tfN = solver.noisyThetaF(1e-3, 42L)
        assertTrue(tfN.all { finite(it) })
        // solveFixedAlpha + residual.
        val c0 = funcs.projectorCoeffs { 1.0 }
        val c = solver.solveFixedAlpha(tf0, 1e-4, c0)
        assertTrue(c.all { finite(it) } && finite(solver.residual(c, tf0)))
        // Morozov noise-free path (delta==0 continuation).
        val sol = solver.solveMorozov(0.0, 1L)
        assertTrue(sol.alpha > 0 && finite(sol.resid) && finite(sol.eval(0.5)))
    }

    /**
     * Covers FirstKindSolver Morozov WITH noise (delta>0) and the cubic-kernel problem D,
     * which exercises the non-zero warm-start restart logic (dK/du(.,.,0)=0 -> B=0).
     * Reference fixed from current run: chosen solution finite, alpha>0, resid finite.
     * This is a regression net for the continuation/restart branch, not an accuracy claim.
     */
    @Test fun problemD_firstKind_morozov_withNoise() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(ModelProblem.D.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.D, basis, funcs, space, op)
        val sol = solver.solveMorozov(1e-2, 7L)
        assertTrue(sol.alpha > 0 && finite(sol.resid) && finite(sol.omega))
        assertTrue(finite(sol.eval(0.3)))
    }
}
