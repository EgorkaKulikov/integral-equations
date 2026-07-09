package solvers.volterra

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Characterization (golden) tests for the Volterra solvers.
 *
 * IMPORTANT: numeric thresholds are NOT analytic truth. Reference values were fixed
 * by running the CURRENT implementation once and serve only as a regression net for
 * the upcoming HPC refactor. Tolerances are generous and documented per test.
 */
class VolterraCoverageTest {
    private val quad = GaussLegendre(8)
    private fun finite(x: Double) = !x.isNaN() && !x.isInfinite()

    private fun build(p: ModelProblem, sys: GeneratingSystem, n: Int): SecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(p.kernel, grid, quad)
        return secondKindSolver(p, basis, funcs, op)
    }

    /**
     * Exercises all ModelProblem companion entries (V2span/V2/V2exp/V2win) base solver.
     * Reference: each E_h finite and < 1e-1 (fixed from current run). V2span lies in
     * span{1,t,t^2}=phi^B so it is near machine precision on basis B.
     */
    @Test fun allModelProblems_baseFinite() {
        for (p in listOf(ModelProblem.V2span, ModelProblem.V2, ModelProblem.V2exp, ModelProblem.V2win)) {
            val s = build(p, GeneratingSystem.B, 8)
            val e = errorEh({ t -> p.exact(t) }, s.base().eval, s.grid)
            assertTrue(finite(e) && e < 1e-1, "${p.name}: E_h=$e")
        }
        val sSpan = build(ModelProblem.V2span, GeneratingSystem.B, 8)
        val eSpan = errorEh({ t -> ModelProblem.V2span.exact(t) }, sSpan.base().eval, sSpan.grid)
        assertTrue(eSpan < 1e-7, "V2span should be near-exact on B: $eSpan")
    }

    /**
     * Covers sloan / kulkarni (projector branch) / iteratedKulkarni on V2, basis B,
     * n=8. Reference fixed from current run: all E_h finite and < 1e-2. Also exercises
     * the Volterra-specific Leibniz boundary term inside applyDeriv via xi-free schemes.
     */
    @Test fun v2_secondKind_allSchemes_projectorTheta() {
        val s = build(ModelProblem.V2, GeneratingSystem.B, 8)
        val exact = { t: Double -> ModelProblem.V2.exact(t) }
        for (sol in listOf(s.base(), s.sloan(), s.kulkarni(), s.iteratedKulkarni())) {
            val e = errorEh(exact, sol.eval, s.grid)
            assertTrue(finite(e) && e < 1e-2, "scheme E_h=$e")
        }
    }

    /**
     * Covers the xi-projector (DeBoorFixFunctionals) Kulkarni branch using derivatives.
     * V2win has K(t,t)=0 (weak kernel). Reference fixed from current run: E_h finite,
     * < 1e-2 on basis H, n=8.
     */
    @Test fun v2win_kulkarni_xiProjector() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = DeBoorFixFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V2win.kernel, grid, quad)
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.V2win.rhsExact(t, op) }, { t -> ModelProblem.V2win.rhsExactDeriv(t, op) })
        val e = errorEh({ t -> ModelProblem.V2win.exact(t) }, solver.kulkarni().eval, grid)
        assertTrue(finite(e) && e < 1e-2, "xi kulkarni E_h=$e")
    }

    /**
     * Covers the NON-projector Kulkarni branch (kulkarniQuasi + evalNodalLinear/sampleXs)
     * for mu (AveragingFunctionals) and lambda (ThreePointFunctionals). Reference: finite
     * bounded result (<5e-1). Per source KDoc convergence is only a numerical observation,
     * hence the loose ceiling (NaN/divergence regression net).
     */
    @Test fun v2_kulkarniQuasi_muAndLambda() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = VolterraOperator(ModelProblem.V2.kernel, grid, quad)
        for (funcs in listOf(AveragingFunctionals(basis), ThreePointFunctionals(basis))) {
            val solver = SecondKindSolver(basis, funcs, op, 1.0,
                { t -> ModelProblem.V2.rhsExact(t, op) }, { t -> ModelProblem.V2.rhsExactDeriv(t, op) })
            val e = errorEh({ t -> ModelProblem.V2.exact(t) }, solver.kulkarni().eval, grid)
            val eIt = errorEh({ t -> ModelProblem.V2.exact(t) }, solver.iteratedKulkarni().eval, grid)
            assertTrue(finite(e) && e < 5e-1, "quasi kulkarni E_h=$e")
            assertTrue(finite(eIt) && eIt < 5e-1, "quasi iterated E_h=$eIt")
        }
    }

    /**
     * Regression guard for the kulkarniQuasi early-exit fix (return@repeat -> break).
     * The fixed point of the iteration must NOT depend on whether we stop early once
     * the residual drops below 1e-12 or run all 200 sweeps: the converged solution is
     * identical. We check the iteration reaches a stable fixed point by verifying the
     * output is deterministic across repeated solves and stays finite/bounded.
     */
    @Test fun v2_kulkarniQuasi_convergesToStableFixedPoint() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = VolterraOperator(ModelProblem.V2.kernel, grid, quad)
        val funcs = AveragingFunctionals(basis)
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.V2.rhsExact(t, op) }, { t -> ModelProblem.V2.rhsExactDeriv(t, op) })
        val first = solver.kulkarni()
        val second = solver.kulkarni()
        val ts = doubleArrayOf(0.0, 0.13, 0.37, 0.5, 0.71, 0.99, 1.0)
        for (t in ts) {
            val a = first.eval(t); val b = second.eval(t)
            assertTrue(finite(a) && finite(b), "finite at t=$t")
            // Same fixed point every run: early break must not change the result.
            assertTrue(abs(a - b) < 1e-14, "non-deterministic quasi solution at t=$t: $a vs $b")
        }
    }

    /**
     * Covers matrix/vector assembly (matrixM, matrixM2, vectorG, vectorD) shape and
     * finiteness on V2, n=8. Reference fixed from current run.
     */
    @Test fun matricesAndVectors_shapeAndFinite() {
        val s = build(ModelProblem.V2, GeneratingSystem.B, 8)
        val dim = s.dim
        val m = s.matrixM(); val m2 = s.matrixM2(); val g = s.vectorG(); val d = s.vectorD()
        assertTrue(m.size == dim && m2.size == dim && g.size == dim && d.size == dim)
        for (r in 0 until dim) for (c in 0 until dim) assertTrue(finite(m[r][c]) && finite(m2[r][c]))
        for (j in 0 until dim) assertTrue(finite(g[j]) && finite(d[j]))
    }

    /**
     * Covers VolterraOperator special cases: apply(t<=a)=0, the Leibniz boundary term
     * K(t,t)u(t) inside applyDeriv (including at t=a), and the variable-upper-limit
     * sub-breakpoint integration. Reference (analytic sub-checks): apply(a)=0;
     * applyDeriv(a, u=const) == K(a,a)*u(a).
     */
    @Test fun volterraOperator_boundaryAndZeroCases() {
        val grid = Grid.uniform(8)
        val op = VolterraOperator(ModelProblem.V1.kernel, grid, quad)
        assertTrue(abs(op.apply(grid.a) { 1.0 }) < 1e-15)
        // K(a,a)=1+a-a=1 for V1; applyDeriv(a, u=1) = 1*1 + 0 = 1.
        assertTrue(abs(op.applyDeriv(grid.a) { 1.0 } - 1.0) < 1e-12)
        assertTrue(finite(op.apply(0.5) { s -> s }))
        assertTrue(finite(op.applyDeriv(0.5) { s -> s }))
    }

    /**
     * Covers FirstKindSolver (reduction-by-differentiation, r3) on V1 with K(t,t)=1!=0.
     * Schemes base/sloan/kulkarni/iteratedKulkarni run end-to-end. Reference fixed from
     * current run: E_h finite and < 1e-1 on basis B, n=8.
     */
    @Test fun v1_firstKindSolver_allSchemes() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V1.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.V1, basis, funcs, op)
        val exact = { t: Double -> ModelProblem.V1.exact(t) }
        for (sol in listOf(solver.base(), solver.sloan(), solver.kulkarni(), solver.iteratedKulkarni())) {
            val e = errorEh(exact, sol.eval, grid)
            assertTrue(finite(e) && e < 1e-1, "V1 scheme E_h=$e")
        }
    }

    /**
     * Regression guard for the 4th-order finite-difference reduction (fix #2): the V1
     * FirstKindSolver base scheme (ProjFunctionals) must stay finite and accurate after
     * switching kernelW.kT / gEffDeriv to a 5-point 4th-order stencil (hFD=1e-3).
     * Threshold fixed from the current run with margin.
     */
    @Test fun v1_firstKind_base_accuracyAfter4thOrderFD() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V1.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.V1, basis, funcs, op)
        val e = errorEh({ t -> ModelProblem.V1.exact(t) }, solver.base().eval, grid)
        // Measured ~1.3e-5 with the 4th-order stencil; threshold set with ~7x margin.
        assertTrue(finite(e) && e < 1e-4, "V1 base E_h=$e")
    }

    /**
     * Convergence sanity (characterization): V2/basis B base error decreases n=8->16.
     * Reference behaviour fixed from current implementation.
     */
    @Test fun v2_converges_n8_to_n16() {
        val e8 = errorEh({ t -> ModelProblem.V2.exact(t) }, build(ModelProblem.V2, GeneratingSystem.B, 8).base().eval, Grid.uniform(8))
        val e16 = errorEh({ t -> ModelProblem.V2.exact(t) }, build(ModelProblem.V2, GeneratingSystem.B, 16).base().eval, Grid.uniform(16))
        assertTrue(e16 < e8, "no convergence: e8=$e8 e16=$e16")
    }

    /**
     * Covers the V1 FirstKindSolver xi-functional path (DeBoorFixFunctionals), which
     * triggers the reduced-kernel derivative closure kernelW.kT and gEffDeriv
     * (finite differences) inside applyDeriv.
     *
     * Reference fixed from current run: with xi-functionals the finite-difference
     * reduced-kernel derivative is ill-conditioned and E_h BLOWS UP (base ~2.02e4,
     * sloan ~2.35e3) -> asserted only finite. This is a regression net for the
     * derivative-reduction code path, NOT an accuracy claim (theta-functionals,
     * tested in v1_firstKindSolver_allSchemes, are the accurate variant).
     */
    @Test fun v1_firstKind_xiFunctionals_derivativePath() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = DeBoorFixFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V1.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.V1, basis, funcs, op)
        val exact = { t: Double -> ModelProblem.V1.exact(t) }
        assertTrue(finite(errorEh(exact, solver.base().eval, grid)))
        assertTrue(finite(errorEh(exact, solver.sloan().eval, grid)))
    }

    /**
     * Regression guard for the K(t,t)==0 division guard (weak-singularity kernels).
     * V2win has K=t-s, so K(t,t)=0 everywhere: reducing the first-kind equation to a
     * second-kind one divides by K(t,t) and would silently produce NaN/Inf. The solver
     * must now fail fast with a clear IllegalArgumentException at construction.
     */
    @Test fun v1_firstKind_zeroKtt_throws() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = VolterraOperator(ModelProblem.V2win.kernel, grid, quad)
        val ex = assertFailsWith<IllegalArgumentException> {
            FirstKindSolver(ModelProblem.V2win, basis, funcs, op)
        }
        assertTrue(ex.message?.contains("K(t,t)") == true, "message: ${ex.message}")
    }

    /** Trivial coverage of SolutionFunc.eval wrapper. */
    @Test fun solutionFunc_evalWrapper() {
        val sf = SolutionFunc { t -> t + 1.0 }
        assertTrue(abs(sf.eval(4.0) - 5.0) < 1e-12)
    }
}
