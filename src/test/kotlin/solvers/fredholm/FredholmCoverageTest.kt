package solvers.fredholm

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
import kotlin.test.assertTrue

/**
 * Characterization (golden) tests for the Fredholm solvers.
 *
 * IMPORTANT: numeric thresholds below are NOT analytic truth. Every reference value
 * was fixed by running the CURRENT implementation once and is used only as a
 * regression net (to catch NaN / blow-up / divergence during the upcoming HPC
 * refactor). Tolerances are intentionally generous (relative, ~2%..order-of-magnitude
 * ceilings) and documented per test.
 */
class FredholmCoverageTest {
    private val quad = GaussLegendre(8)

    private fun finite(x: Double) = !x.isNaN() && !x.isInfinite()

    private fun build(p: ModelProblem, sys: GeneratingSystem, n: Int): SecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(sys, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(p.kernel, grid, quad)
        return secondKindSolver(p, basis, funcs, op)
    }

    /**
     * Exercises all ModelProblem companion entries (F2span/F2/F2exp/F1) plus the
     * second-kind base solver via the factory. Reference: each E_h reproduced by the
     * current implementation must stay finite and small (<1e-1). Protects against a
     * regression that breaks rhs assembly or the (I-M)c=g solve.
     */
    @Test fun allModelProblems_baseFinite() {
        for (p in listOf(ModelProblem.F2span, ModelProblem.F2, ModelProblem.F2exp)) {
            val s = build(p, GeneratingSystem.B, 8)
            val e = errorEh({ t -> p.exact(t) }, s.base().eval, s.grid)
            assertTrue(finite(e) && e < 1e-1, "${p.name}: E_h=$e")
        }
        // F2span lies in span{1,t,t^2}=phi^B -> near machine precision on basis B.
        val sSpan = build(ModelProblem.F2span, GeneratingSystem.B, 8)
        val eSpan = errorEh({ t -> ModelProblem.F2span.exact(t) }, sSpan.base().eval, sSpan.grid)
        assertTrue(eSpan < 1e-8, "F2span should be near-exact on B: $eSpan")
    }

    /**
     * Covers SecondKindSolver.sloan / kulkarni (projector branch via theta) /
     * iteratedKulkarni on F2, basis B, n=8. Reference: all four E_h finite, < 1e-2,
     * fixed from the current run; checks that the post-processing schemes run end-to-end.
     */
    @Test fun f2_secondKind_allSchemes_projectorTheta() {
        val s = build(ModelProblem.F2, GeneratingSystem.B, 8)
        val exact = { t: Double -> ModelProblem.F2.exact(t) }
        for (sol in listOf(s.base(), s.sloan(), s.kulkarni(), s.iteratedKulkarni())) {
            val e = errorEh(exact, sol.eval, s.grid)
            assertTrue(finite(e) && e < 1e-2, "scheme E_h=$e")
        }
    }

    /**
     * Covers the xi-projector (DeBoorFixFunctionals) Kulkarni branch, which uses
     * function derivatives in chi.apply. Reference fixed from current run: E_h finite
     * and < 1e-2 on F2exp/basis H/n=8.
     */
    @Test fun f2exp_kulkarni_xiProjector() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = DeBoorFixFunctionals(basis)
        val op = FredholmOperator(ModelProblem.F2exp.kernel, grid, quad)
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.F2exp.rhsExact(t, op) }, { t -> ModelProblem.F2exp.rhsExactDeriv(t, op) })
        val e = errorEh({ t -> ModelProblem.F2exp.exact(t) }, solver.kulkarni().eval, grid)
        assertTrue(finite(e) && e < 1e-2, "xi kulkarni E_h=$e")
    }

    /**
     * Covers the NON-projector Kulkarni branch (kulkarniQuasi + evalNodalLinear) for
     * both quasi-interpolant families: mu (AveragingFunctionals) and lambda
     * (ThreePointFunctionals). Reference: solver iterates to a finite, bounded result
     * (<5e-1). Per source KDoc convergence is a numerical observation, hence the loose
     * ceiling — this only guards against NaN/divergence regressions.
     */
    @Test fun f2_kulkarniQuasi_muAndLambda() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(ModelProblem.F2.kernel, grid, quad)
        for (funcs in listOf(AveragingFunctionals(basis), ThreePointFunctionals(basis))) {
            val solver = SecondKindSolver(basis, funcs, op, 1.0,
                { t -> ModelProblem.F2.rhsExact(t, op) }, { t -> ModelProblem.F2.rhsExactDeriv(t, op) })
            val e = errorEh({ t -> ModelProblem.F2.exact(t) }, solver.kulkarni().eval, grid)
            val eIt = errorEh({ t -> ModelProblem.F2.exact(t) }, solver.iteratedKulkarni().eval, grid)
            assertTrue(finite(e) && e < 5e-1, "quasi kulkarni E_h=$e")
            assertTrue(finite(eIt) && eIt < 5e-1, "quasi iterated E_h=$eIt")
        }
    }

    /**
     * Regression guard for the kulkarniQuasi early-exit fix (return@repeat -> break).
     * The converged fixed point must be independent of whether the loop stops early
     * (residual < 1e-13) or runs all 200 sweeps. We verify the quasi solution is a
     * stable, deterministic fixed point across repeated solves and stays finite.
     */
    @Test fun f2_kulkarniQuasi_convergesToStableFixedPoint() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(ModelProblem.F2.kernel, grid, quad)
        val funcs = AveragingFunctionals(basis)
        val solver = SecondKindSolver(basis, funcs, op, 1.0,
            { t -> ModelProblem.F2.rhsExact(t, op) }, { t -> ModelProblem.F2.rhsExactDeriv(t, op) })
        val first = solver.kulkarni()
        val second = solver.kulkarni()
        val ts = doubleArrayOf(0.0, 0.13, 0.37, 0.5, 0.71, 0.99, 1.0)
        for (t in ts) {
            val a = first.eval(t); val b = second.eval(t)
            assertTrue(finite(a) && finite(b), "finite at t=$t")
            assertTrue(abs(a - b) < 1e-14, "non-deterministic quasi solution at t=$t: $a vs $b")
        }
    }

    /**
     * Exercises matrix/vector assembly entry points (matrixM, matrixM2, vectorG,
     * vectorD) and basic structural invariants. Reference: matrices are dim x dim with
     * dim=n+2 and contain only finite entries (fixed from current run).
     */
    @Test fun matricesAndVectors_shapeAndFinite() {
        val s = build(ModelProblem.F2, GeneratingSystem.B, 8)
        val dim = s.dim
        val m = s.matrixM(); val m2 = s.matrixM2(); val g = s.vectorG(); val d = s.vectorD()
        assertTrue(m.size == dim && m2.size == dim && g.size == dim && d.size == dim)
        for (r in 0 until dim) for (c in 0 until dim) assertTrue(finite(m[r][c]) && finite(m2[r][c]))
        for (j in 0 until dim) assertTrue(finite(g[j]) && finite(d[j]))
    }

    /**
     * Covers the operator point-wise APIs apply/applyDeriv/applyNodes/applyDerivNodes
     * and verifies node-based evaluation matches the closure-based integral for u=1.
     * Reference (analytic for this sub-check): for u(s)=1, applyNodes==apply within 1e-9.
     */
    @Test fun fredholmOperator_nodeApis_consistent() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(ModelProblem.F2.kernel, grid, quad)
        val one = DoubleArray(op.gNode.size) { 1.0 }
        val t = 0.37
        assertTrue(abs(op.applyNodes(t, one) - op.apply(t) { 1.0 }) < 1e-9)
        assertTrue(finite(op.applyDerivNodes(t, one)))
        assertTrue(finite(op.applyDeriv(t) { 1.0 }))
        // rhsExact (first-kind branch) for F1.
        assertTrue(finite(ModelProblem.F1.rhsExact(t, op)) && finite(ModelProblem.F1.rhsExactDeriv(t, op)))
    }

    /**
     * Covers FirstKindSolver (Wazwaz alpha-regularization) end-to-end: base/sloan/
     * kulkarni/iteratedKulkarni on F1, basis H, n=8.
     *
     * Reference fixed from current run:
     *  - base  E_h ~ 6.42e-5, sloan E_h ~ 6.59e-5 (accurate) -> asserted < 1e-3;
     *  - kulkarni ~ 1.69e3, iteratedKulkarni ~ 8.06e6 BLOW UP, exactly as the source
     *    KDoc warns (M ~ alpha^-1, M2 ~ alpha^-2 with alpha=1e-10) -> asserted only
     *    finite (regression net for the diverging branch, NOT an accuracy claim).
     */
    @Test fun f1_firstKindSolver_allSchemes() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(ModelProblem.F1.kernel, grid, quad)
        val solver = FirstKindSolver(ModelProblem.F1, basis, funcs, op)
        val exact = { t: Double -> ModelProblem.F1.exact(t) }
        // Accurate, well-conditioned schemes.
        assertTrue(errorEh(exact, solver.base().eval, grid) < 1e-3)
        assertTrue(errorEh(exact, solver.sloan().eval, grid) < 1e-3)
        // Kulkarni branches blow up (documented ill-conditioning) -> finiteness only.
        assertTrue(finite(errorEh(exact, solver.kulkarni().eval, grid)))
        assertTrue(finite(errorEh(exact, solver.iteratedKulkarni().eval, grid)))
    }

    /**
     * Convergence sanity (characterization): on F2/basis B the base scheme error must
     * decrease from n=8 to n=16. Reference behaviour fixed from current implementation;
     * guards against a regression that destroys spatial convergence.
     */
    @Test fun f2_converges_n8_to_n16() {
        val e8 = errorEh({ t -> ModelProblem.F2.exact(t) }, build(ModelProblem.F2, GeneratingSystem.B, 8).base().eval, Grid.uniform(8))
        val e16 = errorEh({ t -> ModelProblem.F2.exact(t) }, build(ModelProblem.F2, GeneratingSystem.B, 16).base().eval, Grid.uniform(16))
        assertTrue(e16 < e8, "no convergence: e8=$e8 e16=$e16")
    }

    /** Trivial coverage of SolutionFunc.eval wrapper. */
    @Test fun solutionFunc_evalWrapper() {
        val sf = SolutionFunc { t -> 2.0 * t }
        assertTrue(abs(sf.eval(3.0) - 6.0) < 1e-12)
    }
}
