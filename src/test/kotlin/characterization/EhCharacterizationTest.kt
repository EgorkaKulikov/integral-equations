package characterization

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraSecondKindSolver
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * A CHARACTERIZATION TEST (a safety net against regressions).
 *
 * It does not check mathematical correctness — it records the CURRENT behaviour of all the
 * combinations "problem x generating system x functional family x scheme", so that
 * a subsequent refactoring (moving files, renamings, extracting common code) does not
 * change the numerical results unnoticed.
 *
 * The baseline is stored in `src/test/resources/characterization/baseline-eh.tsv` and was shot
 * by the tool [BaselineSnapshotTool] on the original state of the repository.
 *
 * THE COMPARISON MODE IS TAKEN FROM THE DATA — the third column of the baseline (`class`), and not from
 * constants of this class. The classes and their measured justification are in [BaselineClass];
 * the parsing and the comparison in [BaselineFormat]; the computation of the column is `./gradlew classifyBaseline`.
 * Briefly: `portable` is rel. 1e-9 at the floor 6e-13 (it catches any change of the algorithm,
 * forgives the last bits at a different summation order), `sensitive` is the bound
 * `2*cond*max(omega,eps)*||u||inf`, COMPUTED in the run ([F1SystemConditioning]).
 *
 * IMPORTANT: if a change of the algorithm is JUSTIFIED (a bug fix), the baseline should
 * be revised deliberately, recording the old and the new values in a report, and not by
 * "fitting" the tolerance.
 *
 * THE `machine` TAG IS GONE — and this is the result of a MEASUREMENT, not of a softening of the requirements.
 * Shooting both matrices on `-Dnumerics.backend=java` (netlib F2J) and on `native`
 * (netlib + Apple Accelerate) showed: of the 1366 keys exactly 52 broke the current gate,
 * ALL of them `F1.*`; for the other 1312 the discrepancy of the LU paths does not exceed 1.0e-14 at the floor
 * 6e-13 — a 60x margin. The F1 keys are separated into the class `sensitive` and are cross-checked against a bound
 * computed on the same backend, so the gate is green on BOTH LU paths and runs
 * in CI (ubuntu/OpenBLAS — a third independent path). The details are in `docs/TESTING.md`
 * and `docs/baseline-changes.md`.
 */
@Tag("slow")
class EhCharacterizationTest {

    private companion object {
        /** The resource path of the baseline; shared with the snapshot tool and the guard test. */
        const val RESOURCE_PATH = "/characterization/baseline-eh.tsv"
    }

    /** The pair "baseline key -> the recorded value and the comparison class". */
    private val baseline: Map<String, BaselineEntry> by lazy {
        val resource = javaClass.getResourceAsStream(RESOURCE_PATH)
            ?: fail("The baseline file $RESOURCE_PATH is not found")
        resource.bufferedReader().useLines { BaselineFormat.parse(it, RESOURCE_PATH) }
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> splines.functionals.ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** Cross-checks the computed value against the baseline by the rule of the CLASS of this key. */
    private fun check(key: String, actual: Double, mismatches: MutableList<String>) {
        val expected = baseline[key] ?: run {
            mismatches += "$key: absent from the baseline (computed $actual)"
            return
        }
        // The value is brought to the same representation in which it is stored (17 digits,
        // a round-trip for a double): otherwise the comparison would depend on the way of printing.
        val formatted = String.format(Locale.ROOT, "%.17g", actual)
        BaselineFormat.compare(key, expected, formatted)?.let { mismatches += it }
    }

    private fun reportIfAny(mismatches: List<String>) {
        assertTrue(
            mismatches.isEmpty(),
            "A change of the numerical behaviour was detected (${mismatches.size} cases):\n" +
                mismatches.joinToString("\n").take(4000),
        )
    }

    /** All the Fredholm schemes on all the bases and functional families. */
    @Test
    fun fredholmMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val problems = listOf(
            problems.fredholm.FredholmProblem.F2span,
            problems.fredholm.FredholmProblem.F2,
            problems.fredholm.FredholmProblem.F2exp,
        )
        for (problem in problems) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    for (n in listOf(8, 16)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                        val solver = FredholmSecondKindSolver(
                            basis, funcs, op, 1.0,
                            RhsWithDerivatives(
                                { t -> problem.rhsExact(t, op) },
                                { t -> problem.rhsExactDeriv(t, op) },
                                { t -> problem.rhsExactDeriv2(t, op) },
                            ),
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "F.${problem.name}.${system.name}.$familyName.n$n"
                        check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                        check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                        check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                        check(
                            "$prefix.iterKulkarni",
                            errorEh(exact, solver.iteratedKulkarni().eval, grid),
                            mismatches,
                        )
                        if (!funcs.usesDerivative) {
                            check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                            check(
                                "$prefix.iterNystrom",
                                errorEh(exact, solver.iteratedNystrom().eval, grid),
                                mismatches,
                            )
                        }
                    }
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** All the Volterra schemes on all the bases and functional families. */
    @Test
    fun volterraMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val problems = listOf(
            problems.volterra.VolterraProblem.V2span,
            problems.volterra.VolterraProblem.V2,
            problems.volterra.VolterraProblem.V2exp,
            problems.volterra.VolterraProblem.V2win,
        )
        for (problem in problems) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    for (n in listOf(8, 16)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
                        val solver = VolterraSecondKindSolver(
                            basis, funcs, op, 1.0,
                            RhsWithDerivatives(
                                { t -> problem.rhsExact(t, op) },
                                { t -> problem.rhsExactDeriv(t, op) },
                                { t -> problem.rhsExactDeriv2(t, op) },
                            ),
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "V.${problem.name}.${system.name}.$familyName.n$n"
                        check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                        check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                        check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                        check(
                            "$prefix.iterKulkarni",
                            errorEh(exact, solver.iteratedKulkarni().eval, grid),
                            mismatches,
                        )
                        if (!funcs.usesDerivative) {
                            check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                            check(
                                "$prefix.iterNystrom",
                                errorEh(exact, solver.iteratedNystrom().eval, grid),
                                mismatches,
                            )
                        }
                    }
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** The nonlinear Uryson schemes (second kind) on all the bases. */
    @Test
    fun urysonMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        for (problem in listOf(problems.uryson.UrysonProblem.A, problems.uryson.UrysonProblem.B)) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (n in listOf(8, 16)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system, grid)
                    val funcs = splines.functionals.ProjFunctionals(basis)
                    val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                    val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = problems.uryson.secondKindSolver(problem, basis, funcs, space, op)
                    val exact = { t: Double -> problem.exact(t) }
                    val prefix = "U.${problem.name}.${system.name}.n$n"
                    check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                    check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                    check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                    check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** The first-kind solvers (Fredholm — Wazwaz, Volterra — differentiation). */
    @Test
    fun firstKindMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = splines.functionals.ProjFunctionals(basis)

            val fp = problems.fredholm.FredholmProblem.F1
            val fop = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val fSolver = problems.fredholm.firstKindSolver(fp, basis, funcs, fop)
            check("F1.B.theta.n$n.base", errorEh({ t -> fp.exact(t) }, fSolver.base().eval, grid), mismatches)
            check("F1.B.theta.n$n.sloan", errorEh({ t -> fp.exact(t) }, fSolver.sloan().eval, grid), mismatches)

            val vp = problems.volterra.VolterraProblem.V1
            val vop = solvers.volterra.VolterraOperator(vp.kernel, grid, GaussLegendre(8))
            val vSolver = problems.volterra.firstKindSolver(vp, basis, funcs, vop)
            check("V1.B.theta.n$n.base", errorEh({ t -> vp.exact(t) }, vSolver.base().eval, grid), mismatches)
            check("V1.B.theta.n$n.sloan", errorEh({ t -> vp.exact(t) }, vSolver.sloan().eval, grid), mismatches)
        }
        reportIfAny(mismatches)
    }

    /**
     * THE EXTENDED F1 COVERAGE — a COMPENSATION for the wide tolerance of the cross-check (stage 8.6).
     *
     * Why it exists. `verification.PublishedValuesTest` cross-checks 42 F1 quantities against the
     * publication with a tolerance of 2 %, and six keys from `table-f1.tex` with 12 %
     * (the table was shot on a different LU path, see the KDoc there). MEASURED (stage 8.6):
     * at such tolerances a coarsening of the quadrature `GaussLegendre(8) → 6` and `→ 4`
     * passes the cross-check UNNOTICED on all 42 keys. The reason is not the tolerance:
     * for F1 the `E_h ≈ 8e-5` is determined by the regularization (`alpha = 1e-10`) and not by the
     * discretization — the convergence order is ≈ 0, and the quantity itself is little sensitive
     * to the quality of the quadrature.
     *
     * Here, on the contrary, the rule of the class `sensitive` — the bound `2*cond*max(omega,eps)*||u||inf`
     * of order 3.8e-5 at values `E_h ~ 8e-5` — catches the same mutation
     * immediately. The composition of the coverage is a SUPERSET of what is cross-checked against the
     * publication, so no quantity relaxed there is left without
     * a strict gate. The enumeration of the combinations is taken from [BaselineSnapshotTool.F1_COVERAGE]
     * — one and the same for the gate and for the snapshot tool, so the composition
     * of the baseline and the composition of the check cannot diverge.
     *
     * ON THE `sloan` SCHEME FOR F1 — an important property and NOT a defect. The `F1.*.sloan` keys
     * ARE BOUND TO THE TRAVERSAL ORDER OF THE NODES in `FredholmOperator.applyNodes`. The reason:
     * the Sloan solution is `fEff(t) + c_L·applyNodes(t, ·)`, where `c_L = -1/alpha = -1e10`,
     * and BOTH terms are of order 1.38e10, while their sum is of order 2.7
     * (measured, stage 8.6). That is, a cancellation by 5·10^9 times takes place, and about
     * 9.7 of the 16 significant digits are lost. MEASUREMENT: three MATHEMATICALLY EQUIVALENT
     * summation orders of the same formula (forward, backward, compensated
     * Kahan-Neumaier) give `E_h` differing by 4.3-39.9 % (median 28 %) —
     * more than the discrepancy with the publication and more than the discrepancy of the backends.
     *
     * The practical consequence for a future refactoring: ANY permutation
     * of the summation loop in `applyNodes` (a backward traversal, block or parallel
     * summation, compensated addition) will break this gate ON THE KEYS
     * `F1.*.sloan` WITHOUT CHANGING THE MATHEMATICS. Such a failure does NOT indicate
     * an error in the new code — but neither can it simply be silenced: a
     * deliberate re-shooting of these keys with an entry in `docs/baseline-changes.md` is required.
     */
    @Test
    fun firstKindExtendedFredholmMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val fp = problems.fredholm.FredholmProblem.F1
        for ((system, familyName, n) in BaselineSnapshotTool.F1_COVERAGE) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val funcs = family(familyName, basis)
            val op = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val solver = problems.fredholm.firstKindSolver(fp, basis, funcs, op)
            val exact = { t: Double -> fp.exact(t) }
            val prefix = "F1.${system.name}.$familyName.n$n"
            check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
            check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
        }
        reportIfAny(mismatches)
    }
}
