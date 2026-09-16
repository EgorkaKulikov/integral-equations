package characterization

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.DiscreteDeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraSecondKindSolver
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.TestInstance
import java.io.File
import java.util.Locale
import kotlin.test.Test

/**
 * A utility tool: prints the BASELINE SNAPSHOT of the E_h quantities over all the combinations
 * "problem x generating system x functional family x scheme" for the linear
 * solvers (Fredholm, Volterra) and the nonlinear one (Uryson).
 *
 * This is NOT a checking test: it asserts nothing and always finishes successfully.
 * Its only purpose is to obtain a machine-readable list of values that are
 * then transferred verbatim into the characterization tests (`CharacterizationTest`)
 * as a safety net before a refactoring.
 *
 * Run: `./gradlew captureBaseline` (see the output in the report or with the `-i` flag).
 * It CANNOT be run through `test --tests ...`: this class is deliberately excluded from
 * all the checking tasks by a filter, and Gradle will answer `No tests found for given includes`.
 *
 * The result is the file `build/baseline/baseline-eh.tsv`, which after inspection is copied
 * into `src/test/resources/characterization/baseline-eh.tsv`. The name is DETERMINISTIC, the file
 * is OVERWRITTEN in one operation, the rows are SORTED by key — the same design
 * as in [ExtraBaselineSnapshotTool], and for the same reasons:
 *  - the former name `snapshot-<thread name>.tsv` depended on the JUnit scheduler, because of which
 *    the snapshots had to be collected by the pattern `snapshot-*.tsv`;
 *  - the former writing went in `appendText` mode, so a repeated run without a manual
 *    `rm -rf build/baseline` doubled the contents of the file;
 *  - the former row order was the computation order, so a diff of two snapshots showed
 *    a permutation of rows mixed with a change of the numbers.
 *
 * The row format: `key<TAB>value`, where the value is printed with 17 significant digits.
 * 17 is the minimal precision at which the decimal record of a `double` is restored
 * BITWISE. The former 12 digits do NOT give a round-trip: `"%.12g".format(0.1 + 0.2)` = `0.300000000000`,
 * which is not equal to `0.1 + 0.2`. That is, the baseline itself introduced a relative storage error
 * of ~1e-12 — coarser than the real discrepancy of the backends (at most 1.0e-14 on the non-F1 keys),
 * and coarser than the gate tolerance 1e-9 on the keys with cancellation. Storing exactly what was computed is
 * the only way to separate "the arithmetic changed" from "the record of the number changed".
 *
 * The locale [Locale.ROOT] is set explicitly: `"%.17g".format(x)` takes the default locale and on
 * a machine with a Russian locale writes a comma instead of a dot, after which the snapshot is unreadable.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class BaselineSnapshotTool {

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = familyFor(name, basis)

    /**
     * The snapshot buffer: the shot "key - value" pairs of all the test methods of the class.
     *
     * Accumulating in memory rather than appending to the file is exactly what makes the file
     * overwritable and sorted. The lifecycle [TestInstance.Lifecycle.PER_CLASS]
     * is mandatory: with the standard `PER_METHOD` JUnit creates one instance per test method,
     * and the buffer of each of the five methods would be lost before writing.
     */
    private val rows = mutableListOf<Pair<String, String>>()

    /**
     * Puts a "key-value" pair into the buffer [rows].
     *
     * The output goes ONLY into the file (in [writeSnapshot]) and not to stdout: printing more than a thousand
     * lines breaks the formation of the Gradle XML report. The `build/` directory is not under version
     * control, which is what is needed for a temporary artifact.
     *
     * The `synchronized` is an insurance in case parallel test execution is enabled in
     * JUnit: today the methods of the class go sequentially, but the price of the insurance is zero,
     * while its absence would show up as silently lost rows of the snapshot.
     */
    private fun emit(key: String, value: Double) {
        val formatted = String.format(Locale.ROOT, "%.17g", value)
        synchronized(rows) { rows += key to formatted }
    }

    /**
     * Writes the whole snapshot in ONE operation after the last test method.
     *
     * One write instead of a thousand appends: it is both faster and excludes a partially written
     * file on a failure in the middle of the shooting. Sorting by key makes a diff of two snapshots
     * meaningful.
     */
    @AfterAll
    fun writeSnapshot() {
        // The output directory is set by a property: the task `classifyBaseline` shoots the matrix
        // TWICE (backend=java and backend=native) and must put the snapshots into DIFFERENT directories.
        val dir = File(System.getProperty("baseline.output.dir")?.takeIf { it.isNotBlank() } ?: "build/baseline").apply { mkdirs() }
        val target = File(dir, "baseline-eh.tsv")
        val sorted = rows.sortedBy { it.first }
        target.writeText(sorted.joinToString(separator = "") { (key, value) -> "$key\t$value\n" })
        println("Snapshot of the E_h matrix: ${sorted.size} rows -> ${target.absolutePath}")
    }

    /** A snapshot of the linear Fredholm solver. */
    @Test
    fun snapshotFredholm() {
        val problems = listOf(
            problems.fredholm.FredholmProblem.F2span,
            problems.fredholm.FredholmProblem.F2,
            problems.fredholm.FredholmProblem.F2exp,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val families = listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")
        for (problem in problems) {
            for (system in systems) {
                for (familyName in families) {
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
                        emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                        emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                        emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                        emit("$prefix.iterKulkarni", errorEh(exact, solver.iteratedKulkarni().eval, grid))
                        if (!funcs.usesDerivative) {
                            emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                            emit("$prefix.iterNystrom", errorEh(exact, solver.iteratedNystrom().eval, grid))
                        }
                    }
                }
            }
        }
    }

    /** A snapshot of the linear Volterra solver. */
    @Test
    fun snapshotVolterra() {
        val problems = listOf(
            problems.volterra.VolterraProblem.V2span,
            problems.volterra.VolterraProblem.V2,
            problems.volterra.VolterraProblem.V2exp,
            problems.volterra.VolterraProblem.V2win,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val families = listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")
        for (problem in problems) {
            for (system in systems) {
                for (familyName in families) {
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
                        emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                        emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                        emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                        emit("$prefix.iterKulkarni", errorEh(exact, solver.iteratedKulkarni().eval, grid))
                        if (!funcs.usesDerivative) {
                            emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                            emit("$prefix.iterNystrom", errorEh(exact, solver.iteratedNystrom().eval, grid))
                        }
                    }
                }
            }
        }
    }

    /** A snapshot of the nonlinear Uryson solver (second kind). */
    @Test
    fun snapshotUryson() {
        val urysonProblems = listOf(
            problems.uryson.UrysonProblem.A,
            problems.uryson.UrysonProblem.B,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        for (problem in urysonProblems) {
            for (system in systems) {
                for (n in listOf(8, 16)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system, grid)
                    val funcs = splines.functionals.ProjFunctionals(basis)
                    val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                    val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = problems.uryson.secondKindSolver(problem, basis, funcs, space, op)
                    val exact = { t: Double -> problem.exact(t) }
                    val prefix = "U.${problem.name}.${system.name}.n$n"
                    emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                    emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                    emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                    emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                }
            }
        }
    }

    /** A snapshot of the first-kind solvers (Fredholm — Wazwaz, Volterra — differentiation). */
    @Test
    fun snapshotFirstKind() {
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = splines.functionals.ProjFunctionals(basis)

            val fp = problems.fredholm.FredholmProblem.F1
            val fop = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val fSolver = problems.fredholm.firstKindSolver(fp, basis, funcs, fop)
            emit("F1.B.theta.n$n.base", errorEh({ t -> fp.exact(t) }, fSolver.base().eval, grid))
            emit("F1.B.theta.n$n.sloan", errorEh({ t -> fp.exact(t) }, fSolver.sloan().eval, grid))

            val vp = problems.volterra.VolterraProblem.V1
            val vop = solvers.volterra.VolterraOperator(vp.kernel, grid, GaussLegendre(8))
            val vSolver = problems.volterra.firstKindSolver(vp, basis, funcs, vop)
            emit("V1.B.theta.n$n.base", errorEh({ t -> vp.exact(t) }, vSolver.base().eval, grid))
            emit("V1.B.theta.n$n.sloan", errorEh({ t -> vp.exact(t) }, vSolver.sloan().eval, grid))
        }
    }

    /**
     * The EXTENDED F1 coverage (stage 8.6) — the same set of combinations as
     * in [F1_COVERAGE], see the KDoc there. A separate test method rather than an addition to
     * [snapshotFirstKind]: that one contains the EXISTING keys of the baseline, and any edit
     * in it risks shifting them; here only NEW keys are produced.
     */
    @Test
    fun snapshotFirstKindExtendedFredholm() {
        val fp = problems.fredholm.FredholmProblem.F1
        for ((system, familyName, n) in F1_COVERAGE) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val funcs = family(familyName, basis)
            val op = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val solver = problems.fredholm.firstKindSolver(fp, basis, funcs, op)
            val exact = { t: Double -> fp.exact(t) }
            val prefix = "F1.${system.name}.$familyName.n$n"
            emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
            emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
        }
    }

    companion object {
        /** The functional family by the name in the baseline key; shared with [F1ConditioningTest]. */
        internal fun familyFor(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
            "theta" -> splines.functionals.ProjFunctionals(basis)
            "xi0" -> DeBoorFixFunctionals(basis, 0)
            "xi1" -> DeBoorFixFunctionals(basis, 1)
            "xi2" -> DeBoorFixFunctionals(basis, 2)
            "xitilde1" -> DiscreteDeBoorFixFunctionals(basis, 1)
            "xitilde2" -> DiscreteDeBoorFixFunctionals(basis, 2)
            "mu" -> AveragingFunctionals(basis)
            else -> ThreePointFunctionals(basis)
        }

        /**
         * THE EXTENDED F1 COVERAGE in the characterization baseline (stage 8.6).
         *
         * The composition: triples (generating system, family, grid) — all three systems
         * B/H/T × the families theta/xi1/xi2 × the grids 8/16/32, each triple giving two
         * keys (`base` and `sloan`) — 54 keys.
         *
         * Why EXACTLY this set. It is a SUPERSET of the 42 F1 keys covered
         * by the cross-check with the publication (`verification/published-values.tsv`: 7 combinations
         * B.xi1, B.xi2, H.theta, H.xi1, H.xi2, T.xi1, T.xi2 × 3 grids × 2 schemes). That is,
         * every quantity that is cross-checked with the publication at a tolerance of 2 % or 12 %
         * is HERE protected by the tolerance 1e-9. It is exactly this that makes the wide tolerance
         * of the cross-check safe: it forgives the arithmetic irreproducibility between the LU paths,
         * while a change of the computation itself is caught by this net.
         *
         * The family `xi0` is NOT in the set: the tables of the article do not have it for F1,
         * while its functionals read `f''`, that is, they require `K_tt` quadratures and cost
         * noticeably more at zero gain in comparability with the publication.
         *
         * THE OVERLAP WITH THE OLD KEYS. The combinations `B.theta` at n = 8 and 16
         * are already shot by [snapshotFirstKind] and are therefore EXCLUDED from this set:
         * otherwise one and the same key would be written into the snapshot twice and a duplicate
         * row would appear in the baseline. `B.theta.n32` is a NEW key and is in the set.
         *
         * The list is shared by the snapshot tool and by
         * `EhCharacterizationTest.firstKindExtendedFredholmMatchesBaseline`: a desynchronization
         * of the composition between them is impossible by construction.
         */
        val F1_COVERAGE: List<Triple<GeneratingSystem, String, Int>> = buildList {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (familyName in listOf("theta", "xi1", "xi2")) {
                    for (n in listOf(8, 16, 32)) {
                        // We exclude what is already in the baseline under the same keys.
                        if (system == GeneratingSystem.B && familyName == "theta" && n != 32) continue
                        add(Triple(system, familyName, n))
                    }
                }
            }
        }
    }
}
