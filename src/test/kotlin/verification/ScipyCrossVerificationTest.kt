package verification

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * SMOKE TESTS OF THE EXTERNAL CROSS-CHECK AGAINST SciPy/NumPy.
 *
 * Why they are needed. All the other checks of the project are written on its own code and
 * therefore confirm only internal consistency: an error introduced before the baseline was
 * created would have been recorded together with it. This suite cross-checks the results
 * against THIRD-PARTY libraries (SciPy, NumPy) developed independently, and this is
 * the only evidence in the project not closed on the implementation under test.
 *
 * Why tests and not a one-off script. A one-off cross-check protects against an error only at
 * the moment of the run: any later edit of the numerical core could diverge from
 * SciPy unnoticed. What makes the cross-check a permanent guarantee is the separate `scipyVerify`
 * suite, which CI runs on every branch.
 *
 * Design. The test is self-contained: it dumps the artifacts itself via
 * [VerificationArtifacts] (without relying on the `dumpVerificationArtifacts` task having been
 * run before it), runs `tools/verify_with_scipy.py` and parses
 * its MACHINE-READABLE report. Parsing the report, and not only the exit code, is essential:
 * on a discrepancy the message names the concrete layer, the size of the deviation and the tolerance.
 *
 * Cross-check layers (bottom-up, so that the place of a discrepancy is visible at once):
 *
 *     L1  Gauss--Legendre quadrature          <- numpy leggauss
 *     L2  linear algebra, linear system solve <- scipy.linalg.solve
 *     L3  minimal spline basis                <- scipy.interpolate.BSpline
 *     L4/L5 operator images, right-hand sides <- scipy.integrate.quad (QUADPACK)
 *     L6a suitability of the baseline      <- independent Nystrom against the exact solution
 *     L6b solutions OF THE PROJECT (E_h)   <- limits and convergence order against the baseline
 *
 * Running. The regular way is `./gradlew scipyVerify`: this task prepares the Python
 * environment (`setupScipyVerification`), dumps the artifacts (`dumpVerificationArtifacts`)
 * and passes the interpreter path through the property `scipy.python`.
 *
 * Behaviour without a Python environment depends on the STRICT MODE (`scipy.required`):
 *
 *  - `scipy.required=false` or the property is unset (this is how the ordinary `test` works) —
 *    SKIP (`Assumptions`). A missing venv is a state of the machine, not a
 *    discrepancy with SciPy, and treating it as a failure would be a false signal;
 *  - `scipy.required=true` (this is how `scipyVerify` works) — FAILURE. Here the cross-check is
 *    requested explicitly and prepared by dependent tasks, so a skip would mean
 *    an unnoticed breakage of the preparation and a GREEN build without a single cross-check —
 *    exactly what this check must prevent.
 *
 * The split is made because one and the same skip has a different meaning: in an
 * everyday run it is expected, in a targeted one it is a defect.
 */
@Tag("scipy")
class ScipyCrossVerificationTest {

    private companion object {
        /** The cross-check script; the path is relative to the project root. */
        const val SCRIPT_PATH = "tools/verify_with_scipy.py"

        /**
         * Time limit for the cross-check run. The script takes a few seconds;
         * the margin is needed for a cold start of the interpreter and importing SciPy.
         * The limit is mandatory: without it an environment failure would hang the build.
         */
        const val TIMEOUT_SECONDS = 300L

        /** The layers whose presence in the report is mandatory. */
        val REQUIRED_LAYERS = listOf("L1", "L2", "L3", "L4/L5", "L6a", "L6b")

        /** The Fredholm problems dumped by the dumper and cross-checked in the layer L4/L5. */
        val FREDHOLM_PROBLEMS = listOf("F2", "F2exp")

        /** The Volterra problems dumped by the dumper and cross-checked in the layer L4/L5. */
        val VOLTERRA_PROBLEMS = listOf("V2", "V2exp", "V2win")

        /**
         * The right-hand side quantities whose cross-check is mandatory for EVERY problem.
         *
         * `rhsDeriv`/`rhsDeriv2` are listed EXPLICITLY: it is exactly they that used to silently drop
         * out of the cross-check, and exactly they that check the Leibniz formulas for Volterra.
         */
        val RHS_QUANTITIES = listOf("rhs", "rhsDeriv", "rhsDeriv2")

        /**
         * Strict mode: the cross-check is requested explicitly, the environment must be ready.
         * Set by the `scipyVerify` task; absent in the other runs.
         */
        val STRICT: Boolean = System.getProperty("scipy.required")?.toBoolean() ?: false

        /**
         * The cross-check result in machine-readable form. Parsed without external JSON
         * parsing libraries: the project has none, and the report format is fixed and simple.
         */
        var cachedReport: ScipyReport? = null
    }

    /**
     * A requirement on the ENVIRONMENT (not on the numbers): in strict mode a violation is a failure,
     * otherwise a skip. The message is the same: the cause and the way to fix it are needed
     * in both cases, only the status of the test changes.
     *
     * NOTE: this method never applies to NUMERICAL discrepancies:
     * they always fail through `assertTrue`, in any mode.
     */
    private fun requireEnvironment(condition: Boolean, message: String) {
        if (condition) return
        if (STRICT) {
            fail(
                "$message\n\nThe `scipyVerify` task requires a working environment (scipy.required=true): " +
                    "the external cross-check is the only evidence not closed on the project code, and skipping " +
                    "it silently means getting a green build without a single check performed. " +
                    "How to fix: `./gradlew setupScipyVerification` — the task creates .venv-verify " +
                    "and installs the versions from tools/requirements-verify.txt, after which repeat the cross-check.",
            )
        }
        assumeTrue(false, message)
    }

    /**
     * One check from the report of the script.
     *
     * [compared] is the number of points ACTUALLY compared, [skipped] the number dropped from the
     * comparison. Without the former the report cannot tell "no discrepancies" from
     * "there was nothing to compare": the deviation is zero in both cases.
     */
    private class Check(
        val layer: String,
        val name: String,
        val deviation: Double,
        val tolerance: Double,
        val ok: Boolean,
        val compared: Int,
        val skipped: Int,
    )

    /** The parsed report of the cross-check script. */
    private class ScipyReport(
        val exitCode: Int,
        val numpyVersion: String,
        val scipyVersion: String,
        val checks: List<Check>,
        val failures: List<String>,
        val notes: List<String>,
        val consoleOutput: String,
    )

    /**
     * Prepares the artifacts, runs the cross-check and parses the report.
     *
     * The result is cached: the script performs all layers in one run, and repeating the
     * run for every test method would only multiply the build time.
     */
    private fun report(): ScipyReport {
        cachedReport?.let { return it }

        val artifactDir = VerificationArtifacts.DEFAULT_DIR
        VerificationArtifacts.dumpAll(artifactDir)

        val configuredPython: String? = System.getProperty("scipy.python")
        requireEnvironment(
            configuredPython != null,
            "The SciPy cross-check environment is unavailable: the property scipy.python is not set. The regular " +
                "way to run is ./gradlew scipyVerify (the task prepares the environment and passes the path itself). " +
                "For a manual run specify -Dscipy.python=<path to the interpreter>.",
        )
        val python = configuredPython!!
        requireEnvironment(
            File(python).exists(),
            "The SciPy cross-check environment is unavailable: the Python interpreter is not found ($python). " +
                "Run ./gradlew setupScipyVerification — the task creates the environment " +
                "and installs SciPy/NumPy of the versions from tools/requirements-verify.txt.",
        )
        val script = File(SCRIPT_PATH)
        if (!script.exists()) fail("The cross-check script ${script.absolutePath} is not found")

        val jsonFile = File(artifactDir, "scipy-report.json")
        jsonFile.delete()
        val process = ProcessBuilder(
            python,
            script.absolutePath,
            "--artifacts", artifactDir.absolutePath,
            "--json", jsonFile.absolutePath,
        ).redirectErrorStream(true).start()
        val consoleOutput = process.inputStream.bufferedReader().readText()
        val finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        if (!finished) {
            process.destroyForcibly()
            fail("The SciPy cross-check did not finish in $TIMEOUT_SECONDS s. Output:\n$consoleOutput")
        }
        val exitCode = process.exitValue()
        // Code 2 means SciPy/NumPy are unavailable in the found interpreter — the case
        // "the venv exists, the packages do not". This is a state of the environment, not a discrepancy of
        // numbers, so the reaction depends on the strict mode (see [requireEnvironment]).
        requireEnvironment(
            exitCode != 2,
            "The SciPy cross-check environment is not operational: SciPy/NumPy are unavailable in $python " +
                "(the interpreter exists, the packages do not). Run ./gradlew setupScipyVerification. " +
                "Script output:\n$consoleOutput",
        )
        if (!jsonFile.exists()) {
            fail(
                "The cross-check script did not create the machine-readable report ${jsonFile.absolutePath} " +
                    "(exit code $exitCode). Output:\n$consoleOutput",
            )
        }
        val parsed = parseReport(jsonFile.readText(), exitCode, consoleOutput)
        cachedReport = parsed
        return parsed
    }

    /**
     * Parses the report of the script.
     *
     * The format is produced by `json.dump` from [SCRIPT_PATH] with indentation, so line-wise
     * parsing is reliable: every field is on its own line. A separate dependency
     * just to parse this file is not justified.
     */
    private fun parseReport(text: String, exitCode: Int, consoleOutput: String): ScipyReport {
        fun scalar(field: String): String =
            Regex("\"$field\"\\s*:\\s*\"([^\"]*)\"").find(text)?.groupValues?.get(1) ?: ""

        fun stringList(field: String): List<String> {
            val block = Regex("\"$field\"\\s*:\\s*\\[(.*?)]", RegexOption.DOT_MATCHES_ALL)
                .find(text)?.groupValues?.get(1) ?: return emptyList()
            return Regex("\"((?:[^\"\\\\]|\\\\.)*)\"").findAll(block)
                .map { it.groupValues[1] }
                .toList()
        }

        val checks = mutableListOf<Check>()
        // One element of the checks array: the fields follow the order set by the script.
        val entry = Regex(
            "\\{\\s*\"layer\"\\s*:\\s*\"([^\"]*)\"\\s*,\\s*\"name\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"\\s*," +
                "\\s*\"deviation\"\\s*:\\s*([-0-9.eE+]+)\\s*,\\s*\"tolerance\"\\s*:\\s*([-0-9.eE+]+)\\s*," +
                "\\s*\"kind\"\\s*:\\s*\"[^\"]*\"\\s*,\\s*\"ok\"\\s*:\\s*(true|false)\\s*," +
                "\\s*\"compared\"\\s*:\\s*(\\d+)\\s*,\\s*\"skipped\"\\s*:\\s*(\\d+)",
            RegexOption.DOT_MATCHES_ALL,
        )
        for (match in entry.findAll(text)) {
            checks += Check(
                layer = match.groupValues[1],
                name = match.groupValues[2],
                deviation = match.groupValues[3].toDouble(),
                tolerance = match.groupValues[4].toDouble(),
                ok = match.groupValues[5] == "true",
                compared = match.groupValues[6].toInt(),
                skipped = match.groupValues[7].toInt(),
            )
        }
        return ScipyReport(
            exitCode = exitCode,
            numpyVersion = scalar("numpy"),
            scipyVersion = scalar("scipy"),
            checks = checks,
            failures = stringList("failures"),
            notes = stringList("notes"),
            consoleOutput = consoleOutput,
        )
    }

    /** The checks of one layer; an empty list means the layer was not performed. */
    private fun checksOf(layer: String): List<Check> = report().checks.filter { it.layer == layer }

    /**
     * A check must be BOTH CONVERGED AND NON-EMPTY.
     *
     * The second requirement is not redundant: the script excludes points from the comparison in a number
     * of cases (index outside the basis, NaN of the baseline, a point at a node, a degenerate integration
     * interval), and if ALL points drop out, the deviation stays zero —
     * outwardly indistinguishable from perfect agreement.
     */
    private fun assertMeaningful(check: Check) {
        assertTrue(
            check.compared > 0,
            "${check.layer} ${check.name}: 0 points compared (${check.skipped} skipped) — the check " +
                "was not performed. A zero deviation here means absence of data, not agreement.",
        )
        assertTrue(
            check.ok,
            "${check.layer} ${check.name}: the deviation ${check.deviation} exceeds the tolerance " +
                "${check.tolerance} (points compared ${check.compared}, skipped ${check.skipped})",
        )
    }

    /**
     * Summary check: there are no discrepancies with SciPy on any layer.
     *
     * The error message contains the list of discrepancies with the values and tolerances,
     * and also the full output of the script — otherwise diagnosis from the exit code alone
     * would be impossible.
     */
    @Test
    fun allLayersAgreeWithScipy() {
        val result = report()
        val failed = result.checks.filterNot { it.ok }
        assertTrue(
            result.exitCode == 0 && failed.isEmpty(),
            buildString {
                appendLine(
                    "The cross-check against SciPy ${result.scipyVersion} / NumPy ${result.numpyVersion} " +
                        "found discrepancies (${failed.size} of them, exit code ${result.exitCode}).",
                )
                appendLine("A discrepancy must NOT be fixed by loosening the tolerance: find the cause first.")
                for (check in failed) {
                    appendLine(
                        "  ${check.layer} / ${check.name}: deviation ${check.deviation} " +
                            "> tolerance ${check.tolerance} " +
                            "(points compared ${check.compared}, skipped ${check.skipped})",
                    )
                }
                for (failure in result.failures) appendLine("  $failure")
                appendLine("--- script output ---")
                append(result.consoleOutput.take(4000))
            },
        )
    }

    /**
     * The report must contain ALL layers and must not be empty.
     *
     * Without this check the cross-check could "pass" degenerately: if the script does not
     * find the artifacts, it merely records a note and returns a zero code, and a green
     * test would create a false impression of being verified.
     */
    @Test
    fun everyLayerIsActuallyExecuted() {
        val result = report()
        assertTrue(result.checks.isNotEmpty(), "The cross-check report is empty: not a single check was performed")
        val missing = REQUIRED_LAYERS.filter { layer -> result.checks.none { it.layer == layer } }
        assertTrue(
            missing.isEmpty(),
            "The report is missing the layers: $missing. Probably the artifacts were not dumped. " +
                "Notes of the script: ${result.notes}",
        )
        assertTrue(
            result.notes.isEmpty(),
            "The script reported unverified layers: ${result.notes}. " +
                "The cross-check must be performed in full, otherwise it is not a guarantee.",
        )
    }

    /**
     * NO check of the report may be degenerate (`compared == 0`).
     *
     * Why a separate test. The presence of a layer in the report does not yet mean the layer
     * compared anything: points drop out of the comparison silently (index outside the basis, NaN
     * of the baseline, a point at a node, a degenerate integration interval), while the worst
     * deviation is initialized with zero. Without this requirement any error in the
     * skip condition would turn the cross-check into a silently green dummy.
     */
    @Test
    fun everyCheckComparedAtLeastOnePoint() {
        val result = report()
        val empty = result.checks.filter { it.compared <= 0 }
        assertTrue(
            empty.isEmpty(),
            buildString {
                appendLine(
                    "The report contains checks that compared NOT A SINGLE point (${empty.size} of them). " +
                        "Such a check proves nothing: the deviation equals zero because " +
                        "there was nothing to compare.",
                )
                for (check in empty) {
                    appendLine("  ${check.layer} / ${check.name}: ${check.skipped} points skipped")
                }
            },
        )
        // The total number of compared points for every mandatory layer is positive.
        for (layer in REQUIRED_LAYERS) {
            val total = result.checks.filter { it.layer == layer }.sumOf { it.compared }
            assertTrue(total > 0, "Layer $layer compared no points at all (total compared = 0)")
        }
    }

    /**
     * L1: the nodes and weights of the Gauss-Legendre quadrature coincide with `numpy leggauss`.
     *
     * The check is independent in substance: the project computes the nodes by Newton's method from
     * the zeros of the Legendre polynomial, NumPy uses a different algorithm.
     */
    @Test
    fun quadratureNodesMatchNumpy() {
        val checks = checksOf("L1")
        assertTrue(checks.size >= 2, "Checks of the nodes and the weights were expected, got ${checks.size}")
        for (check in checks) assertMeaningful(check)
    }

    /** L2: the solution of the assembled system `(I - M) c = g` coincides with `scipy.linalg.solve`. */
    @Test
    fun linearAlgebraMatchesScipy() {
        val checks = checksOf("L2")
        assertTrue(checks.isNotEmpty(), "Layer L2 was not performed")
        for (check in checks) assertMeaningful(check)
    }

    /**
     * L3: the minimal spline basis of the system `B` and its two derivatives coincide
     * with `scipy.interpolate.BSpline` on four kinds of grids.
     *
     * The cross-check is possible only for the system `B`: breakpoints of multiplicity 3 at the ends define
     * a clamped knot vector of degree 2. For the systems `H` and `T` there is no analogue in SciPy
     * (see docs/REFERENCES.md, section 6).
     */
    @Test
    fun splineBasisMatchesScipyBSpline() {
        val checks = checksOf("L3")
        // Four grids by three quantities (the value and two derivatives).
        assertTrue(checks.size >= 12, "At least 12 L3 checks were expected, got ${checks.size}")
        for (check in checks) assertMeaningful(check)
    }

    /**
     * L4/L5: the operator images, the right-hand sides AND THEIR DERIVATIVES coincide with
     * `scipy.integrate.quad`.
     *
     * This is the only numerical check in the project of the Leibniz formulas for `(Vu)'`
     * and `(Vu)''` (there is no separate publication for `(Vu)''`, see docs/REFERENCES.md, sec. 4):
     * it is performed by the checks `V/<problem>/rhsDeriv` and `V/<problem>/rhsDeriv2`, where
     * the baseline is assembled from INDEPENDENTLY derived `K_t`, `K_tt`, `K(t,t)` and the FULL
     * derivative of the diagonal `d/dt K(t,t) = K_t(t,t) + K_s(t,t)` (the derivation is written out in the
     * docstring of `volterra_image_deriv` in `tools/verify_with_scipy.py`).
     *
     * The checks are required BY NAME, not merely as "the list is non-empty". The reason is
     * factual: before that the layer silently dropped `rhsDeriv`/`rhsDeriv2` into a `continue` —
     * the dumped data were there, the cross-check was not, and the test was green all the same.
     * The requirement on the names makes a repetition of such a disappearance impossible.
     */
    @Test
    fun operatorImagesMatchQuadpack() {
        val checks = checksOf("L4/L5")
        assertTrue(checks.isNotEmpty(), "Layer L4/L5 was not performed")
        val present = checks.map { it.name }.toSet()
        val required = buildList {
            for (problem in FREDHOLM_PROBLEMS) {
                add("F/$problem/Ku")
                for (quantity in RHS_QUANTITIES) add("F/$problem/$quantity")
            }
            for (problem in VOLTERRA_PROBLEMS) {
                add("V/$problem/Vu")
                for (quantity in RHS_QUANTITIES) add("V/$problem/$quantity")
            }
        }
        val missing = required.filterNot { it in present }
        assertTrue(
            missing.isEmpty(),
            "The layer L4/L5 is missing the mandatory checks: $missing. The dumped data " +
                "are always there, so a missing check means the script again drops lines " +
                "into a `continue`. Actually present: ${present.sorted()}",
        )
        for (check in checks) assertMeaningful(check)
    }

    /**
     * L6a: the baseline Nyström method on the Gauss-Legendre quadrature reproduces
     * the exact solutions of the model problems.
     *
     * The method is implemented with NumPy/SciPy without the splines and functionals of the project
     * (a textbook scheme, Atkinson 1997, ch. 4) and serves as an external baseline. The check
     * confirms the suitability of the baseline itself: if it does not reproduce the exact
     * solution, it cannot be cross-checked against.
     */
    @Test
    fun referenceNystromReproducesExactSolutions() {
        val checks = checksOf("L6a")
        assertTrue(checks.isNotEmpty(), "Layer L6a was not performed")
        for (check in checks) assertMeaningful(check)
    }

    /**
     * L6b: the resulting errors `E_h` OF THE PROJECT SCHEMES THEMSELVES are cross-checked against the baseline.
     *
     * Why separately from L6a. The layer L6a checks ONLY the oracle (the baseline against
     * the exact solution) and reads no artifact of the project — a green L6a
     * says nothing about the project. It is L6b that reads `solution-errors.tsv` and cross-checks
     * the dumped `E_h` against the limits and the observed convergence order. Before this
     * layer the dumped file was read NOWHERE, and any corruption of its values went
     * unnoticed.
     *
     * AT LEAST 12 limit checks are required: two problems by three generating
     * systems by two schemes. The number is stated explicitly so that the disappearance of part of the
     * combinations from the dump or from the table of limits does not pass silently.
     */
    @Test
    fun projectSolutionErrorsMatchReference() {
        val checks = checksOf("L6b")
        val limitChecks = checks.filter { it.name.contains("E_h") }
        assertTrue(
            limitChecks.size >= 12,
            "At least 12 checks of the E_h limits were expected (2 problems x 3 systems x 2 schemes), " +
                "got ${limitChecks.size}. The actual L6b checks: ${checks.map { it.name }}",
        )
        // The convergence order checks must be present too: without them a scheme
        // stuck at the accuracy of a coarse grid would still pass the limits.
        assertTrue(
            checks.any { it.name.contains("min p") },
            "The layer L6b has not a single convergence order check: limits without a check " +
                "of the decrease are not enough",
        )
        for (check in checks) assertMeaningful(check)
    }
}
