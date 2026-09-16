package characterization

import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

/**
 * THE BASELINE CLASSIFICATION TOOL: the `class` column is COMPUTED, not written by hand.
 *
 * Run: `./gradlew classifyBaseline`. The task shoots both matrices TWICE — on
 * `-Dnumerics.backend=java` (netlib F2J, pure Java) and on `native` (netlib + the system
 * LAPACK, on the developer machine Apple Accelerate) — after which this tool
 * compares the snapshots and writes `build/baseline/classified/baseline-{eh,extra}.tsv`
 * with the values of the NATIVE run and a third column with the class.
 *
 * THE CLASSIFICATION RULE (in the order of application):
 *  1. a `*.iters` key — [BaselineClass.EXACT]. The counter is integral, and the measurement
 *     permits the strictness: between the backends it never diverged (0 of 336).
 *     If it does diverge, the tool FAILS: this is a regression, not a new class;
 *  2. a `*.residual` key — [BaselineClass.RESIDUAL];
 *  3. the java and native values agree by the `portable` rule — [BaselineClass.PORTABLE];
 *  4. otherwise [BaselineClass.SENSITIVE], BUT only if the key is supported by the bound
 *     machinery ([F1SystemConditioning.supports], that is, the `base`/`sloan` schemes of the problem F1).
 *     A key of any other scheme that went beyond the `portable` rule is a DEFECT: the tool
 *     fails with a listing of such keys. For `kulkarni`/`nystrom`/Uryson no bound
 *     exists without exposing the matrix of the corresponding scheme, and a silent
 *     widening of the class would mean switching the gate off on these keys.
 *
 * GROUP PROMOTION. The class is assigned to the TRIPLE (system, family, n) as a whole: if
 * at least one `base`/`sloan` key of the group diverged, both get `sensitive`. Otherwise
 * the composition of `sensitive` would depend on which of the two keys is a bit closer to the
 * bound today — that is, on noise rather than on a property of the system.
 */
object BaselineClassifier {

    private const val EH_NAME = "baseline-eh.tsv"
    private const val EXTRA_NAME = "baseline-extra.tsv"

    private val EH_HEADER = """
        # CHARACTERIZATION BASELINE OF E_h (a snapshot of the implementation's own behaviour).
        #
        # Cross-checked by the test characterization.EhCharacterizationTest.
        # ROW FORMAT:  key <TAB> value <TAB> class.
        #   value — 17 significant digits (%.17g): this is the MINIMUM at which the decimal record
        #           of a double is restored bitwise; at 12 digits the record of the baseline itself
        #           introduced a storage error of ~1e-12 — coarser than the measured discrepancy
        #           of the BLAS implementations (<= 1.0e-14 on the non-F1 keys);
        #   class — the comparison mode, COMPUTED by the tool `./gradlew classifyBaseline`
        #           from two snapshots (-Dnumerics.backend=java against native). It is not set
        #           by hand: a handwritten list would repeat "what fails today".
        #     portable  — rel. 1e-9 at the floor 6e-13 = 10^3*eps*||u||inf. The discrepancy of the LU paths
        #                 on these keys is <= 1.0e-14, that is, a 60x margin;
        #     sensitive — |dlt| <= 2*cond_1*max(omega,eps)*||u||inf, where cond and omega are COMPUTED
        #                 in the same run and on the same backend (characterization.F1SystemConditioning).
        #                 A constant must not be stored here: it would bring back the binding to a machine;
        #     exact     — a strict match of the strings (integral iteration counters);
        #     residual  — rel. 1e-3 at the noise 6e-15.
        # Lines starting with # and empty ones are comments. ANY other line must
        # split into exactly three columns, otherwise the parsing FAILS (the former parser silently
        # treated such a line as a comment, and the key became "absent from the baseline").
        #
        # THE SHOOTING ENVIRONMENT (essential: the numbers are bound to it).
        #   linear algebra backend : netlib + system LAPACK (Apple Accelerate), -Dnumerics.backend=native
        #   libraries              : numerical-core 1.1.0, minimal-splines 1.1.0
        #   JDK                     : 21 (jvmToolchain(21))
        #   platform               : macOS aarch64 (Apple silicon)
        # PORTABILITY IS VERIFIED: the gate is green both on `native` and on `java` (a different LU path).
        # Exactly for that reason the `machine` tag was removed from the gate, and it runs in CI on ubuntu/OpenBLAS.
        #
        # THE F1.*.sloan KEYS ARE BOUND TO THE SUMMATION ORDER, and not only to the backend.
        # F1 is an equation of the first kind, solved by Wazwaz regularization with alpha = 1e-10, that is,
        # c_L = -1/alpha = -1e10. The Sloan solution is fEff(t) + c_L*applyNodes(t, .), where BOTH
        # terms are of order 1.38e10, while their sum is of order 2.7: a cancellation by 5e9 times,
        # about 9.7 of the 16 significant digits are lost. Three MATHEMATICALLY EQUIVALENT summation
        # orders give E_h differing by 4.3-39.9 % (median 28 %). It is exactly these keys
        # that got the class sensitive — they are cross-checked against a computed bound, not a tolerance.
        #
        # THE CHANGE HISTORY OF THE BASELINE: docs/baseline-changes.md.
    """.trimIndent()

    private val EXTRA_HEADER = """
        # AN ADDITIONAL CHARACTERIZATION BASELINE: the combined Nyström,
        # non-uniform grids, the interval [0,2] — what is NOT covered by baseline-eh.tsv.
        #
        # Cross-checked by the test characterization.ExtraCharacterizationTest.
        # ROW FORMAT:  key <TAB> value <TAB> class — the same as for baseline-eh.tsv.
        #   value — 17 significant digits (%.17g, a round-trip for a double) or the marker
        #           NaN / Infinity / -Infinity / ERROR:<class>, compared AS A STRING:
        #           turning a failure into a number is the same change of behaviour as
        #           a change of the number itself;
        #   class — COMPUTED by the tool `./gradlew classifyBaseline` from two snapshots
        #           (-Dnumerics.backend=java against native), see the header of baseline-eh.tsv.
        #           Here occur portable (the E_h values), residual (the *.residual keys,
        #           rel. 1e-3 at the noise 6e-15) and exact (the *.iters counters, strict equality).
        #           There is NO sensitive class in this file: the measurement gave 0 failures on both
        #           backends, that is, the additional matrix has no binding to the LU path.
        # The former choice of the mode by the SUFFIX of the key in the test code is exactly what this replaces: the mode is taken
        # from the data, and adding a scheme does not require editing a chain of `if`s.
        #
        # THE SHOOTING ENVIRONMENT (essential: the numbers are bound to it).
        #   linear algebra backend : netlib + system LAPACK (Apple Accelerate), -Dnumerics.backend=native
        #   libraries              : numerical-core 1.1.0, minimal-splines 1.1.0
        #   JDK                     : 21 (jvmToolchain(21))
        #   platform               : macOS aarch64 (Apple silicon)
        # PORTABILITY IS VERIFIED: the gate is green both on `native` and on `java`; the `machine` tag is removed,
        # the gate runs in CI on ubuntu/OpenBLAS.
        #
        # THE CHANGE HISTORY OF THE BASELINE: docs/baseline-changes.md.
    """.trimIndent()

    @JvmStatic
    fun main(args: Array<String>) {
        val javaDir = File(args.getOrElse(0) { "build/baseline/java" })
        val nativeDir = File(args.getOrElse(1) { "build/baseline/native" })
        val outDir = File(args.getOrElse(2) { "build/baseline/classified" }).apply { mkdirs() }
        val problems = mutableListOf<String>()
        var total = 0
        for ((name, header) in listOf(EH_NAME to EH_HEADER, EXTRA_NAME to EXTRA_HEADER)) {
            total += classifyFile(File(javaDir, name), File(nativeDir, name), File(outDir, name), header, problems)
        }
        if (problems.isNotEmpty()) {
            System.err.println("CLASSIFICATION NOT PERFORMED (${problems.size} problems):")
            problems.take(60).forEach { System.err.println("  $it") }
            exitProcess(1)
        }
        println("Classified $total keys -> ${outDir.absolutePath}")
    }

    private fun readSnapshot(file: File): Map<String, String> {
        check(file.isFile) { "Snapshot not found: ${file.absolutePath} (a captureBaseline* run of both backends is needed)" }
        return file.readLines().mapNotNull { line ->
            val parts = line.trim().split('\t')
            if (parts.size >= 2 && !line.startsWith("#")) parts[0] to parts[1] else null
        }.toMap()
    }

    private fun classifyFile(
        javaFile: File,
        nativeFile: File,
        target: File,
        header: String,
        problems: MutableList<String>,
    ): Int {
        val javaRows = readSnapshot(javaFile)
        val nativeRows = readSnapshot(nativeFile)
        val onlyJava = javaRows.keys - nativeRows.keys
        val onlyNative = nativeRows.keys - javaRows.keys
        if (onlyJava.isNotEmpty() || onlyNative.isNotEmpty()) {
            problems += "${target.name}: the compositions of the snapshots differ (java only: ${onlyJava.size}, " +
                "native only: ${onlyNative.size}) — there is nothing to compare"
            return 0
        }
        val classes = LinkedHashMap<String, BaselineClass>()
        for (key in nativeRows.keys.sorted()) {
            val j = javaRows.getValue(key)
            val n = nativeRows.getValue(key)
            val cls = when {
                key.endsWith(BaselineFormat.ITERATIONS_SUFFIX) -> {
                    if (j != n) {
                        problems += "$key: the iteration counter diverged between the backends (java=$j, native=$n) — " +
                            "this is a regression of the stopping criterion, not a new class"
                    }
                    BaselineClass.EXACT
                }
                key.endsWith(ExtraCharacterizationMatrix.RESIDUAL_SUFFIX) -> BaselineClass.RESIDUAL
                portableAgrees(j, n) -> BaselineClass.PORTABLE
                F1SystemConditioning.supports(key) -> BaselineClass.SENSITIVE
                else -> {
                    problems += "$key: the value depends on the LU path (java=$j, native=$n), but the scheme is NOT base/sloan " +
                        "of the problem F1 — no cond*omega bound exists for it. This is a DEFECT, not a class: " +
                        "find the cause of the discrepancy instead of marking the key sensitive"
                    BaselineClass.PORTABLE
                }
            }
            classes[key] = cls
        }
        promoteGroups(classes)
        target.writeText(
            buildString {
                append(header).append('\n')
                for ((key, cls) in classes) {
                    append(key).append('\t').append(nativeRows.getValue(key)).append('\t').append(cls.tag).append('\n')
                }
            },
        )
        val counts = classes.values.groupingBy { it.tag }.eachCount().toSortedMap()
        println("${target.name}: ${classes.size} rows, classes $counts")
        return classes.size
    }

    /** Whether the values agree by the rule of the class [BaselineClass.PORTABLE]. */
    private fun portableAgrees(expected: String, actual: String): Boolean {
        val e = expected.toDoubleOrNull()
        val a = actual.toDoubleOrNull()
        if (e == null || a == null) return expected == actual
        if (e.isNaN() || a.isNaN()) return e.isNaN() == a.isNaN()
        val difference = abs(a - e)
        if (difference <= BaselineFormat.PORTABLE_ABSOLUTE_FLOOR) return true
        return difference / maxOf(abs(e), BaselineFormat.PORTABLE_ABSOLUTE_FLOOR) <=
            BaselineFormat.PORTABLE_RELATIVE_TOLERANCE
    }

    /** The class is assigned to the triple (system, family, n) as a whole — see the KDoc of the object. */
    private fun promoteGroups(classes: MutableMap<String, BaselineClass>) {
        val sensitiveGroups = classes.filterValues { it == BaselineClass.SENSITIVE }
            .keys.mapNotNull { key -> F1SystemConditioning.parseKey(key)?.let { it } }
            .toSet()
        if (sensitiveGroups.isEmpty()) return
        for (key in classes.keys.toList()) {
            val group = F1SystemConditioning.parseKey(key) ?: continue
            if (group in sensitiveGroups) classes[key] = BaselineClass.SENSITIVE
        }
    }
}
