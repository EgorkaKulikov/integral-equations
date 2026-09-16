package characterization

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * A LIMITER OF THE RELAXATION: the class `sensitive` must stand exactly at those keys
 * for which the bound `2*cond*max(omega,eps)*||u||inf` is MEASURED.
 *
 * Why. The comparison mode is taken from the DATA (the third column of the TSV), and this is right —
 * but exactly for that reason a change of the class of a row would SILENTLY weaken the gate: `sensitive` is wider
 * than `portable` by many orders (the bound ~3.8e-5 against the floor 6e-13). An edit of one
 * letter in the resource must not pass unnoticed.
 *
 * The technique is borrowed from `verification.PublishedValuesTest.luPathDependentToleranceCoversExactlyTheDeclaredKeys`:
 * the SET of keys is compared and not their number — otherwise a substitution at an unchanged
 * number (changing the class of one key to sensitive and of another one back) would pass.
 *
 * THE `fast` TAG ON THE CLASS is not an optimization but a requirement on the run frequency: a limiter
 * is pointless if it is executed less often than the baseline changes. The test only parses two
 * resources (a few milliseconds) and does no numerical computation at all.
 */
@Tag("fast")
class BaselineClassGuardTest {

    private companion object {
        /**
         * The DECLARED set of keys of the class `sensitive` — 54 keys of the problem F1.
         *
         * The composition: all three generating systems B/H/T x the families theta/xi1/xi2 x the grids
         * 8/16/32, each combination giving `base` and `sloan`. These are exactly the keys for
         * which (a) the discrepancy of the LU paths is measured (52 of 54 differ, at most
         * 1.25e-05 abs.) and (b) there exists a system `(I-M)c=g` from which the bound
         * is computed. The enumeration is EXPLICIT and not derived from [BaselineSnapshotTool.F1_COVERAGE]:
         * the list must break when the COMPOSITION of the coverage changes, and not follow it.
         */
        val DECLARED_SENSITIVE_KEYS: Set<String> = buildSet {
            for (system in listOf("B", "H", "T")) {
                for (family in listOf("theta", "xi1", "xi2")) {
                    for (n in listOf(8, 16, 32)) {
                        add("F1.$system.$family.n$n.base")
                        add("F1.$system.$family.n$n.sloan")
                    }
                }
            }
        }
    }

    private fun load(path: String): Map<String, BaselineEntry> {
        val resource = javaClass.getResourceAsStream(path) ?: fail("The baseline file $path is not found")
        return resource.bufferedReader().useLines { BaselineFormat.parse(it, path) }
    }

    @Test
    fun sensitiveClassCoversExactlyTheDeclaredKeys() {
        val eh = load("/characterization/baseline-eh.tsv")
        val extra = load(ExtraCharacterizationMatrix.RESOURCE_PATH)

        val actual = (eh + extra).filterValues { it.cls == BaselineClass.SENSITIVE }.keys
        val unexpected = actual - DECLARED_SENSITIVE_KEYS
        val missing = DECLARED_SENSITIVE_KEYS - actual
        assertTrue(
            unexpected.isEmpty() && missing.isEmpty(),
            "The class sensitive stands NOT AT THE KEYS for which the cond*omega bound is measured.\n" +
                "EXTRA (relaxed but not measured), ${unexpected.size}: ${unexpected.sorted()}\n" +
                "MISSING (measured but compared by the strict rule), ${missing.size}: ${missing.sorted()}\n" +
                "If the composition of the baseline has really changed, re-shoot the classification " +
                "(`./gradlew classifyBaseline`) and update the JUSTIFICATION in docs/baseline-changes.md, " +
                "not only the list.",
        )

        // The class sensitive is meaningful only where the bound is computable: otherwise the gate would silently
        // turn into "let everything through" (the comparison would return a computability error).
        val unsupported = actual.filterNot { F1SystemConditioning.supports(it) }
        assertTrue(
            unsupported.isEmpty(),
            "The class sensitive is assigned to keys that have no available system (I-M)c=g: $unsupported",
        )

        // The additional matrix contains no sensitive class at all (the measurement: 0 failures
        // on both backends). Its appearance here would mean an unnoticed regression.
        val sensitiveInExtra = extra.filterValues { it.cls == BaselineClass.SENSITIVE }.keys
        assertTrue(
            sensitiveInExtra.isEmpty(),
            "The class sensitive appeared in baseline-extra.tsv: $sensitiveInExtra",
        )
    }
}
