package characterization

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * AN ADDITIONAL CHARACTERIZATION NET — it closes the holes of the main gate
 * [EhCharacterizationTest] before extracting the common code of the solvers into a base class.
 *
 * What exactly is closed (the main baseline `baseline-eh.tsv` does NOT cover this):
 *  - the schemes `combinedNystrom` and `iteratedCombinedNystrom` of both solvers. Their stopping
 *    criteria are DIFFERENT (Fredholm — the Gauss nodes `op.gNode`, Volterra — `4n+1`
 *    uniform control points), so a mechanical merging of the bodies will change the number
 *    of iterations and the result;
 *  - the non-uniform grids `quasiUniform`, `geometric`, `graded`;
 *  - the interval `[0,2]` — on it the scaling of `Grid.breakpointInclusionEps` works.
 *
 * The composition of the matrix and the justification of the choice of the combinations are in the KDoc of [ExtraCharacterizationMatrix];
 * there is also the single enumeration of the combinations, shared by this test and the snapshot
 * tool [ExtraBaselineSnapshotTool] (a desynchronization is impossible by construction).
 *
 * The baseline: `src/test/resources/characterization/baseline-extra.tsv`. The main baseline
 * `baseline-eh.tsv` is NOT TOUCHED and stays a separate untouchable gate.
 *
 * WHAT THIS TEST CATCHES (verified by mutations, see the report of stage 4.1):
 *  - a substitution of the stopping criterion of the combined Nyström in the Volterra solver
 *    (the control points `4n+1` → the Gauss nodes, as in Fredholm) — BUT IT CATCHES IT
 *    ONLY BY THE `*.residual` KEYS. This is an important and NON-OBVIOUS result of the measurement:
 *    under such a substitution the E_h values and the number of iterations do not change BY A SINGLE BIT (both
 *    control sets are dense enough, the iteration arrives at the same point in the same
 *    number of steps), and a net without the residual keys would let this error through silently;
 *  - a change of the summation order in the matrix assembly loop — a discrepancy above the tolerance
 *    (a permutation of the summands in floating-point arithmetic is not associative).
 *
 * WHAT THIS TEST DOES NOT CATCH (a deliberate limitation of the tolerance, VERIFIED).
 * A shift of a value by 1 ULP is NOT detected. Measurement: the key
 * `F.F2.B.theta.uniform.s01.n8.combNystrom` (baseline 1.11661491164e-08) was shifted by
 * one ULP upwards, which gave a relative discrepancy of 1.48e-16, and the test stayed
 * GREEN — the tolerance [ExtraCharacterizationMatrix.RELATIVE_TOLERANCE] = 1e-9 is seven
 * orders coarser.
 *
 * This is chosen on purpose: a permutation of the summands under a PARALLEL assembly of the matrices
 * legitimately gives a difference of this scale, and a stricter tolerance would make the net
 * untrustworthy (false positives instead of findings). The practical consequence:
 * a refactoring changing only the last bits of the E_h VALUES will pass this test —
 * and this is expected and not defective behaviour. A CAVEAT: the `*.residual` keys
 * are far more sensitive (see [ExtraCharacterizationMatrix.RESIDUAL_RELATIVE_TOLERANCE]).
 *
 * THE COST AND THE TAG. The run takes ~17 s (1344 values), which does not fit into the budget of the
 * fast suite (255 tests in ~8 s in total), hence the tag `slow`, while for a separate
 * run the task `./gradlew extraCharacterizationTest` is provided.
 *
 * THE BASELINE FORMAT: `key<TAB>value<TAB>class` (17 significant digits). The value is either
 * a number, or one of the markers `NaN`, `Infinity`, `-Infinity`, `ERROR:<class>`.
 * The markers are compared AS STRINGS and not numerically: the snapshot records the current behaviour,
 * failures included, and turning a failure into a number is the same change of behaviour as
 * a change of the number itself.
 *
 * THE COMPARISON MODE IS TAKEN FROM THE `class` COLUMN, and not from the suffix of the key. Previously there
 * was a chain `if (key.endsWith(".residual"))` here, choosing one of five pairs
 * "tolerance + floor"; now the mode is a property of the DATA ([BaselineClass], [BaselineFormat]),
 * and the column is COMPUTED by the task `./gradlew classifyBaseline`. In this file occur
 * `portable` (the E_h values), `residual` (rel. 1e-3 at the noise 6e-15) and `exact`
 * (the `*.iters` counters — a strict equality of the strings).
 *
 * THE `machine` TAG IS GONE. The measurement: shooting the matrix on `-Dnumerics.backend=java`
 * and on `native` gives 0 gate failures on BOTH backends (the maximum discrepancy is
 * 1.0e-15 abs. for the E_h keys and 1.33e-2 rel. for the `*.residual` keys at a value of ~1e-14,
 * all absorbed by the floors), and the `*.iters` counters never diverged (0 of 336).
 * That is, this baseline had no binding to a machine at all, and the gate runs in CI.
 */
@Tag("slow")
class ExtraCharacterizationTest {

    /** The pair "baseline key -> the recorded value and the comparison class". */
    private val baseline: Map<String, BaselineEntry> by lazy {
        val resource = javaClass.getResourceAsStream(ExtraCharacterizationMatrix.RESOURCE_PATH)
            ?: fail("The baseline file ${ExtraCharacterizationMatrix.RESOURCE_PATH} is not found")
        resource.bufferedReader().useLines {
            BaselineFormat.parse(it, ExtraCharacterizationMatrix.RESOURCE_PATH)
        }
    }

    /**
     * Cross-checks a fresh run of the whole additional matrix against the baseline.
     *
     * Both sides of the correspondence are checked: both that every computed value matched
     * the baseline one, and that no keys are left in the baseline that the matrix no longer
     * produces (otherwise removing a combination from [ExtraCharacterizationMatrix] would quietly narrow
     * the net while the test stayed green).
     */
    @Test
    fun extraMatrixMatchesBaseline() {
        val actual = ExtraCharacterizationMatrix.collect()
        val mismatches = mutableListOf<String>()

        for ((key, actualValue) in actual) {
            val expected = baseline[key]
            if (expected == null) {
                mismatches += "$key: absent from the baseline (computed $actualValue)"
                continue
            }
            BaselineFormat.compare(key, expected, actualValue)?.let { mismatches += it }
        }

        val producedKeys = actual.map { it.first }.toSet()
        for (key in baseline.keys.sorted()) {
            if (key !in producedKeys) {
                mismatches += "$key: present in the baseline but no longer computed (the combination disappeared from the matrix)"
            }
        }

        assertTrue(
            mismatches.isEmpty(),
            "A change of the numerical behaviour of the additional matrix was detected " +
                "(${mismatches.size} of ${actual.size} values):\n" +
                mismatches.joinToString("\n").take(6000),
        )
    }
}
