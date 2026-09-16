package characterization

import java.io.File
import kotlin.test.Test

/**
 * A utility tool: shoots the ADDITIONAL characterization snapshot
 * (the combined Nyström, non-uniform grids, an interval other than `[0,1]`).
 * The composition of the matrix is described in [ExtraCharacterizationMatrix].
 *
 * This is NOT a checking test: it asserts nothing and always finishes successfully.
 * The result is the file `build/baseline/baseline-extra.tsv`, which after inspection
 * is copied into `src/test/resources/characterization/baseline-extra.tsv` and becomes
 * the baseline for [ExtraCharacterizationTest].
 *
 * Run: `./gradlew captureExtraBaseline`.
 *
 * THE DESIGN OF THE WRITING (since 2026-09-15 the same in [BaselineSnapshotTool] — it was moved
 * there from here; before that these were three differences from the old tool):
 *  - the file name is DETERMINISTIC and does not contain the name of the JUnit thread, otherwise it would depend
 *    on the scheduler, and a comparison of snapshots would have to be done by a pattern;
 *  - the file is OVERWRITTEN and not appended to: with `appendText` a repeated run
 *    without a manual cleanup of the directory doubles the contents;
 *  - the rows are SORTED by key (by the sorting in [ExtraCharacterizationMatrix.collect]),
 *    so a diff of two snapshots shows a change of the numbers and not a permutation of the rows.
 */
class ExtraBaselineSnapshotTool {

    /** Shoots the whole additional matrix into one file. */
    @Test
    fun captureExtraSnapshot() {
        val rows = ExtraCharacterizationMatrix.collect()
        // See [BaselineSnapshotTool]: the directory is set by a property for the sake of two snapshots in a row.
        val dir = File(System.getProperty("baseline.output.dir")?.takeIf { it.isNotBlank() } ?: "build/baseline").apply { mkdirs() }
        val target = File(dir, "baseline-extra.tsv")
        // One write operation instead of a thousand appends: it is both faster and excludes
        // a partially written file on a failure in the middle of the shooting.
        target.writeText(rows.joinToString(separator = "") { (key, value) -> "$key\t$value\n" })
        println("Snapshot of the additional matrix: ${rows.size} rows -> ${target.absolutePath}")
    }
}
