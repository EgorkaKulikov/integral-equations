package verification

import kotlin.test.Test

/**
 * A DATA GENERATOR FOR MANUAL ANALYSIS (not a check).
 *
 * It dumps the internal computation artifacts into `build/verification/` by the Gradle task
 * `dumpVerificationArtifacts`, without running the cross-check. It is needed when a discrepancy is already
 * detected and the dumped numbers have to be investigated by hand (or processed by one's
 * own script), without waiting for a test run.
 *
 * For the AUTOMATIC cross-check this class is not needed: the smoke test
 * [ScipyCrossVerificationTest] prepares the artifacts itself through [VerificationArtifacts],
 * so it does not depend on whether this task was run before it.
 *
 * It is excluded from the ordinary `test` (see `build.gradle.kts`): this is not a check but a generator.
 */
class VerificationArtifactDumpTool {

    @Test
    fun dumpAllArtifacts() {
        val files = VerificationArtifacts.dumpAll()
        println("Files dumped: ${files.size}")
        for (file in files) println("  ${file.absolutePath} (${file.length()} bytes)")
    }
}
