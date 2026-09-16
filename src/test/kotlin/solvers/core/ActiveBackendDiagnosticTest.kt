package solvers.core

import numerics.NumericsContext
import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * THE DIAGNOSTICS OF THE ACTIVE BACKEND for a build in which numerical baselines are compared.
 *
 * Why. `Backends.default()` in the `auto` mode, when the native BLAS/LAPACK is unavailable,
 * SILENTLY falls back to the pure-Java implementation (F2J), while the baselines `baseline-eh.tsv`/
 * `baseline-extra.tsv` are bound to the native LAPACK: on another LU implementation the characterization
 * gate fails on the F1 keys (cond ~1e10) with a discrepancy of up to 5.7e-2 at a tolerance of
 * 1e-9. Without an explicit check such a fallback on another machine or in CI would look like
 * "the refactoring spoiled the numbers", and the investigation would go in the wrong direction.
 *
 * What is checked. The active backend agrees with the property `numerics.backend`:
 * `native` → native, `java` → pure Java, `auto`/unset → native if it is available.
 * In all the modes except an explicit `java` the native implementation MUST be available —
 * otherwise this test fails rather than hundreds of characterization values.
 */
@Tag("fast")
class ActiveBackendDiagnosticTest {

    @Test
    fun activeBackendMatchesRequestedMode() {
        println("numerical-core: ${Backends.describe()}")
        val mode = System.getProperty("numerics.backend")?.trim()?.lowercase() ?: "auto"
        if (mode != "java") {
            assertTrue(
                Backends.isNativeAvailable(),
                "The native BLAS/LAPACK implementation is unavailable (numerics.backend=$mode). " +
                    "The baselines characterization/*.tsv were shot on the native LAPACK and are not reproduced on F2J.",
            )
        }
        val expectNative = when (mode) {
            "native" -> true
            "java" -> false
            else -> Backends.isNativeAvailable()
        }
        assertEquals(
            expectNative, Backends.default().isNative,
            "The default active backend is ${Backends.default().name} (isNative=${Backends.default().isNative}), " +
                "isNative=$expectNative was expected at numerics.backend=$mode. " +
                "Check the system property numerics.backend (the Gradle tasks pass it explicitly).",
        )
        assertSame(
            Backends.default(), NumericsContext.default().backend,
            "NumericsContext.default() must carry the same backend as Backends.default().",
        )
    }
}
