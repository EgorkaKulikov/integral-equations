package solvers.core

import numerics.NumericsContext
import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * ДИАГНОСТИКА АКТИВНОГО БЭКЕНДА для сборки, в которой сравниваются численные эталоны.
 *
 * Зачем. `Backends.default()` в режиме `auto` при недоступности нативного BLAS/LAPACK
 * МОЛЧА откатывается на чисто-Java реализацию (F2J), а эталоны `baseline-eh.tsv`/
 * `baseline-extra.tsv` привязаны к нативному LAPACK: на другой реализации LU гейт
 * характеризации падает на ключах F1 (cond ~1e10) с расхождением до 5.7e-2 при допуске
 * 1e-9. Без явной проверки такой откат на другой машине или в CI выглядел бы как
 * «рефакторинг испортил числа», и разбирательство ушло бы не туда.
 *
 * Что проверяется. Активный бэкенд согласован со свойством `numerics.backend`:
 * `native` → нативный, `java` → чисто-Java, `auto`/не задано → нативный, если он доступен.
 * Во всех режимах, кроме явного `java`, нативная реализация ОБЯЗАНА быть доступна —
 * иначе падает этот тест, а не сотни характеризационных значений.
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
                "Нативная реализация BLAS/LAPACK недоступна (numerics.backend=$mode). " +
                    "Эталоны characterization/*.tsv сняты на нативном LAPACK и на F2J не воспроизводятся.",
            )
        }
        val expectNative = when (mode) {
            "native" -> true
            "java" -> false
            else -> Backends.isNativeAvailable()
        }
        assertEquals(
            expectNative, Backends.default().isNative,
            "Активный бэкенд по умолчанию — ${Backends.default().name} (isNative=${Backends.default().isNative}), " +
                "ожидался isNative=$expectNative при numerics.backend=$mode. " +
                "Проверьте системное свойство numerics.backend (задачи Gradle передают его явно).",
        )
        assertSame(
            Backends.default(), NumericsContext.default().backend,
            "NumericsContext.default() обязан нести тот же бэкенд, что и Backends.default().",
        )
    }
}
