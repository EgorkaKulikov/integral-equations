package solvers.core

import numerics.NumericsContext
import numerics.backend.Backends
import numerics.backend.MultikCpuBackend
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * ДИАГНОСТИКА АКТИВНОГО БЭКЕНДА для сборки, в которой сравниваются численные эталоны.
 *
 * Зачем. `Backends` при недоступности нативного OpenBLAS МОЛЧА откатывается на
 * `ReferenceBackend`, а эталоны `baseline-eh.tsv`/`baseline-extra.tsv` привязаны к multik:
 * на reference гейт характеризации падает 4/4 с расхождением до 5.7e-2 при допуске 1e-9.
 * Без явной проверки такой откат на другой машине или в CI выглядел бы как «рефакторинг
 * испортил числа», и разбирательство ушло бы не туда.
 *
 * Раньше эту роль в CI играл `numerics.backend.BackendSpiTest.multikCpuBackendIsAvailableHere`;
 * после выделения библиотеки `numerical-core` он живёт там и проверяет сам SPI. Здесь
 * проверяется то, что важно ИМЕННО решателям: активный бэкенд по умолчанию — multik.
 *
 * Тест умышленно не имеет обходного пути: если нативная библиотека не поднялась, падает
 * он, а не сотни характеризационных значений.
 */
@Tag("fast")
class ActiveBackendDiagnosticTest {

    @Test
    fun multikBackendIsAvailableAndActive() {
        assertTrue(
            MultikCpuBackend.isAvailable(),
            "MultikCpuBackend недоступен: нативная библиотека OpenBLAS не загрузилась. " +
                "Эталоны characterization/*.tsv сняты на multik и на reference не воспроизводятся.",
        )
        assertSame(
            MultikCpuBackend, Backends.default(),
            "Активный бэкенд по умолчанию — ${Backends.default().name}, ожидался ${MultikCpuBackend.name}. " +
                "Проверьте системное свойство numerics.backend (задачи Gradle передают его явно).",
        )
        assertSame(
            MultikCpuBackend, NumericsContext.default().backend,
            "NumericsContext.default() обязан нести тот же бэкенд, что и Backends.default().",
        )
    }
}
