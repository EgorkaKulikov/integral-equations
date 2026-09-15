package characterization

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * ОГРАНИЧИТЕЛЬ ПОСЛАБЛЕНИЯ: класс `sensitive` обязан стоять ровно у тех ключей,
 * для которых граница `2*cond*max(omega,eps)*||u||inf` ИЗМЕРЕНА.
 *
 * Зачем. Режим сравнения берётся из ДАННЫХ (третья колонка TSV), и это правильно —
 * но ровно поэтому смена класса у строки МОЛЧА ослабила бы гейт: `sensitive` шире
 * `portable` на много порядков (граница ~3.8e-5 против пола 6e-13). Правка одной
 * буквы в ресурсе не должна проходить незамеченной.
 *
 * Приём заимствован у `verification.PublishedValuesTest.luPathDependentToleranceCoversExactlyTheDeclaredKeys`:
 * сравнивается МНОЖЕСТВО ключей, а не их количество — иначе подмена при неизменном
 * количестве (одному ключу сменить класс на sensitive, другому обратно) прошла бы.
 *
 * ТЕГ `fast` НА КЛАССЕ — не оптимизация, а требование к частоте прогона: ограничитель
 * бессмыслен, если исполняется реже, чем меняется эталон. Тест только разбирает два
 * ресурса (единицы миллисекунд) и численных вычислений не делает вовсе.
 */
@Tag("fast")
class BaselineClassGuardTest {

    private companion object {
        /**
         * ОБЪЯВЛЕННОЕ множество ключей класса `sensitive` — 54 ключа задачи F1.
         *
         * Состав: все три порождающие системы B/H/T x семейства theta/xi1/xi2 x сетки
         * 8/16/32, каждое сочетание даёт `base` и `sloan`. Это в точности те ключи, для
         * которых (а) измерено расхождение путей LU (52 из 54 различаются, максимум
         * 1.25e-05 абс.) и (б) существует система `(I-M)c=g`, из которой считается
         * граница. Перечисление ЯВНОЕ, а не производное от [BaselineSnapshotTool.F1_COVERAGE]:
         * список должен ломаться при изменении СОСТАВА покрытия, а не следовать за ним.
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
        val resource = javaClass.getResourceAsStream(path) ?: fail("Не найден файл эталона $path")
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
            "Класс sensitive стоит НЕ У ТЕХ ключей, для которых измерена граница cond*omega.\n" +
                "ЛИШНИЕ (послаблены, но не измерены), ${unexpected.size} шт.: ${unexpected.sorted()}\n" +
                "ПРОПАВШИЕ (измерены, но сравниваются строгим правилом), ${missing.size} шт.: ${missing.sorted()}\n" +
                "Если состав эталона действительно изменился, переснимите классификацию " +
                "(`./gradlew classifyBaseline`) и обновите ОБОСНОВАНИЕ в docs/baseline-changes.md, " +
                "а не только список.",
        )

        // Класс sensitive осмыслен лишь там, где граница вычислима: иначе гейт молча
        // превратился бы в «пропускать всё» (сравнение вернуло бы ошибку вычислимости).
        val unsupported = actual.filterNot { F1SystemConditioning.supports(it) }
        assertTrue(
            unsupported.isEmpty(),
            "Класс sensitive назначен ключам, у которых нет доступной системы (I-M)c=g: $unsupported",
        )

        // Дополнительная матрица класса sensitive не содержит вовсе (измерение: 0 падений
        // на обоих бэкендах). Появление его здесь означало бы незамеченную регрессию.
        val sensitiveInExtra = extra.filterValues { it.cls == BaselineClass.SENSITIVE }.keys
        assertTrue(
            sensitiveInExtra.isEmpty(),
            "В baseline-extra.tsv появился класс sensitive: $sensitiveInExtra",
        )
    }
}
