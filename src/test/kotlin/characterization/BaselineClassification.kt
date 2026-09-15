package characterization

import kotlin.math.abs

/**
 * КЛАСС СТРОКИ ЭТАЛОНА — режим сравнения, взятый ИЗ ДАННЫХ, а не из суффикса ключа.
 *
 * До этого режимов было пять, и выбирались они цепочкой `if` по имени ключа
 * ([ExtraCharacterizationTest] ветвился на `.residual`) либо вовсе не выбирались
 * ([EhCharacterizationTest] сравнивал всё одним допуском). Класс переносит решение
 * в третью колонку TSV, а саму колонку ВЫЧИСЛЯЕТ инструмент [BaselineClassifier]
 * по двум снимкам (`-Dnumerics.backend=java` и `native`): рукописного списка
 * «что сегодня падает» в проекте нет.
 *
 * Классы и их обоснование (все числа — измерение, см. `docs/baseline-changes.md`):
 *  - [PORTABLE] — значение переносимо между путями LU: расхождение бэкендов не выше
 *    `1.0e-14` при поле `6e-13` (запас 60x). Правило прежнее: отн. 1e-9 при поле 6e-13;
 *  - [SENSITIVE] — значение привязано к пути LU: задача F1 (регуляризация Вазваза,
 *    `alpha = 1e-10`, `cond_1 ~ 2.2e10`). Сравнивается с ВЫЧИСЛЯЕМОЙ в прогоне границей
 *    `2*cond*max(omega,eps)*||u||inf` ([F1SystemConditioning]), а не с константой:
 *    константа снова привязала бы эталон к машине;
 *  - [RESIDUAL] — невязка итерационных схем: отн. 1e-3 при шуме 6e-15 (прежнее
 *    поведение ключей `*.residual`);
 *  - [EXACT] — целочисленные счётчики итераций: строгое равенство строк. Ужесточение
 *    разрешено измерением: `.iters` не разошлись НИ РАЗУ (0 из 336 ключей).
 */
enum class BaselineClass(val tag: String) {
    PORTABLE("portable"),
    SENSITIVE("sensitive"),
    RESIDUAL("residual"),
    EXACT("exact"),
    ;

    companion object {
        /** Класс по метке из колонки; неизвестная метка — ОШИБКА, а не «значение по умолчанию». */
        fun ofTag(tag: String): BaselineClass =
            entries.firstOrNull { it.tag == tag }
                ?: error(
                    "неизвестный класс строки эталона '$tag'; допустимы: " +
                        entries.joinToString(", ") { it.tag },
                )
    }
}

/** Строка эталона: зафиксированное значение (как строка) и класс сравнения. */
data class BaselineEntry(val value: String, val cls: BaselineClass)

/**
 * ФОРМАТ ЭТАЛОНА И ЕДИНСТВЕННОЕ МЕСТО СРАВНЕНИЯ для обоих характеризационных гейтов.
 *
 * Формат строки: `ключ <TAB> значение <TAB> класс`. Строки-комментарии начинаются с `#`,
 * пустые игнорируются; ЛЮБАЯ иная строка обязана разбиваться ровно на три части —
 * иначе разбор падает. Прежний парсер («ровно две части, иначе строка молча считается
 * комментарием») превращал опечатку в формате в «ключ отсутствует в эталоне», то есть
 * ослаблял гейт бесшумно.
 *
 * Допуски НЕ дублируются: числа берутся из [ExtraCharacterizationMatrix], где при них
 * стоит измеренное обоснование.
 */
object BaselineFormat {
    /** Отн. допуск класса [BaselineClass.PORTABLE]. */
    const val PORTABLE_RELATIVE_TOLERANCE = ExtraCharacterizationMatrix.RELATIVE_TOLERANCE

    /** Абсолютный «пол» класса [BaselineClass.PORTABLE]: `10^3*eps*||u||inf` при `||u||inf ~ e`. */
    const val PORTABLE_ABSOLUTE_FLOOR = ExtraCharacterizationMatrix.NOISE_FLOOR

    /** Отн. допуск класса [BaselineClass.RESIDUAL]. */
    const val RESIDUAL_RELATIVE_TOLERANCE = ExtraCharacterizationMatrix.RESIDUAL_RELATIVE_TOLERANCE

    /** Шум класса [BaselineClass.RESIDUAL]: ниже него расхождение не проверяется. */
    const val RESIDUAL_NOISE_FLOOR = ExtraCharacterizationMatrix.RESIDUAL_NOISE_FLOOR

    /** «Пол» относительной меры класса [BaselineClass.RESIDUAL]. */
    const val RESIDUAL_ABSOLUTE_FLOOR = ExtraCharacterizationMatrix.RESIDUAL_ABSOLUTE_FLOOR

    /** Суффикс ключей со счётчиком итераций (класс [BaselineClass.EXACT]). */
    const val ITERATIONS_SUFFIX = ".iters"

    /** Разбирает файл эталона. Нарушение формата или неизвестный класс — падение. */
    fun parse(lines: Sequence<String>, source: String): Map<String, BaselineEntry> {
        val result = LinkedHashMap<String, BaselineEntry>()
        var number = 0
        for (raw in lines) {
            number++
            val line = raw.trim()
            if (line.isEmpty() || line.startsWith("#")) continue
            val parts = line.split('\t')
            check(parts.size == 3) {
                "$source:$number: ожидались ровно три колонки «ключ<TAB>значение<TAB>класс», " +
                    "получено ${parts.size}: '$line'"
            }
            val cls = try {
                BaselineClass.ofTag(parts[2])
            } catch (e: IllegalStateException) {
                error("$source:$number: ${e.message}")
            }
            check(result.put(parts[0], BaselineEntry(parts[1], cls)) == null) {
                "$source:$number: ключ '${parts[0]}' встречается в эталоне дважды"
            }
        }
        check(result.isNotEmpty()) { "$source: не разобрано ни одной строки данных" }
        return result
    }

    /**
     * Сравнивает вычисленное значение с эталонным ПО ПРАВИЛУ КЛАССА.
     *
     * Возвращает описание расхождения либо `null`, если значения согласованы.
     * Для класса [BaselineClass.SENSITIVE] граница берётся у [boundFor] — она
     * вычисляется в ТОМ ЖЕ прогоне и на ТОМ ЖЕ бэкенде, поэтому не привязывает
     * эталон к машине.
     */
    fun compare(
        key: String,
        expected: BaselineEntry,
        actual: String,
        boundFor: (String) -> F1SystemConditioning.Measurement? = { F1SystemConditioning.measureFor(it) },
    ): String? {
        val expectedValue = expected.value
        if (expected.cls == BaselineClass.EXACT) {
            return if (expectedValue == actual) {
                null
            } else {
                "$key [exact]: эталон=$expectedValue, получено=$actual (требуется совпадение строк)"
            }
        }
        val e = expectedValue.toDoubleOrNull()
        val a = actual.toDoubleOrNull()
        if (e == null || a == null) {
            // Хотя бы одна сторона — маркер NaN/Infinity/ERROR:<класс>: сравнение строгое, по строке.
            return if (expectedValue == actual) {
                null
            } else {
                "$key: эталон=$expectedValue, получено=$actual (специальное значение сравнивается строго)"
            }
        }
        if (e.isNaN() || a.isNaN()) {
            return if (e.isNaN() == a.isNaN()) null else "$key: эталон=$expectedValue, получено=$actual (NaN против числа)"
        }
        val difference = abs(a - e)
        return when (expected.cls) {
            BaselineClass.PORTABLE -> {
                if (difference <= PORTABLE_ABSOLUTE_FLOOR) {
                    null
                } else {
                    val relative = difference / maxOf(abs(e), PORTABLE_ABSOLUTE_FLOOR)
                    if (relative <= PORTABLE_RELATIVE_TOLERANCE) {
                        null
                    } else {
                        "$key [portable]: эталон=$expectedValue, получено=$actual, " +
                            "отн.расхождение=$relative (допуск $PORTABLE_RELATIVE_TOLERANCE, пол $PORTABLE_ABSOLUTE_FLOOR)"
                    }
                }
            }
            BaselineClass.RESIDUAL -> {
                if (difference <= RESIDUAL_NOISE_FLOOR) {
                    null
                } else {
                    val relative = difference / maxOf(abs(e), RESIDUAL_ABSOLUTE_FLOOR)
                    if (relative <= RESIDUAL_RELATIVE_TOLERANCE) {
                        null
                    } else {
                        "$key [residual]: эталон=$expectedValue, получено=$actual, " +
                            "отн.расхождение=$relative (допуск $RESIDUAL_RELATIVE_TOLERANCE, шум $RESIDUAL_NOISE_FLOOR)"
                    }
                }
            }
            BaselineClass.SENSITIVE -> {
                val m = boundFor(key)
                    ?: return "$key [sensitive]: граница cond*omega для этого ключа не вычислима — " +
                        "класс назначен ключу, у которого нет доступной системы (I-M)c=g"
                if (!m.reliable) {
                    return "$key [sensitive]: оценка обусловленности недостоверна " +
                        "(cond_1=${m.cond}, omega=${m.omega}, граница=${m.bound}, ||u||inf=${m.uNorm})"
                }
                if (difference <= m.bound) {
                    null
                } else {
                    "$key [sensitive]: эталон=$expectedValue, получено=$actual, |dlt|=$difference > " +
                        "граница=${m.bound} (cond_1=${m.cond}, omega=${m.omega}, ||u||inf=${m.uNorm}, " +
                        "|dlt|/граница=${difference / m.bound})"
                }
            }
            BaselineClass.EXACT -> null
        }
    }
}
