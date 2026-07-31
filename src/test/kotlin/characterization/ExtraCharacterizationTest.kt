package characterization

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * ДОПОЛНИТЕЛЬНАЯ ХАРАКТЕРИЗАЦИОННАЯ СЕТЬ — закрывает дыры основного гейта
 * [EhCharacterizationTest] перед выносом общего кода решателей в базовый класс.
 *
 * Что именно закрыто (основной эталон `baseline-eh.tsv` этого НЕ покрывает):
 *  - схемы `combinedNystrom` и `iteratedCombinedNystrom` обоих решателей. Их критерии
 *    останова РАЗЛИЧНЫ (Фредгольм — гауссовы узлы `op.gNode`, Вольтерра — `4n+1`
 *    равномерных контрольных точек), поэтому механическое слияние тел изменит число
 *    итераций и результат;
 *  - неравномерные сетки `quasiUniform`, `geometric`, `graded`;
 *  - отрезок `[0,2]` — на нём работает масштабирование `Grid.breakpointInclusionEps`.
 *
 * Состав матрицы и обоснование выбора сочетаний — в KDoc [ExtraCharacterizationMatrix];
 * там же единственное перечисление сочетаний, общее для этого теста и инструмента
 * снятия [ExtraBaselineSnapshotTool] (рассинхронизация невозможна по построению).
 *
 * Эталон: `src/test/resources/characterization/baseline-extra.tsv`. Основной эталон
 * `baseline-eh.tsv` НЕ ЗАТРАГИВАЕТСЯ и остаётся отдельным неприкосновенным гейтом.
 *
 * ЧТО ЭТОТ ТЕСТ ЛОВИТ (проверено мутациями, см. отчёт этапа 4.1):
 *  - подмену критерия останова комбинированного Nyström в решателе Вольтерры
 *    (контрольные точки `4n+1` → гауссовы узлы, как у Фредгольма) — НО ЛОВИТ ЕЁ
 *    ТОЛЬКО КЛЮЧАМИ `*.residual`. Это важный и НЕОЧЕВИДНЫЙ результат замера:
 *    при такой подмене значения E_h и число итераций не меняются НИ НА БИТ (оба
 *    контрольных множества достаточно плотны, итерация приходит в ту же точку за то
 *    же число шагов), и сеть без ключей невязки пропустила бы эту ошибку молча;
 *  - изменение порядка суммирования в цикле сборки матрицы — расхождение выше допуска
 *    (перестановка слагаемых в плавающей арифметике не ассоциативна).
 *
 * ЧЕГО ЭТОТ ТЕСТ НЕ ЛОВИТ (осознанное ограничение допуска, ПРОВЕРЕНО).
 * Сдвиг значения на 1 ULP НЕ обнаруживается. Замер: ключу
 * `F.F2.B.theta.uniform.s01.n8.combNystrom` (эталон 1.11661491164e-08) был сдвинут на
 * один ULP вверх, что дало относительное расхождение 1.48e-16, и тест остался
 * ЗЕЛЁНЫМ — допуск [ExtraCharacterizationMatrix.RELATIVE_TOLERANCE] = 1e-9 на семь
 * порядков грубее.
 *
 * Это выбрано намеренно: перестановка слагаемых при ПАРАЛЛЕЛЬНОЙ сборке матриц
 * законно даёт разницу такого масштаба, и более жёсткий допуск сделал бы сеть
 * недостоверной (ложные срабатывания вместо находок). Практическое следствие:
 * рефакторинг, меняющий только последние биты ЗНАЧЕНИЙ E_h, этот тест пройдёт —
 * и это ожидаемое, а не дефектное поведение. ОГОВОРКА: ключи `*.residual`
 * гораздо чувствительнее (см. [ExtraCharacterizationMatrix.RESIDUAL_RELATIVE_TOLERANCE]).
 *
 * СТОИМОСТЬ И ТЕГ. Прогон — ~17 с (1344 значения), что не укладывается в бюджет
 * fast-набора (255 тестов за ~8 с целиком), поэтому тег — `slow`, а для отдельного
 * запуска заведена задача `./gradlew extraCharacterizationTest`.
 *
 * ФОРМАТ ЭТАЛОНА: `ключ<TAB>значение`. Значение — либо число (12 значащих цифр),
 * либо один из специальных маркеров `NaN`, `Infinity`, `-Infinity`, `ERROR:<класс>`.
 * Маркеры сравниваются ПОДСТРОЧНО, а не численно: снимок фиксирует текущее поведение,
 * включая отказы, и превращение отказа в число — такое же изменение поведения, как
 * и изменение самого числа.
 */
@Tag("slow")
class ExtraCharacterizationTest {

    /** Пара «ключ эталона -> зафиксированная строка значения». */
    private val baseline: Map<String, String> by lazy {
        val resource = javaClass.getResourceAsStream(ExtraCharacterizationMatrix.RESOURCE_PATH)
            ?: fail("Не найден файл эталона ${ExtraCharacterizationMatrix.RESOURCE_PATH}")
        resource.bufferedReader().useLines { lines ->
            lines.mapNotNull { line ->
                val parts = line.trim().split('\t')
                if (parts.size == 2) parts[0] to parts[1] else null
            }.toMap()
        }
    }

    /**
     * Сверяет свежий прогон всей дополнительной матрицы с эталоном.
     *
     * Проверяются обе стороны соответствия: и что каждое вычисленное значение совпало
     * с эталонным, и что в эталоне не осталось ключей, которых больше не производит
     * матрица (иначе удаление сочетания из [ExtraCharacterizationMatrix] тихо сузило бы
     * сеть, а тест остался бы зелёным).
     */
    @Test
    fun extraMatrixMatchesBaseline() {
        val actual = ExtraCharacterizationMatrix.collect()
        val mismatches = mutableListOf<String>()

        for ((key, actualValue) in actual) {
            val expectedValue = baseline[key]
            if (expectedValue == null) {
                mismatches += "$key: отсутствует в эталоне (вычислено $actualValue)"
                continue
            }
            val expectedNumber = expectedValue.toDoubleOrNull()
            val actualNumber = actualValue.toDoubleOrNull()
            if (expectedNumber == null || actualNumber == null) {
                // Хотя бы одна сторона — специальный маркер: сравнение строгое, по строке.
                if (expectedValue != actualValue) {
                    mismatches += "$key: эталон=$expectedValue, получено=$actualValue " +
                        "(специальное значение сравнивается строго)"
                }
                continue
            }
            if (expectedNumber.isNaN() || actualNumber.isNaN()) {
                if (expectedNumber.isNaN() != actualNumber.isNaN()) {
                    mismatches += "$key: эталон=$expectedValue, получено=$actualValue (NaN против числа)"
                }
                continue
            }
            // Ключи невязки сравниваются СВОИМИ порогами — и допуском, и «полом».
            // Общий пол 1e-11 для них губителен: любая сошедшаяся невязка меньше 1e-13,
            // то есть все такие ключи молча выпадали бы из сравнения целиком
            // (см. KDoc [ExtraCharacterizationMatrix.RESIDUAL_ABSOLUTE_FLOOR]).
            val isResidual = key.endsWith(ExtraCharacterizationMatrix.RESIDUAL_SUFFIX)
            val tolerance = if (isResidual) {
                ExtraCharacterizationMatrix.RESIDUAL_RELATIVE_TOLERANCE
            } else {
                ExtraCharacterizationMatrix.RELATIVE_TOLERANCE
            }
            val floor = if (isResidual) {
                ExtraCharacterizationMatrix.RESIDUAL_ABSOLUTE_FLOOR
            } else {
                ExtraCharacterizationMatrix.ABSOLUTE_FLOOR
            }
            // Ранее «оба значения ниже пола» означало ПРОПУСК сравнения, и это было дырой:
            // под общий пол 1e-11 попадают 53 из 672 ключей E_h, включая 3 `combNystrom`
            // и 21 `iterCombNystrom` — целевые схемы этапа. Теперь такая пара не пропускается,
            // а сравнивается по ослабленному, но КОНЕЧНОМУ критерию: относительный допуск
            // 1e-3 при поле 1e-16 (см. KDoc [ExtraCharacterizationMatrix.SMALL_VALUE_RELATIVE_TOLERANCE]).
            // Ключи невязки сюда не попадают: у них свой пол 1e-18, ниже любого их значения.
            val degenerate = abs(expectedNumber) < floor && abs(actualNumber) < floor
            val effectiveTolerance = if (degenerate) {
                ExtraCharacterizationMatrix.SMALL_VALUE_RELATIVE_TOLERANCE
            } else {
                tolerance
            }
            val effectiveFloor = if (degenerate) {
                ExtraCharacterizationMatrix.SMALL_VALUE_ABSOLUTE_FLOOR
            } else {
                floor
            }
            val relative =
                abs(actualNumber - expectedNumber) / maxOf(abs(expectedNumber), effectiveFloor)
            if (relative > effectiveTolerance) {
                mismatches += "$key: эталон=$expectedValue, получено=$actualValue, " +
                    "отн.расхождение=$relative (допуск $effectiveTolerance)"
            }
        }

        val producedKeys = actual.map { it.first }.toSet()
        for (key in baseline.keys.sorted()) {
            if (key !in producedKeys) {
                mismatches += "$key: есть в эталоне, но больше не вычисляется (сочетание исчезло из матрицы)"
            }
        }

        assertTrue(
            mismatches.isEmpty(),
            "Обнаружено изменение численного поведения дополнительной матрицы " +
                "(${mismatches.size} из ${actual.size} значений):\n" +
                mismatches.joinToString("\n").take(6000),
        )
    }
}
