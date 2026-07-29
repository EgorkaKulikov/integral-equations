package verification

import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * СМОУК-ТЕСТЫ ВНЕШНЕЙ СВЕРКИ СО SciPy/NumPy.
 *
 * Зачем нужны. Все остальные проверки проекта написаны на его же коде и потому
 * подтверждают лишь внутреннюю согласованность: ошибка, внесённая до создания
 * эталона, была бы зафиксирована вместе с ним. Настоящий набор сверяет результаты
 * со СТОРОННИМИ библиотеками (SciPy, NumPy), разработанными независимо, и это
 * единственное доказательство в проекте, не замкнутое на проверяемую реализацию.
 *
 * Почему это тесты, а не разовый скрипт. Разовая сверка защищает от ошибки лишь в
 * момент запуска: любая последующая правка численного ядра могла бы разойтись со
 * SciPy незаметно. Включение в обычный `test` делает сверку постоянной гарантией.
 *
 * Устройство. Тест самодостаточен: сам выгружает артефакты через
 * [VerificationArtifacts] (не полагаясь на то, что перед ним запускалась задача
 * `dumpVerificationArtifacts`), запускает `tools/verify_with_scipy.py` и разбирает
 * его МАШИНОЧИТАЕМЫЙ отчёт. Разбор отчёта, а не только кода возврата, принципиален:
 * при расхождении сообщение называет конкретный слой, величину отклонения и допуск.
 *
 * Слои сверки (снизу вверх, чтобы место расхождения было видно сразу):
 *
 *     L1  квадратура Гаусса--Лежандра      <- numpy leggauss
 *     L2  линейная алгебра, решение СЛАУ   <- scipy.linalg.solve
 *     L3  базис минимальных сплайнов       <- scipy.interpolate.BSpline
 *     L4/L5 образы операторов, правые части<- scipy.integrate.quad (QUADPACK)
 *     L6  итоговые решения                 <- независимый Nystrom на leggauss
 *
 * Окружение Python готовит задача Gradle `setupScipyVerification`, от которой
 * зависит `test`; путь к интерпретатору передаётся свойством `scipy.python`.
 */
class ScipyCrossVerificationTest {

    private companion object {
        /** Скрипт сверки; путь относительно корня проекта. */
        const val SCRIPT_PATH = "tools/verify_with_scipy.py"

        /**
         * Предел времени на прогон сверки. Скрипт выполняется единицы секунд;
         * запас нужен для холодного старта интерпретатора и импорта SciPy.
         * Ограничение обязательно: без него сбой окружения подвесил бы сборку.
         */
        const val TIMEOUT_SECONDS = 300L

        /** Слои, наличие которых в отчёте обязательно. */
        val REQUIRED_LAYERS = listOf("L1", "L2", "L3", "L4/L5", "L6")

        /**
         * Результат сверки в машиночитаемом виде. Разбирается без внешних библиотек
         * разбора JSON: в проекте их нет, а формат отчёта фиксирован и прост.
         */
        var cachedReport: ScipyReport? = null
    }

    /** Одна проверка из отчёта скрипта. */
    private class Check(
        val layer: String,
        val name: String,
        val deviation: Double,
        val tolerance: Double,
        val ok: Boolean,
    )

    /** Разобранный отчёт скрипта сверки. */
    private class ScipyReport(
        val exitCode: Int,
        val numpyVersion: String,
        val scipyVersion: String,
        val checks: List<Check>,
        val failures: List<String>,
        val notes: List<String>,
        val consoleOutput: String,
    )

    /**
     * Готовит артефакты, запускает сверку и разбирает отчёт.
     *
     * Результат кэшируется: скрипт выполняет все слои за один прогон, и повторный
     * запуск для каждого тестового метода лишь умножал бы время сборки.
     */
    private fun report(): ScipyReport {
        cachedReport?.let { return it }

        val artifactDir = VerificationArtifacts.DEFAULT_DIR
        VerificationArtifacts.dumpAll(artifactDir)

        val python = System.getProperty("scipy.python")
            ?: fail(
                "Не задано свойство scipy.python. Тест рассчитан на запуск через Gradle: " +
                    "задача `test` зависит от `setupScipyVerification` и передаёт путь к " +
                    "интерпретатору. Для ручного запуска укажите -Dscipy.python=<путь>.",
            )
        if (!File(python).exists()) {
            fail(
                "Интерпретатор Python не найден: $python. Выполните ./gradlew setupScipyVerification " +
                    "— задача создаст окружение и установит SciPy/NumPy.",
            )
        }
        val script = File(SCRIPT_PATH)
        if (!script.exists()) fail("Не найден скрипт сверки ${script.absolutePath}")

        val jsonFile = File(artifactDir, "scipy-report.json")
        jsonFile.delete()
        val process = ProcessBuilder(
            python,
            script.absolutePath,
            "--artifacts", artifactDir.absolutePath,
            "--json", jsonFile.absolutePath,
        ).redirectErrorStream(true).start()
        val consoleOutput = process.inputStream.bufferedReader().readText()
        val finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        if (!finished) {
            process.destroyForcibly()
            fail("Сверка со SciPy не завершилась за $TIMEOUT_SECONDS с. Вывод:\n$consoleOutput")
        }
        val exitCode = process.exitValue()
        // Код 2 означает недоступность SciPy/NumPy. Это ошибка окружения, а не
        // расхождение, поэтому сообщение указывает способ исправления.
        if (exitCode == 2) {
            fail(
                "SciPy/NumPy недоступны в $python. Выполните ./gradlew setupScipyVerification. " +
                    "Вывод скрипта:\n$consoleOutput",
            )
        }
        if (!jsonFile.exists()) {
            fail(
                "Скрипт сверки не создал машиночитаемый отчёт ${jsonFile.absolutePath} " +
                    "(код возврата $exitCode). Вывод:\n$consoleOutput",
            )
        }
        val parsed = parseReport(jsonFile.readText(), exitCode, consoleOutput)
        cachedReport = parsed
        return parsed
    }

    /**
     * Разбирает отчёт скрипта.
     *
     * Формат создаётся `json.dump` из [SCRIPT_PATH] с отступами, поэтому строчный
     * разбор надёжен: каждое поле находится на своей строке. Отдельная зависимость
     * ради разбора этого файла не оправдана.
     */
    private fun parseReport(text: String, exitCode: Int, consoleOutput: String): ScipyReport {
        fun scalar(field: String): String =
            Regex("\"$field\"\\s*:\\s*\"([^\"]*)\"").find(text)?.groupValues?.get(1) ?: ""

        fun stringList(field: String): List<String> {
            val block = Regex("\"$field\"\\s*:\\s*\\[(.*?)]", RegexOption.DOT_MATCHES_ALL)
                .find(text)?.groupValues?.get(1) ?: return emptyList()
            return Regex("\"((?:[^\"\\\\]|\\\\.)*)\"").findAll(block)
                .map { it.groupValues[1] }
                .toList()
        }

        val checks = mutableListOf<Check>()
        // Один элемент массива checks: поля идут в порядке, заданном скриптом.
        val entry = Regex(
            "\\{\\s*\"layer\"\\s*:\\s*\"([^\"]*)\"\\s*,\\s*\"name\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"\\s*," +
                "\\s*\"deviation\"\\s*:\\s*([-0-9.eE+]+)\\s*,\\s*\"tolerance\"\\s*:\\s*([-0-9.eE+]+)\\s*," +
                "\\s*\"kind\"\\s*:\\s*\"[^\"]*\"\\s*,\\s*\"ok\"\\s*:\\s*(true|false)",
            RegexOption.DOT_MATCHES_ALL,
        )
        for (match in entry.findAll(text)) {
            checks += Check(
                layer = match.groupValues[1],
                name = match.groupValues[2],
                deviation = match.groupValues[3].toDouble(),
                tolerance = match.groupValues[4].toDouble(),
                ok = match.groupValues[5] == "true",
            )
        }
        return ScipyReport(
            exitCode = exitCode,
            numpyVersion = scalar("numpy"),
            scipyVersion = scalar("scipy"),
            checks = checks,
            failures = stringList("failures"),
            notes = stringList("notes"),
            consoleOutput = consoleOutput,
        )
    }

    /** Проверки одного слоя; пустой список означает, что слой не выполнялся. */
    private fun checksOf(layer: String): List<Check> = report().checks.filter { it.layer == layer }

    /**
     * Сводная проверка: расхождений со SciPy нет ни на одном слое.
     *
     * Сообщение об ошибке содержит перечень расхождений с величинами и допусками,
     * а также полный вывод скрипта — иначе диагностика по одному коду возврата
     * была бы невозможна.
     */
    @Test
    fun allLayersAgreeWithScipy() {
        val result = report()
        val failed = result.checks.filterNot { it.ok }
        assertTrue(
            result.exitCode == 0 && failed.isEmpty(),
            buildString {
                appendLine(
                    "Сверка со SciPy ${result.scipyVersion} / NumPy ${result.numpyVersion} " +
                        "обнаружила расхождения (${failed.size} шт., код возврата ${result.exitCode}).",
                )
                appendLine("Расхождение НЕ следует устранять ослаблением допуска: сначала причина.")
                for (check in failed) {
                    appendLine(
                        "  ${check.layer} / ${check.name}: отклонение ${check.deviation} " +
                            "> допуска ${check.tolerance}",
                    )
                }
                for (failure in result.failures) appendLine("  $failure")
                appendLine("--- вывод скрипта ---")
                append(result.consoleOutput.take(4000))
            },
        )
    }

    /**
     * Отчёт обязан содержать ВСЕ слои и не быть пустым.
     *
     * Без этой проверки сверка могла бы «проходить» вырожденно: если скрипт не
     * найдёт артефакты, он лишь запишет замечание и вернёт нулевой код, а зелёный
     * тест создаст ложное впечатление проверенности.
     */
    @Test
    fun everyLayerIsActuallyExecuted() {
        val result = report()
        assertTrue(result.checks.isNotEmpty(), "Отчёт сверки пуст: ни одна проверка не выполнена")
        val missing = REQUIRED_LAYERS.filter { layer -> result.checks.none { it.layer == layer } }
        assertTrue(
            missing.isEmpty(),
            "В отчёте отсутствуют слои: $missing. Вероятно, не выгружены артефакты. " +
                "Замечания скрипта: ${result.notes}",
        )
        assertTrue(
            result.notes.isEmpty(),
            "Скрипт сообщил о непроверенных слоях: ${result.notes}. " +
                "Сверка обязана выполняться полностью, иначе она не является гарантией.",
        )
    }

    /**
     * L1: узлы и веса квадратуры Гаусса–Лежандра совпадают с `numpy leggauss`.
     *
     * Проверка независима по существу: проект вычисляет узлы методом Ньютона по
     * нулям многочлена Лежандра, NumPy использует другой алгоритм.
     */
    @Test
    fun quadratureNodesMatchNumpy() {
        val checks = checksOf("L1")
        assertTrue(checks.size >= 2, "Ожидались проверки узлов и весов, получено ${checks.size}")
        for (check in checks) {
            assertTrue(
                check.ok,
                "L1 ${check.name}: отклонение ${check.deviation} превышает допуск ${check.tolerance}",
            )
        }
    }

    /** L2: решение собранной системы `(I - M) c = g` совпадает со `scipy.linalg.solve`. */
    @Test
    fun linearAlgebraMatchesScipy() {
        val checks = checksOf("L2")
        assertTrue(checks.isNotEmpty(), "Слой L2 не выполнен")
        for (check in checks) {
            assertTrue(
                check.ok,
                "L2 ${check.name}: отклонение ${check.deviation} превышает допуск ${check.tolerance}",
            )
        }
    }

    /**
     * L3: базис минимальных сплайнов системы `B` и две его производные совпадают
     * с `scipy.interpolate.BSpline` на четырёх типах сеток.
     *
     * Сверка возможна лишь для системы `B`: узлы кратности 3 на концах задают
     * клампованный вектор узлов степени 2. Для систем `H` и `T` аналога в SciPy
     * нет (см. docs/REFERENCES.md, раздел 6).
     */
    @Test
    fun splineBasisMatchesScipyBSpline() {
        val checks = checksOf("L3")
        // Четыре сетки по три величины (значение и две производные).
        assertTrue(checks.size >= 12, "Ожидалось не менее 12 проверок L3, получено ${checks.size}")
        for (check in checks) {
            assertTrue(
                check.ok,
                "L3 ${check.name}: отклонение ${check.deviation} превышает допуск ${check.tolerance}",
            )
        }
    }

    /**
     * L4/L5: образы операторов и правые части совпадают с `scipy.integrate.quad`.
     *
     * Это единственная численная проверка формул Лейбница для `(Vu)'` и `(Vu)''`,
     * у которых, согласно docs/REFERENCES.md, отдельной публикации нет.
     */
    @Test
    fun operatorImagesMatchQuadpack() {
        val checks = checksOf("L4/L5")
        assertTrue(checks.isNotEmpty(), "Слой L4/L5 не выполнен")
        for (check in checks) {
            assertTrue(
                check.ok,
                "L4/L5 ${check.name}: отклонение ${check.deviation} превышает допуск ${check.tolerance}",
            )
        }
    }

    /**
     * L6: эталонный метод Nyström на квадратуре Гаусса–Лежандра воспроизводит
     * точные решения модельных задач.
     *
     * Метод реализован средствами NumPy/SciPy без сплайнов и функционалов проекта
     * (учебная схема, Atkinson 1997, гл. 4) и служит внешним эталоном. Проверка
     * подтверждает пригодность самого эталона: если он не воспроизводит точное
     * решение, сверять им нельзя.
     */
    @Test
    fun referenceNystromReproducesExactSolutions() {
        val checks = checksOf("L6")
        assertTrue(checks.isNotEmpty(), "Слой L6 не выполнен")
        for (check in checks) {
            assertTrue(
                check.ok,
                "L6 ${check.name}: отклонение ${check.deviation} превышает допуск ${check.tolerance}",
            )
        }
    }
}
