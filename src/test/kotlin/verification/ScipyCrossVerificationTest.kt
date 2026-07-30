package verification

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
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
 * SciPy незаметно. Постоянной гарантией сверку делает отдельный набор `scipyVerify`,
 * который в CI выполняется на каждой ветке.
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
 * Запуск. Штатный способ — `./gradlew scipyVerify`: эта задача готовит окружение
 * Python (`setupScipyVerification`), выгружает артефакты (`dumpVerificationArtifacts`)
 * и передаёт путь к интерпретатору свойством `scipy.python`.
 *
 * Поведение без окружения Python зависит от СТРОГОГО РЕЖИМА (`scipy.required`):
 *
 *  - `scipy.required=false` или свойство не задано (так работает обычный `test`) —
 *    ПРОПУСК (`Assumptions`). Отсутствие venv — состояние машины, а не
 *    расхождение с SciPy, и трактовать его как провал было бы ложным сигналом;
 *  - `scipy.required=true` (так работает `scipyVerify`) — ПАДЕНИЕ. Здесь сверка
 *    запрошена явно и подготовлена зависимыми задачами, поэтому пропуск означал бы
 *    незамеченную поломку подготовки и ЗЕЛЁНУЮ сборку без единой сверки —
 *    ровно то, чего эта проверка обязана не допускать.
 *
 * Разделение сделано потому, что один и тот же пропуск имеет разный смысл: в
 * повседневном прогоне он ожидаем, в целевом — это дефект.
 */
@Tag("scipy")
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
         * Строгий режим: сверка запрошена явно, окружение обязано быть готово.
         * Выставляется задачей `scipyVerify`; в остальных прогонах отсутствует.
         */
        val STRICT: Boolean = System.getProperty("scipy.required")?.toBoolean() ?: false

        /**
         * Результат сверки в машиночитаемом виде. Разбирается без внешних библиотек
         * разбора JSON: в проекте их нет, а формат отчёта фиксирован и прост.
         */
        var cachedReport: ScipyReport? = null
    }

    /**
     * Требование к ОКРУЖЕНИЮ (не к числам): в строгом режиме нарушение — падение,
     * иначе — пропуск. Сообщение одно и то же: причина и способ исправления нужны
     * в обоих случаях, меняется только статус теста.
     *
     * ПРИМЕЧАНИЕ: к ЧИСЛЕННЫМ расхождениям этот метод не применяется никогда:
     * они всегда дают падение через `assertTrue`, в любом режиме.
     */
    private fun requireEnvironment(condition: Boolean, message: String) {
        if (condition) return
        if (STRICT) {
            fail(
                "$message\n\nЗадача `scipyVerify` требует работоспособного окружения (scipy.required=true): " +
                    "внешняя сверка — единственное доказательство, не замкнутое на код проекта, и тихо " +
                    "пропустить её значит получить зелёную сборку без единой выполненной проверки. " +
                    "Способ исправить: `./gradlew setupScipyVerification` — задача создаст .venv-verify " +
                    "и установит версии из tools/requirements-verify.txt, после чего повторить сверку.",
            )
        }
        assumeTrue(false, message)
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

        val configuredPython: String? = System.getProperty("scipy.python")
        requireEnvironment(
            configuredPython != null,
            "Окружение сверки со SciPy недоступно: не задано свойство scipy.python. Штатный " +
                "запуск — ./gradlew scipyVerify (задача сама готовит окружение и передаёт путь). " +
                "Для ручного запуска укажите -Dscipy.python=<путь к интерпретатору>.",
        )
        val python = configuredPython!!
        requireEnvironment(
            File(python).exists(),
            "Окружение сверки со SciPy недоступно: интерпретатор Python не найден ($python). " +
                "Выполните ./gradlew setupScipyVerification — задача создаст окружение " +
                "и установит SciPy/NumPy версий из tools/requirements-verify.txt.",
        )
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
        // Код 2 означает недоступность SciPy/NumPy в найденном интерпретаторе — случай
        // «venv есть, пакетов нет». Это состояние окружения, а не расхождение чисел,
        // поэтому реакция зависит от строгого режима (см. [requireEnvironment]).
        requireEnvironment(
            exitCode != 2,
            "Окружение сверки со SciPy неработоспособно: SciPy/NumPy недоступны в $python " +
                "(интерпретатор есть, пакетов нет). Выполните ./gradlew setupScipyVerification. " +
                "Вывод скрипта:\n$consoleOutput",
        )
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
