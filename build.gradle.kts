import org.gradle.api.plugins.jvm.JvmTestSuite
import java.io.ByteArrayOutputStream
import java.io.File
import javax.inject.Inject

plugins {
    kotlin("jvm") version "2.0.0"
    application
    // Coverage measurement (JetBrains Kover, Kotlin-native).
    id("org.jetbrains.kotlinx.kover") version "0.8.3"
}

repositories {
    mavenCentral()
}

// --- Разделение исходных кодов по назначению -------------------------------
// main     — вычислительное ядро: только численные методы, без ввода-вывода,
//            без модельных задач и без точек входа.
// problems — каталог модельных задач (фикстуры): нужен и демонстрациям, и тестам,
//            но НЕ является частью библиотеки.
// demo     — печать таблиц сходимости, точки входа main(), бенчмарк.
sourceSets {
    val main by getting
    val problems by creating {
        compileClasspath += main.output
        runtimeClasspath += main.output
    }
    val demo by creating {
        compileClasspath += main.output + problems.output
        runtimeClasspath += main.output + problems.output
    }
    val test by getting {
        compileClasspath += main.output + problems.output
        runtimeClasspath += main.output + problems.output
    }
}

val problemsImplementation: Configuration by configurations.getting {
    extendsFrom(configurations.implementation.get())
}
val demoImplementation: Configuration by configurations.getting {
    extendsFrom(configurations.implementation.get())
}
configurations {
    named("testImplementation") { extendsFrom(configurations.implementation.get()) }
}

dependencies {
    // Linear-algebra backend: multik with native OpenBLAS (multik-default).
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
}

application {
    // Точка входа по умолчанию; отдельные задачи на каждый решатель описаны ниже.
    mainClass.set("demo.fredholm.FredholmDemoKt")
}

// --- Внешняя сверка со SciPy: подготовка окружения ---------------------------
// Сверка со SciPy/NumPy — единственная проверка, не замкнутая на код проекта.
// Она вынесена в отдельную задачу `scipyVerify` (тег `scipy`), а не входит в
// обычный прогон: повседневная работа не должна требовать интерпретатора Python
// и сети. Постоянной гарантией сверку делает CI, где `scipyVerify` — отдельный job.

/** Каталог виртуального окружения со SciPy (вне репозитория, см. .gitignore). */
val scipyVenvDir = layout.projectDirectory.dir(".venv-verify")

/** Интерпретатор внутри venv; на Windows структура каталогов иная. */
val scipyPython: String = if (System.getProperty("os.name").startsWith("Windows")) {
    scipyVenvDir.file("Scripts/python.exe").asFile.absolutePath
} else {
    scipyVenvDir.file("bin/python").asFile.absolutePath
}

// --- Фиксация бэкенда линейной алгебры в тестах -----------------------------
// Обоснование (по факту, не по осторожности). Эталон `baseline-eh.tsv` снят на
// бэкенде multik/OpenBLAS, и он К НЕМУ ПРИВЯЗАН: прогон
// `-Dnumerics.backend=reference` даёт 4/4 падения `EhCharacterizationTest` с
// расхождением до 5.7e-2 (худший ключ `F1.B.theta.n8.sloan`) при допуске 1e-9 —
// в 5·10^7 раз больше. Причина: F1 — уравнение первого рода, плохо
// обусловленное, и разные реализации LU расходятся на нём закономерно.
//
// Одновременно `Backends.select` при недоступности нативной библиотеки МОЛЧА
// откатывается на `ReferenceBackend`. Без явного выбора такой откат на другой
// машине или в CI выглядел бы как «рефакторинг испортил числа».
//
// Внешнее `-Dnumerics.backend=...` УВАЖАЕТСЯ и перекрывает значение по умолчанию:
// этап 2.2 спека требует гонять `fastTest` НА ОБОИХ бэкендах.
val numericsBackend: String = System.getProperty("numerics.backend") ?: "multik"

/**
 * Подготовка venv со SciPy/NumPy.
 *
 * Почему отдельный класс задачи, а не `doLast { exec { ... } }`: метод `Project.exec`
 * объявлен устаревшим в Gradle 8 и удалён в Gradle 9, а его замена — сервис
 * [ExecOperations], который доступен только через инъекцию в конструктор задачи.
 *
 * Версии зависимостей задаются файлом требований (входом задачи), поэтому смена
 * пиннинга автоматически приводит к переустановке окружения.
 */
abstract class SetupScipyEnvironment @Inject constructor(
    private val execOperations: ExecOperations,
) : DefaultTask() {

    /** Путь к интерпретатору внутри venv. */
    @get:Input
    abstract val pythonPath: Property<String>

    /** Каталог venv (создаётся, если интерпретатора нет). */
    @get:Input
    abstract val venvPath: Property<String>

    /** Файл с закреплёнными версиями SciPy/NumPy. */
    @get:InputFile
    abstract val requirements: RegularFileProperty

    /** Файл-отметка о готовности окружения (для наглядности в build/). */
    @get:OutputFile
    abstract val marker: RegularFileProperty

    @TaskAction
    fun prepare() {
        val python = File(pythonPath.get())
        if (!python.exists()) {
            logger.lifecycle("Создаётся виртуальное окружение ${venvPath.get()}")
            execOperations.exec { commandLine("python3", "-m", "venv", venvPath.get()) }
            // pip из состава свежего venv обычно устарел и спотыкается на колёсах SciPy.
            execOperations.exec {
                commandLine(python.absolutePath, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
            }
        }
        val requirementsFile = requirements.get().asFile
        // Проба С ПРОВЕРКОЙ ВЕРСИЙ, а не «просто импортируется». Зачем она нужна,
        // если всё равно есть `pip install -r`: проба даёт ВНЯТНУЮ ДИАГНОСТИКУ
        // сценария «venv есть, пакетов нет/версии не те»: без неё в журнале не видно,
        // что именно было не так и почему сеть внезапно потребовалась.
        // Сравнение идёт именно с закреплёнными версиями: сверка обязана быть
        // воспроизводимой, иначе расхождение не отличить от смены поведения SciPy.
        //
        // КОНТРАКТ ФОРМАТА файла требований (важно при добавлении новых пакетов):
        // поддерживается только формат `name==version`, и имя пакета обязано совпадать
        // с именем МОДУЛЯ (проба делает `import <name>`). Для `scipy`/`numpy` это верно,
        // но, например, `pyyaml` импортируется как `yaml` — такой пакет потребует
        // явного отображения «пакет -> модуль». Строки без `==` (комментарии, пустые,
        // ограничения вида `>=`) игнорируются пробой — их проверяет только pip.
        val pinned: Map<String, String> = requirementsFile.readLines()
            .map { it.substringBefore('#').trim() }
            .filter { it.contains("==") }
            .associate { it.substringBefore("==").trim() to it.substringAfter("==").trim() }
        // Каждый пакет — СВОЙ `import`. Общий префикс `import a; b` импортировал бы
        // только первый, а остальные стали бы голыми выражениями и давали NameError:
        // проба всегда падала бы, ветка «версии совпали» стала бы мёртвым кодом,
        // а `pip install` запускался бы каждый раз.
        val probeScript = pinned.keys.joinToString("; ") { "import $it" } +
            "; print(" + pinned.keys.joinToString(" + ' ' + ") { "$it.__version__" } + ")"
        // Потоки РАЗВЕДЕНЫ: любой DeprecationWarning от Python идёт в stderr и, будь
        // потоки слиты, попал бы в строку версий и сломал сравнение → лишняя
        // переустановка на исправном окружении. stderr используется только для диагностики.
        val probeOut = ByteArrayOutputStream()
        val probeErr = ByteArrayOutputStream()
        val probe = execOperations.exec {
            commandLine(python.absolutePath, "-c", probeScript)
            standardOutput = probeOut
            errorOutput = probeErr
            isIgnoreExitValue = true
        }
        val installed: List<String> = probeOut.toString("UTF-8").trim()
            .split(Regex("\\s+")).filter { it.isNotEmpty() }
        val matches = probe.exitValue == 0 && installed == pinned.values.toList()
        if (matches) {
            logger.lifecycle(
                "Окружение SciPy соответствует ${requirementsFile.name}: " +
                    pinned.entries.joinToString { "${it.key} ${it.value}" } +
                    " — установка не требуется",
            )
        } else {
            val reason = if (probe.exitValue != 0) {
                "пакеты не импортируются (код ${probe.exitValue}): " +
                    probeErr.toString("UTF-8").trim().lines().lastOrNull()?.take(300).orEmpty()
            } else {
                "версии не совпали: установлено " +
                    pinned.keys.zip(installed) { name, v -> "$name $v" }.joinToString() +
                    ", требуется " + pinned.entries.joinToString { "${it.key} ${it.value}" }
            }
            logger.lifecycle("Окружение SciPy приводится к ${requirementsFile.name} — $reason")
            execOperations.exec {
                commandLine(
                    python.absolutePath, "-m", "pip", "install", "--quiet",
                    "--requirement", requirementsFile.absolutePath,
                )
            }
        }
        val markerFile = marker.get().asFile
        markerFile.parentFile.mkdirs()
        markerFile.writeText(
            "Окружение SciPy готово: ${python.absolutePath}\n" +
                requirementsFile.readLines().filterNot { it.trimStart().startsWith("#") }
                    .filter { it.isNotBlank() }.joinToString("\n", postfix = "\n"),
        )
    }
}

tasks.register<SetupScipyEnvironment>("setupScipyVerification") {
    group = "verification"
    description = "Подготовить окружение Python со SciPy для внешней сверки"
    pythonPath.set(scipyPython)
    venvPath.set(scipyVenvDir.asFile.absolutePath)
    requirements.set(layout.projectDirectory.file("tools/requirements-verify.txt"))
    marker.set(layout.buildDirectory.file("verification/scipy-env.ok"))
    // Актуальность определяется НАЛИЧИЕМ ИНТЕРПРЕТАТОРА, а не файлом-маркером:
    // маркер лежит в build/ и переживал удаление .venv-verify, после чего задача
    // считалась выполненной, а сверка падала невнятным сообщением.
    val pythonFile = File(scipyPython)
    outputs.upToDateWhen { pythonFile.exists() }
}

// --- Разделение тестов по назначению ------------------------------------------
// Единственная задача `test` гоняла ВСЁ и не завершалась за десятки минут, что
// делало невозможной проверку после каждой правки. Поэтому тесты размечены
// тегами на уровне класса, а на каждый тег есть своя задача:
//
//   fast   (206 тестов) — единицы секунд, набор для повседневной работы и каждого PR;
//   slow   (19 тестов)  — прогон по сеткам до n=64: десятки минут, запускается
//                         осознанно перед мержем и по расписанию в CI;
//   scipy  (7 тестов)   — быстрые сами по себе, но требуют venv с Python и
//                         выгруженных артефактов, поэтому вынесены отдельно.
//
// Разметка сделана ПО ЗАМЕРАМ, а не по именам классов: например, `*GoldenTest`
// и `*CoverageTest` решателей укладываются в единицы секунд и попали в fast.
//
// Задача `test` сохранена как полный прогон (все теги), но СЕЙЧАС ОНА НЕ ЗАВЕРШАЕТСЯ
// — см. её KDoc ниже. Из неё, как и из наборов по тегам, исключены генераторы
// данных: это не проверки, а инструменты (см. ниже).
//
// НЕОЧЕВИДНЫЙ МЕХАНИЗМ, который важно не потерять при правках. Любой вызов
// `excludeTestsMatching` включает внутренний флаг `patternFiltersSpecified`, и Gradle 8.9
// тогда бросает `No tests found for given includes`, если не отобралось ни одного
// теста. Именно это защищает все наборы от вырожденного прогона «ноль тестов —
// зелёная сборка» (опции `failOnNoDiscoveredTests` в Gradle 8.9 нет). Защита —
// побочный эффект exclude-фильтров: уберёте их — исчезнет и она.
/**
 * Полный прогон всех 232 тестов — только для ручного запуска.
 *
 * ВНИМАНИЕ: СЕЙЧАС ЭТА ЗАДАЧА НЕ ЗАВЕРШАЕТСЯ. Причина локализована
 * замерами: `verification.PublishedValuesTest` (сетки до n=64,
 * `PublishedValuesTest.kt:187`) прерывался вручную на 600 с, 2400 с и 12 мин,
 * идя на 100-400% CPU. Ускорение горячего пути — ЭТАП 1 СПЕКА
 * (`.tasks/code-review-remediation/SPEC.md`, пп. 1.1-1.2). В этапе 0 численный код
 * не трогаем, поэтому задача оставлена как есть и ВЫВЕДЕНА ИЗ `check`.
 *
 * ПОСЛЕ ЭТАПА 1 пересмотреть состав `check`: когда полный прогон станет
 * выполнимым, есть смысл вернуть туда `test` или `slowTest`.
 */
tasks.test {
    useJUnitPlatform()
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
        excludeTestsMatching("characterization.ExtraBaselineSnapshotTool")
        excludeTestsMatching("verification.VerificationArtifactDumpTool")
    }
    // Путь к интерпретатору передаётся тесту явно: искать его самостоятельно тест
    // не должен — иначе на разных машинах он находил бы разные интерпретаторы.
    // Зависимости от `setupScipyVerification` здесь НЕТ намеренно: обычный прогон
    // не должен требовать Python и сети. Свойство `scipy.required` здесь НЕ задано:
    // без окружения сверка сообщает о пропуске (мягкое поведение).
    systemProperty("scipy.python", scipyPython)
    systemProperty("numerics.backend", numericsBackend)
}

/**
 * СОСТАВ `check` (и следом — `build`): `fastTest` + `characterizationTest`.
 *
 * Почему `test` из `check` исключён. Он тянет все 232 теста, включая
 * `PublishedValuesTest`, который не завершается (см. KDoc `tasks.test`). Значит
 * `./gradlew build` тоже не завершался — стандартная команда сборки была
 * неработоспособна. А неработоспособный `check` перестаёт быть гейтом вовсе.
 *
 * Почему именно эта пара. Вместе они дают и широту (206 тестов fast-набора,
 * ~13 с), и главную гарантию численной нейтральности (1316 значений E_h против
 * эталона, ~90 с) — и обе гарантированно завершаются. `slowTest` сюда НЕ входит
 * по той же причине, что и `test`; `scipyVerify` — потому что требует Python и сети.
 *
 * Замечание о взаимодействии с Kover: `koverVerify` уже входит в `check` и тянет
 * `fastTest` сам — но опираться на это нельзя: состав источников покрытия будет
 * меняться после этапа 1, а требование «`check` гоняет fast-набор» от этого
 * не зависит. Поэтому зависимость объявлена явно.
 */
tasks.named("check") {
    // Убираем ИМЕННО `test`, сохраняя всё остальное (в том числе `koverVerify`),
    // что добавили плагины: жёсткий `setDependsOn(listOf("fastTest", ...))` тихо
    // выкинул бы и будущие проверки.
    //
    // Фильтр ищет НЕ строку и НЕ `TaskProvider`: плагин `java` в Gradle 8.9 привязывает
    // к `check` набор тестов как `Provider<JvmTestSuite>` (проверено выводом типов:
    // `NamedDomainObjectCreatingProvider :: provider(TestSuite 'test', ...)`), и проверка
    // по имени задачи его не ловит.
    setDependsOn(
        dependsOn.filterNot { dependency ->
            val resolved = if (dependency is Provider<*>) dependency.orNull else dependency
            resolved is JvmTestSuite && resolved.name == "test"
        },
    )
    // `extraCharacterizationTest` входит в `check` ОБЯЗАТЕЛЬНО: это единственный числовой
    // гейт схем `combinedNystrom`/`iteratedCombinedNystrom` и неравномерных сеток.
    // Класс помечен тегом `slow`, а `slowTest` из `check` исключён (он красный по
    // независимой причине), поэтому без явной зависимости сеть не запускалась ни
    // в `build`, ни в `check` — только вручную.
    dependsOn("fastTest", "characterizationTest", "extraCharacterizationTest")
}

tasks.register<Test>("fastTest") {
    group = "verification"
    description = "Быстрый набор тестов (тег fast): проверка после каждой правки"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("fast") }
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
        excludeTestsMatching("characterization.ExtraBaselineSnapshotTool")
        excludeTestsMatching("verification.VerificationArtifactDumpTool")
    }
    systemProperty("numerics.backend", numericsBackend)
}

tasks.register<Test>("slowTest") {
    group = "verification"
    description = "Медленный набор тестов (тег slow): прогон по крупным сеткам перед мержем"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("slow") }
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
        excludeTestsMatching("characterization.ExtraBaselineSnapshotTool")
        excludeTestsMatching("verification.VerificationArtifactDumpTool")
    }
    systemProperty("numerics.backend", numericsBackend)
}

/**
 * ГЛАВНЫЙ ГЕЙТ ЧИСЛЕННОЙ НЕЙТРАЛЬНОСТИ рефакторинга.
 *
 * Почему отдельная задача, а не часть `slowTest`. `EhCharacterizationTest` — единственная
 * защита от порчи чисел в последующих этапах: 1316 значений против
 * `src/test/resources/characterization/baseline-eh.tsv` с допуском 1e-9. По времени
 * (~90 с) он отнесён к тегу `slow`, но весь `slowTest` сейчас НЕ ЗАВЕРШАЕТСЯ
 * из-за `PublishedValuesTest` (этап 1 спека) — то есть гейт оказался исполним, но
 * недостижим. Эта задача делает его автоматизируемым сегодня, не дожидаясь этапа 1.
 * Тег класса при этом не меняется: он остаётся `slow` и входит в `slowTest`.
 */
tasks.register<Test>("characterizationTest") {
    group = "verification"
    description = "Гейт численной нейтральности: 1316 значений E_h против эталона, допуск 1e-9"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter {
        includeTestsMatching("characterization.EhCharacterizationTest")
        excludeTestsMatching("characterization.BaselineSnapshotTool")
    }
    // Бэкенд здесь критичен: на `reference` гейт падает 4/4 (см. выше).
    systemProperty("numerics.backend", numericsBackend)
}

/**
 * ДОПОЛНИТЕЛЬНЫЙ ГЕЙТ численной нейтральности — то, что не покрыто `baseline-eh.tsv`.
 *
 * Закрывает три дыры основного гейта перед выносом общего кода решателей
 * (этап 4 спека): схемы `combinedNystrom`/`iteratedCombinedNystrom` (у решателей
 * РАЗНЫЕ критерии останова, и слияние тел изменит числа), неравномерные сетки
 * `quasiUniform`/`geometric`/`graded` и отрезок, отличный от `[0,1]`.
 *
 * Почему ОТДЕЛЬНАЯ задача, а не расширение `characterizationTest`. `characterizationTest`
 * — протокол доказательства нейтральности вокруг НЕПРИКОСНОВЕННОГО эталона
 * `baseline-eh.tsv` (ровно 4 теста, 1316 значений); добавление в него пятого теста
 * с другим эталоном смешало бы два независимых гейта, и падение одного нельзя было бы
 * отличить от падения другого по имени задачи. Эталоны и наборы разведены намеренно.
 *
 * Тег класса — `slow` (фактический прогон ~20 с, в бюджет fast-набора не укладывается),
 * поэтому тест входит и в `slowTest`; эта задача делает его исполнимым отдельно, как и
 * `characterizationTest`, — `slowTest` сейчас красный по независимой причине
 * (`PublishedValuesTest`, этап 8 спека). Именно эта задача, а не `slowTest`,
 * подключена к `check`.
 */
tasks.register<Test>("extraCharacterizationTest") {
    group = "verification"
    description = "Гейт combinedNystrom и неравномерных сеток против baseline-extra.tsv, допуск 1e-9"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter {
        includeTestsMatching("characterization.ExtraCharacterizationTest")
        excludeTestsMatching("characterization.ExtraBaselineSnapshotTool")
    }
    // Тот же довод, что и у `characterizationTest`: снимок привязан к бэкенду.
    systemProperty("numerics.backend", numericsBackend)
}

tasks.register<Test>("scipyVerify") {
    group = "verification"
    description = "Внешняя сверка со SciPy/NumPy (тег scipy): требует окружения Python"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("scipy") }
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
        excludeTestsMatching("characterization.ExtraBaselineSnapshotTool")
        excludeTestsMatching("verification.VerificationArtifactDumpTool")
    }
    // Окружение Python и артефакты сверки — обязательные предпосылки именно этой
    // задачи: тест читает выгрузку из build/verification/.
    dependsOn("setupScipyVerification", "dumpVerificationArtifacts")
    systemProperty("scipy.python", scipyPython)
    systemProperty("numerics.backend", numericsBackend)
    // СТРОГИЙ РЕЖИМ. Здесь сверка запрошена явно и окружение подготовлено
    // зависимостями, поэтому неработоспособное окружение — ДЕФЕКТ, а не обстоятельство.
    // Без этого битый venv (интерпретатор есть, SciPy нет) давал 7 SKIP и ЗЕЛЁНУЮ
    // сборку — то есть единственное внешнее доказательство тихо исчезало.
    systemProperty("scipy.required", "true")
}

// Снятие эталонного снимка численных результатов (в build/baseline/*.tsv).
// Запускается вручную перед обоснованным изменением алгоритма, чтобы
// зафиксировать старое и новое поведение.
tasks.register<Test>("captureBaseline") {
    group = "verification"
    description = "Снять эталонный снимок E_h всех схем в build/baseline/"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("characterization.BaselineSnapshotTool") }
    outputs.upToDateWhen { false }
    // Снимок обязан сниматься на том же бэкенде, с которым его потом сверяют.
    systemProperty("numerics.backend", numericsBackend)
}

// Снятие ДОПОЛНИТЕЛЬНОГО снимка (комбинированный Nyström, неравномерные сетки,
// отрезок [0,2]) в build/baseline/baseline-extra.tsv. В отличие от `captureBaseline`,
// имя файла детерминировано, а файл перезаписывается: повторный запуск даёт тот же
// файл, пригодный для побайтового сравнения.
tasks.register<Test>("captureExtraBaseline") {
    group = "verification"
    description = "Снять дополнительный снимок в build/baseline/baseline-extra.tsv"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("characterization.ExtraBaselineSnapshotTool") }
    outputs.upToDateWhen { false }
    systemProperty("numerics.backend", numericsBackend)
}

// Выгрузка внутренних артефактов (узлы квадратуры, значения базиса, собранные
// матрицы, образы операторов, правые части, E_h) в build/verification/ для
// НЕЗАВИСИМОЙ внешней сверки скриптом tools/verify_with_scipy.py.
// Это не проверка, а генератор данных, поэтому в обычный `test` не входит;
// её запускает задача `scipyVerify`, которой выгрузка нужна по сути.
tasks.register<Test>("dumpVerificationArtifacts") {
    group = "verification"
    description = "Выгрузить внутренние артефакты в build/verification/ для сверки со SciPy"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("verification.VerificationArtifactDumpTool") }
    outputs.upToDateWhen { false }
    // Артефакты выгружаются для внешней сверки, то есть тоже являются числами,
    // сравниваемыми со сторонним эталоном — бэкенд фиксируется так же.
    systemProperty("numerics.backend", numericsBackend)
}

// --- Настройка измерения покрытия -------------------------------------------
// Цель — высокое покрытие вычислительного ядра и логики решателей.
// Демонстрации, бенчмарк и форматтер чисел живут в отдельном sourceSet `demo`
// и в отчёт не попадают вовсе, поэтому исключений по именам классов не требуется:
// ранее исключавшийся `numerics.Fmt` перенесён в `demo.format.Fmt`.
kover {
    currentProject {
        instrumentation {
            // Kover собирает покрытие со ВСЕХ задач типа `Test` и делает каждую
            // зависимостью генерации отчёта, а тот входит в `check`/`build`. Список
            // ниже — единственный рычаг в Kover 0.8.3, который одновременно и выбирает
            // источник покрытия, и убирает задачи из графа.
            //
            // ИСТОЧНИК ПОКРЫТИЯ = `fastTest`, и только он. Выбор сделан по замерам графа,
            // а не по вкусу — были проверены два варианта:
            //
            //   (а) источник `test`  — граф `slowTest koverXmlReport` содержит И `slowTest`,
            //       И `test`: самые дорогие классы прогоняются ДВАЖДЫ, и отчёта нет вовсе;
            //   (б) источники `fastTest` + `slowTest` — отчёт начинает ЗАВИСЕТЬ от `slowTest`,
            //       а тот не завершается (`PublishedValuesTest`, n=64) — отчёта снова нет.
            //
            // Поэтому покрытие снимается с `fastTest` (206 тестов, ~10 с): отчёт всегда
            // фактически строится, и CI-job его действительно публикует. Плата — строки,
            // покрытые только slow-классами, в отчёт не попадают. После этапа 1 спека,
            // когда `slowTest` станет выполнимым, его надо УБРАТЬ из списка ниже — тогда
            // покрытие станет полным (вариант (б) заработает как задумано).
            //
            // Исключены из источников покрытия:
            //   test                 — полный набор, дублирует fastTest+slowTest;
            //   slowTest             — не завершается, снять после этапа 1;
            //   characterizationTest — подмножество slowTest;
            //   scipyVerify          — требует Python, в сборке его быть не обязано;
            //   captureBaseline, dumpVerificationArtifacts — генераторы данных, не проверки.
            disabledForTestTasks.addAll(
                "test",
                "slowTest",
                "characterizationTest",
                "extraCharacterizationTest",
                "scipyVerify",
                "captureBaseline",
                "captureExtraBaseline",
                "dumpVerificationArtifacts",
            )
        }
    }
}

// Демонстрационные запуски: каждый решатель печатает свои таблицы сходимости.
tasks.register<JavaExec>("runFredholm") {
    group = "application"
    description = "Демонстрация: таблицы сходимости для уравнения Фредгольма"
    mainClass.set("demo.fredholm.FredholmDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

tasks.register<JavaExec>("runVolterra") {
    group = "application"
    description = "Демонстрация: таблицы сходимости для уравнения Вольтерры"
    mainClass.set("demo.volterra.VolterraDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

tasks.register<JavaExec>("runUryson") {
    group = "application"
    description = "Демонстрация: таблицы сходимости для уравнения Урысона"
    mainClass.set("demo.uryson.UrysonDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

tasks.register<JavaExec>("runBenchmark") {
    group = "application"
    description = "Бенчмарк производительности (время от N, масштабируемость по потокам)"
    mainClass.set("demo.bench.BenchmarkKt")
    classpath = sourceSets["demo"].runtimeClasspath
}
