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
// Сверка со SciPy/NumPy — единственная проверка, не замкнутая на код проекта,
// поэтому она входит в ОБЫЧНЫЙ набор тестов, а не запускается разово.
// Окружение готовится автоматически: иначе на машине без SciPy сверка молча
// пропускалась бы и перестала быть гарантией.

/** Каталог виртуального окружения со SciPy (вне репозитория, см. .gitignore). */
val scipyVenvDir = layout.projectDirectory.dir(".venv-verify")

/** Интерпретатор внутри venv; на Windows структура каталогов иная. */
val scipyPython: String = if (System.getProperty("os.name").startsWith("Windows")) {
    scipyVenvDir.file("Scripts/python.exe").asFile.absolutePath
} else {
    scipyVenvDir.file("bin/python").asFile.absolutePath
}

// Создание venv и установка SciPy/NumPy. Задача идемпотентна: при готовом
// окружении она ничего не делает, поэтому пригодна как зависимость `test`.
// К сети обращается только при первом запуске (далее срабатывает файл-маркер).
tasks.register("setupScipyVerification") {
    group = "verification"
    description = "Подготовить окружение Python со SciPy для внешней сверки"
    val marker = layout.buildDirectory.file("verification/scipy-env.ok")
    outputs.file(marker)
    val pythonPath = scipyPython
    val venvPath = scipyVenvDir.asFile.absolutePath
    doLast {
        if (!File(pythonPath).exists()) {
            logger.lifecycle("Создаётся виртуальное окружение $venvPath")
            exec { commandLine("python3", "-m", "venv", venvPath) }
        }
        val probe = exec {
            commandLine(pythonPath, "-c", "import scipy, numpy")
            isIgnoreExitValue = true
        }
        if (probe.exitValue != 0) {
            logger.lifecycle("Устанавливаются SciPy и NumPy (требуется сеть)")
            exec { commandLine(pythonPath, "-m", "pip", "install", "--quiet", "--upgrade", "pip") }
            exec { commandLine(pythonPath, "-m", "pip", "install", "--quiet", "scipy", "numpy") }
        }
        val markerFile = marker.get().asFile
        markerFile.parentFile.mkdirs()
        markerFile.writeText("Окружение SciPy готово: $pythonPath\n")
    }
}

// --- Разделение тестов по назначению ------------------------------------------
// Обычный `test` — быстрый набор для повседневной работы и CI.
// Из него исключён `BaselineSnapshotTool` — это не проверка, а генератор
// эталонного снимка, который запускается осознанно задачей `captureBaseline`.
tasks.test {
    useJUnitPlatform()
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
        excludeTestsMatching("verification.VerificationArtifactDumpTool")
    }
    // Смок-тест сверки со SciPy требует готового окружения Python.
    dependsOn("setupScipyVerification")
    // Путь к интерпретатору передаётся тесту явно: искать его самостоятельно тест
    // не должен — иначе на разных машинах он находил бы разные интерпретаторы.
    systemProperty("scipy.python", scipyPython)
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
}

// Выгрузка внутренних артефактов (узлы квадратуры, значения базиса, собранные
// матрицы, образы операторов, правые части, E_h) в build/verification/ для
// НЕЗАВИСИМОЙ внешней сверки скриптом tools/verify_with_scipy.py.
// Это не проверка, а генератор данных, поэтому в обычный `test` не входит.
//
// ЗАМЕЧАНИЕ об общей сборке: плагин Kover привязывается ко ВСЕМ задачам типа
// `Test`, поэтому и эта задача, и `captureBaseline` выполняются в составе
// `./gradlew build`. На корректность это не влияет (обе только пишут файлы в
// build/ и ничего не проверяют), но удлиняет сборку. Оставлено как есть:
// побочным эффектом артефакты сверки всегда свежие, а попытка отвязаться от
// Kover потребовала бы вмешательства в его внутреннюю конфигурацию.
tasks.register<Test>("dumpVerificationArtifacts") {
    group = "verification"
    description = "Выгрузить внутренние артефакты в build/verification/ для сверки со SciPy"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("verification.VerificationArtifactDumpTool") }
    outputs.upToDateWhen { false }
}

// --- Настройка измерения покрытия -------------------------------------------
// Цель — высокое покрытие вычислительного ядра и логики решателей.
// Демонстрации, бенчмарк и форматтер чисел живут в отдельном sourceSet `demo`
// и в отчёт не попадают вовсе, поэтому исключений по именам классов не требуется:
// ранее исключавшийся `numerics.Fmt` перенесён в `demo.format.Fmt`.

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
