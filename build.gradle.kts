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

// --- Разделение тестов по назначению ------------------------------------------
// Обычный `test` — быстрый набор для повседневной работы и CI.
// Из него исключён `BaselineSnapshotTool` — это не проверка, а генератор
// эталонного снимка, который запускается осознанно задачей `captureBaseline`.
tasks.test {
    useJUnitPlatform()
    filter {
        excludeTestsMatching("characterization.BaselineSnapshotTool")
    }
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

// --- Настройка измерения покрытия -------------------------------------------
// Цель — высокое покрытие вычислительного ядра и логики решателей.
// Демонстрации и бенчмарк живут в отдельном sourceSet `demo` и в отчёт не попадают
// вовсе, поэтому исключать их по именам классов больше не требуется.
// Остаются исключёнными только чистые форматтеры: они отвечают за представление
// чисел, а не за алгоритмы.
kover {
    reports {
        filters {
            excludes {
                // Форматирование чисел — задача представления, а не вычислений.
                classes("numerics.Fmt", "numerics.FmtKt")
            }
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
