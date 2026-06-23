plugins {
    kotlin("jvm") version "2.0.0"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
}

application {
    // Default run target; per-solver tasks are defined below.
    mainClass.set("solvers.fredholm.FredholmSolverKt")
}

tasks.test {
    useJUnitPlatform()
}

// Per-solver run tasks. Each solver keeps its own fun main().
tasks.register<JavaExec>("runFredholm") {
    group = "application"
    description = "Run the Fredholm solver"
    mainClass.set("solvers.fredholm.FredholmSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runVolterra") {
    group = "application"
    description = "Run the Volterra solver"
    mainClass.set("solvers.volterra.VolterraSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runUryson") {
    group = "application"
    description = "Run the Uryson solver"
    mainClass.set("solvers.uryson.UrysonSolverKt")
    classpath = sourceSets["main"].runtimeClasspath
}
