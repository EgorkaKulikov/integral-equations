import kotlinx.kover.gradle.plugin.dsl.CoverageUnit
import org.gradle.api.plugins.jvm.JvmTestSuite
import java.io.ByteArrayOutputStream
import java.io.File
import javax.inject.Inject

// Tool versions live in `gradle/libs.versions.toml`; the numerical-core and
// minimal-splines versions stay in `gradle.properties` (the release flow edits them).
plugins {
    alias(libs.plugins.kotlin.jvm)
    // `java-library` — for the sake of the `api` configuration: the types of both libraries
    // (MinimalSplineBasis, Grid, GaussLegendre, NumericsContext, DenseMatrix)
    // appear in the signatures of the public classes of this project.
    `java-library`
    application
    // Coverage measurement (JetBrains Kover, Kotlin-native).
    alias(libs.plugins.kover)
}

repositories {
    // The libraries numerical-core and minimal-splines are consumed as PUBLISHED
    // Maven artifacts rather than as the sources of neighbouring repositories (no
    // `project(":...")` and no `../other-repo/src`). The order: mavenLocal first — a local
    // build (`./gradlew publishToMavenLocal` in each library) takes priority; then
    // GitHub Packages — one registry per library (numerical-core and minimal-splines
    // are published into the registries of their own repositories). GitHub Packages requires
    // authentication even for reading: GITHUB_ACTOR/GITHUB_TOKEN in the environment (in GitHub
    // Actions the built-in token) or gpr.user/gpr.token in ~/.gradle/gradle.properties
    // (a token with read:packages). The content filter restricts the repository to the group of the
    // libraries, so that Gradle does not go to GitHub Packages for the other dependencies.
    mavenLocal()
    maven {
        name = "GitHubPackagesNumericalCore"
        url = uri("https://maven.pkg.github.com/EgorkaKulikov/numerical-core")
        credentials {
            username = providers.environmentVariable("GITHUB_ACTOR").orNull ?: providers.gradleProperty("gpr.user").orNull ?: ""
            password = providers.environmentVariable("GITHUB_TOKEN").orNull ?: providers.gradleProperty("gpr.token").orNull ?: ""
        }
        content { includeGroup("io.github.egorkakulikov") }
    }
    maven {
        name = "GitHubPackagesMinimalSplines"
        url = uri("https://maven.pkg.github.com/EgorkaKulikov/minimal-splines")
        credentials {
            username = providers.environmentVariable("GITHUB_ACTOR").orNull ?: providers.gradleProperty("gpr.user").orNull ?: ""
            password = providers.environmentVariable("GITHUB_TOKEN").orNull ?: providers.gradleProperty("gpr.token").orNull ?: ""
        }
        content { includeGroup("io.github.egorkakulikov") }
    }
    mavenCentral()
    // An additional registry is supplied through the property `numericsRepositoryUrl`.
    providers.gradleProperty("numericsRepositoryUrl").orNull?.let { maven(url = uri(it)) }
}

// --- Separation of the sources by purpose ----------------------------------
// main     — the solvers of the integral equations (Fredholm, Volterra, Uryson) and their
//            common core; no input/output, no model problems and no entry points.
// problems — the catalogue of model problems (fixtures): needed both by the demonstrations and by the tests.
// demo     — printing of the convergence tables, the main() entry points, the benchmark.
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
    // benchmark — performance measurements: not a part of the demonstrations (those have no
    // warm-up, repetitions or aggregation) and not a part of the tests (no PASS/FAIL criterion).
    // A separate source set, as in minimal-splines, and excluded from the coverage.
    val benchmark by creating {
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
val benchmarkImplementation: Configuration by configurations.getting {
    extendsFrom(configurations.implementation.get())
}
configurations {
    named("testImplementation") { extendsFrom(configurations.implementation.get()) }
}

val numericalCoreVersion: String by project
val minimalSplinesVersion: String by project

dependencies {
    // Both libraries are direct dependencies: the solvers use the splines and the
    // quadrature/linear algebra directly. minimal-splines pulls in numerical-core
    // transitively (api), but an explicit declaration pins the version and the intent.
    //
    // `api` rather than `implementation`: the types of both libraries appear in the SIGNATURES
    // of the public code of this project (Grid and MinimalSplineBasis in the constructors
    // of the solvers, GaussLegendre/NumericsContext in their fields, DenseMatrix in the
    // return values), that is, a consumer must see them transitively.
    //
    // The coordinates come from the version catalog and the versions from gradle.properties: they are edited by
    // the library release flow, and the catalog would have hidden them from it.
    api("${libs.numerical.core.get().module}:$numericalCoreVersion")
    api("${libs.minimal.splines.get().module}:$minimalSplinesVersion")

    testImplementation(kotlin("test"))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

kotlin {
    jvmToolchain(21)
    // An explicit visibility is mandatory on EVERY declaration of the non-internal source sets
    // (main, problems, demo, benchmark): the compiler stops inferring `public`
    // for the author, so the appearance of an auxiliary class in the public API
    // becomes a decision rather than a default. The mode does not affect the test source sets.
    explicitApi()
}

// --- Check of the independence from the sources of neighbouring repositories ----
// Both libraries must be present on the compile classpath ONLY as jar artifacts.
// An accidental `project(":...")`/`files("../minimal-splines/build/...")` dependency
// will fail the task. It is part of `check`.
tasks.register("verifyArtifactDependencies") {
    group = "verification"
    description = "Verify that numerical-core and minimal-splines are consumed as artifacts"
    val classpath = configurations.compileClasspath
    doLast {
        val files = classpath.get().files
        val offenders = files.filter { f ->
            !f.name.endsWith(".jar") ||
                f.path.contains("${File.separator}numerical-core${File.separator}build${File.separator}") ||
                f.path.contains("${File.separator}minimal-splines${File.separator}build${File.separator}")
        }
        check(offenders.isEmpty()) { "Dependencies must be jar artifacts from a Maven repository, found: $offenders" }
        for (lib in listOf("numerical-core", "minimal-splines")) {
            val hits = files.filter { it.name.startsWith("$lib-") && it.name.endsWith(".jar") }
            check(hits.size == 1) { "Exactly one artifact of $lib was expected on the classpath, found: $hits" }
            logger.lifecycle("$lib is consumed as an artifact: ${hits.single().name}")
        }
    }
}

application {
    // The default entry point; separate tasks for each solver are described below.
    mainClass.set("demo.fredholm.FredholmDemoKt")
}

/*
 * WHY `run` NEEDS ITS CLASSPATH FIXED.
 *
 * The `application` plugin builds the `run` task over the classpath of the `main` source set,
 * while the entry point `demo.fredholm.FredholmDemoKt` lives in `src/demo` — a separate source set
 * created so that the computational core does not depend on the demonstrations.
 * As a result `./gradlew run` failed:
 *
 *     Error: Could not find or load main class demo.fredholm.FredholmDemoKt
 *     Caused by: java.lang.ClassNotFoundException
 *
 * Why it went unnoticed: all the demonstrations are launched through tasks of their own
 * (`runFredholm`, `runVolterra`, `runUryson`, `runBenchmark`), where the classpath is set explicitly,
 * and `run` is not mentioned in the documentation. Yet the task is still visible in `tasks`
 * and in the IDE, and the first thing a newcomer to a Gradle project tries is precisely `run`.
 *
 * The alternative of "removing the `application` plugin" was rejected: it also provides `distZip`/
 * `installDist`, while the broken task was fixed with a single line.
 */
tasks.named<JavaExec>("run") {
    classpath = sourceSets["demo"].runtimeClasspath
}

// --- External cross-check against SciPy: preparation of the environment ------
// The cross-check against SciPy/NumPy is the only verification not closed over the project code.
// It is placed in a separate task `scipyVerify` (tag `scipy`) and is not part of
// an ordinary run: everyday work must not require a Python interpreter
// and network access. What makes the cross-check a permanent guarantee is CI, where `scipyVerify` is a separate job.

/** The directory of the virtual environment with SciPy (outside the repository, see .gitignore). */
val scipyVenvDir = layout.projectDirectory.dir(".venv-verify")

/** The interpreter inside the venv; on Windows the directory structure is different. */
val scipyPython: String = if (System.getProperty("os.name").startsWith("Windows")) {
    scipyVenvDir.file("Scripts/python.exe").asFile.absolutePath
} else {
    scipyVenvDir.file("bin/python").asFile.absolutePath
}

// --- Pinning of the linear-algebra backend in the tests ---------------------
// The justification (by fact, not by caution). The baseline `baseline-eh.tsv` was captured on the
// multik/OpenBLAS backend and IS BOUND TO IT: a run with
// `-Dnumerics.backend=java` gives 4/4 failures of `EhCharacterizationTest` with a
// divergence of up to 5.7e-2 (the worst key being `F1.B.theta.n8.sloan`) against a tolerance of 1e-9 —
// 5e7 times larger. The reason: F1 is an equation of the first kind, ill
// conditioned, and different LU implementations diverge on it as a matter of course.
//
// At the same time `Backends.select`, when the native library is unavailable, SILENTLY
// falls back to the pure Java backend (`java`). Without an explicit selection such a fallback on another
// machine or in CI would look like "a refactoring spoilt the numbers".
//
// An external `-Dnumerics.backend=...` IS RESPECTED and overrides the default value:
// stage 2.2 of the specification requires running `fastTest` ON BOTH backends.
val numericsBackend: String = System.getProperty("numerics.backend") ?: "auto"

// --- The machine-dependent gate (tag `machine`) -----------------------------
//
// ONE CHECK IS MEANINGFUL ONLY ON THE MACHINE ON WHICH THE BASELINE WAS CAPTURED — the cross-check of F1 against the numbers
// from the article (`PublishedValuesTest.fredholmFirstKindMatchesPublishedValues`).
// On another CPU architecture the native BLAS selects other kernels (NEON against AVX),
// that is, another order of block summation; on the ill-conditioned problems F1
// this goes beyond the tolerance of the cross-check — and the check is red as a matter of course.
//
// THE HISTORY OF THE QUESTION (so as not to return to what was rejected). Formerly the tag `machine`
// was carried by both characterization classes as well: their baselines were compared with a single tolerance of
// 1e-9, and CI on `ubuntu-latest`
// (x86_64) was RED FROM THE VERY FIRST RUN precisely on them, although locally
// they were all green. Two options were considered and rejected:
//  - A SECOND BASELINE for linux-x86_64: it cannot be captured locally (only through CI),
//    while `captureBaseline` is a central routine of the project (see `docs/baseline-changes.md`):
//    every edit of an algorithm would require a double recapture with a round trip through CI.
//    Besides, this does NOT fix `PublishedValuesTest` (see below).
//  - MOVING THE GATES TO `macos-latest`: fewer runners (a queue), 3 cores against 4,
//    that is, `slowTest` would grow from 8.5 min to 15-25; and it would still break on a change of
//    the runner image.
//
// THE DECISION: a separation BY PORTABILITY. It was measured that the machine-dependent
// surface is NARROW, and since 2026-09-16 it has narrowed to a SINGLE method. Both
// characterization classes LOST the tag `machine`: the baselines received a column
// `class`, computed by comparing the snapshots of two LU routes, and both gates are green
// on `-Dnumerics.backend=native` and on `java` alike (see docs/baseline-changes.md,
// the entry of 2026-09-16). The only thing that remained non-portable is
// `PublishedValuesTest.fredholmFirstKindMatchesPublishedValues` (F1, 11 discrepancies
// of up to 11.49 % against a tolerance of 2 %) — and it CANNOT be cured by the same technique: the cross-check is made
// against the numbers FROM THE ARTICLE, and they cannot be "recaptured" for a platform.
//
// LOCALLY THE FLAG IS ENABLED BY DEFAULT; in CI the tag `machine` excludes a SINGLE
// step — `slowTest` in the job `full`, which passes `-PmachineDependentGates=false`.
// The characterization gates are no longer governed by this flag and are run in CI
// unconditionally (the job `characterization`). There is no separate "skip the whole task" function
// any more: no task is left in which ALL the tests are machine-dependent,
// and an exclusion by tag for a single method suffices.
val machineDependentGates: Boolean =
    (findProperty("machineDependentGates") as String?)?.toBoolean() ?: true

// --- Data generators among the test classes ----------------------------------
// These four classes live in `src/test` but contain no checks: they CAPTURE
// baselines and EXPORT artifacts (there are no PASS/FAIL criteria). They are run only by the
// dedicated tasks `captureBaseline`, `captureExtraBaseline`,
// `dumpVerificationArtifacts`, `sec4Tables`; from all the other sets they are
// excluded by name. The list is a SINGLE one for the whole script: formerly the same four lines
// stood in four tasks, and adding a fifth generator required remembering
// all four places.
val generatorTestClasses = listOf(
    "characterization.BaselineSnapshotTool",
    "characterization.ExtraBaselineSnapshotTool",
    "verification.VerificationArtifactDumpTool",
    "verification.Sec4VerificationTool",
)

/**
 * Removes the data generators from a set.
 *
 * THE SIDE EFFECT ON WHICH THE WHOLE BUILD RELIES: any call of
 * `excludeTestsMatching` turns on the flag `patternFiltersSpecified`, and Gradle 8.9
 * then fails with `No tests found for given includes` if not a single test
 * was selected. This is the only protection against the degenerate case "zero tests — a green
 * build" (there is no `failOnNoDiscoveredTests` option in Gradle 8.9).
 */
fun Test.excludeGeneratorTools() {
    filter { generatorTestClasses.forEach { excludeTestsMatching(it) } }
}

/**
 * The common configuration of the TARGETED NUMERICAL GATES — the tasks that select ONE class
 * already part of `slowTest` and exist for the sake of a fast targeted signal
 * (`characterizationTest`, `extraCharacterizationTest`, `convergenceOrderTest`).
 *
 * They differ in exactly one thing — the name of the class — so the body of the task is reduced to
 * a single line. The backend is pinned in all three: each of them compares numbers against a
 * baseline captured on a particular LU implementation.
 */
fun Test.gateOverTestClass(testClass: String) {
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching(testClass) }
    excludeGeneratorTools()
    systemProperty("numerics.backend", numericsBackend)
}

/**
 * Preparation of the venv with SciPy/NumPy.
 *
 * Why a separate task class rather than `doLast { exec { ... } }`: the method `Project.exec`
 * was deprecated in Gradle 8 and removed in Gradle 9, and its replacement is the service
 * [ExecOperations], which is available only through injection into the constructor of a task.
 *
 * The versions of the dependencies are set by the requirements file (an input of the task), so a change of the
 * pinning automatically leads to a reinstallation of the environment.
 */
abstract class SetupScipyEnvironment @Inject constructor(
    private val execOperations: ExecOperations,
) : DefaultTask() {

    /** The path to the interpreter inside the venv. */
    @get:Input
    abstract val pythonPath: Property<String>

    /** The venv directory (it is created if the interpreter is absent). */
    @get:Input
    abstract val venvPath: Property<String>

    /** The file with the pinned versions of SciPy/NumPy. */
    @get:InputFile
    abstract val requirements: RegularFileProperty

    /** A marker file signalling that the environment is ready (for visibility in build/). */
    @get:OutputFile
    abstract val marker: RegularFileProperty

    @TaskAction
    fun prepare() {
        val python = File(pythonPath.get())
        if (!python.exists()) {
            logger.lifecycle("Creating the virtual environment ${venvPath.get()}")
            execOperations.exec { commandLine("python3", "-m", "venv", venvPath.get()) }
            // The pip shipped with a fresh venv is usually outdated and stumbles on the SciPy wheels.
            execOperations.exec {
                commandLine(python.absolutePath, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
            }
        }
        val requirementsFile = requirements.get().asFile
        // A probe WITH A VERSION CHECK rather than "it merely imports". Why it is needed
        // when there is a `pip install -r` anyway: the probe gives CLEAR DIAGNOSTICS for the
        // scenario "the venv exists, the packages are missing or the versions are wrong": without it the log does not show
        // what exactly went wrong and why network access was suddenly required.
        // The comparison is made precisely against the pinned versions: the cross-check must be
        // reproducible, otherwise a discrepancy could not be told apart from a change of the behaviour of SciPy.
        //
        // THE FORMAT CONTRACT of the requirements file (important when adding new packages):
        // only the format `name==version` is supported, and the name of the package must coincide
        // with the name of the MODULE (the probe performs `import <name>`). For `scipy`/`numpy` this holds,
        // but `pyyaml`, for example, is imported as `yaml` — such a package would require
        // an explicit mapping "package -> module". Lines without `==` (comments, empty ones,
        // constraints of the form `>=`) are ignored by the probe — they are checked by pip only.
        val pinned: Map<String, String> = requirementsFile.readLines()
            .map { it.substringBefore('#').trim() }
            .filter { it.contains("==") }
            .associate { it.substringBefore("==").trim() to it.substringAfter("==").trim() }
        // Every package gets its OWN `import`. A common prefix `import a; b` would import
        // only the first one, while the rest would become bare expressions and would give a NameError:
        // the probe would always fail, the branch "the versions matched" would become dead code,
        // and `pip install` would be run every time.
        val probeScript = pinned.keys.joinToString("; ") { "import $it" } +
            "; print(" + pinned.keys.joinToString(" + ' ' + ") { "$it.__version__" } + ")"
        // THE STREAMS ARE SEPARATED: any DeprecationWarning from Python goes to stderr and, were the
        // streams merged, would land in the line of versions and break the comparison → an unnecessary
        // reinstallation on a healthy environment. stderr is used for diagnostics only.
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
                "The SciPy environment matches ${requirementsFile.name}: " +
                    pinned.entries.joinToString { "${it.key} ${it.value}" } +
                    " — no installation is required",
            )
        } else {
            val reason = if (probe.exitValue != 0) {
                "the packages do not import (code ${probe.exitValue}): " +
                    probeErr.toString("UTF-8").trim().lines().lastOrNull()?.take(300).orEmpty()
            } else {
                "the versions did not match: installed " +
                    pinned.keys.zip(installed) { name, v -> "$name $v" }.joinToString() +
                    ", required " + pinned.entries.joinToString { "${it.key} ${it.value}" }
            }
            logger.lifecycle("The SciPy environment is being brought to ${requirementsFile.name} — $reason")
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
            "The SciPy environment is ready: ${python.absolutePath}\n" +
                requirementsFile.readLines().filterNot { it.trimStart().startsWith("#") }
                    .filter { it.isNotBlank() }.joinToString("\n", postfix = "\n"),
        )
    }
}

tasks.register<SetupScipyEnvironment>("setupScipyVerification") {
    group = "verification"
    description = "Prepare the Python environment with SciPy for the external cross-check"
    pythonPath.set(scipyPython)
    venvPath.set(scipyVenvDir.asFile.absolutePath)
    requirements.set(layout.projectDirectory.file("tools/requirements-verify.txt"))
    marker.set(layout.buildDirectory.file("verification/scipy-env.ok"))
    // Up-to-dateness is determined by THE PRESENCE OF THE INTERPRETER rather than by a marker file:
    // the marker lies in build/ and survived a removal of .venv-verify, after which the task
    // was considered executed while the cross-check failed with an obscure message.
    val pythonFile = File(scipyPython)
    outputs.upToDateWhen { pythonFile.exists() }
}

// --- Separation of the tests by purpose ---------------------------------------
// A single `test` task ran EVERYTHING and did not finish within tens of minutes, which
// made a check after every edit impossible. For that reason the tests are marked
// with tags at the class level, and every tag has a task of its own:
//
//   fast   (141 tests)  — a few seconds, the set for everyday work and for every PR;
//   slow   (23 tests)   — a run over grids up to n=64: measured at 8 min 39 s; part of `check`
//                         (through the Kover sources) and of the job `full` in CI;
//   scipy  (9 tests)    — fast in themselves, but they require a venv with Python and
//                         exported artifacts, and are therefore kept separate.
//
// The marking was made BY MEASUREMENT rather than by class names: for example, the `*GoldenTest`
// and `*CoverageTest` of the solvers fit within a few seconds and ended up in fast.
//
// The `test` task is kept as a full run (all the tags) and is EXCLUDED from `check`
// — see its KDoc below. Like the tag-based sets, it excludes the data
// generators: those are not checks but tools (see below).
//
// A NON-OBVIOUS MECHANISM that must not be lost in edits. Any call of
// `excludeTestsMatching` turns on the internal flag `patternFiltersSpecified`, and Gradle 8.9
// then throws `No tests found for given includes` if not a single test
// was selected. It is precisely this that protects all the sets from the degenerate run "zero tests —
// a green build" (there is no `failOnNoDiscoveredTests` option in Gradle 8.9). The protection is a
// side effect of the exclude filters: remove them and it disappears too.
/**
 * A full run of all the tests (fast + slow + scipy) — for a manual invocation only.
 *
 * The task is GREEN (stage 8.6 closed the F1 discrepancy with the publication), but it is kept out of `check`
 * BY TIME rather than by result: it duplicates `fastTest` and `slowTest`
 * together, that is, it runs the most expensive classes again. The gate is provided by the TAG-BASED
 * SETS: `fastTest` (in `check`, a few seconds) and `slowTest` (the separate CI job
 * `full`, run on every push and PR).
 *
 * HISTORY (so as not to return to what was rejected). Before stage 8.6 `slowTest` was
 * RED: `PublishedValuesTest.fredholmFirstKindMatchesPublishedValues` gave 4
 * discrepancies on the keys `F.F1.H.theta.*` against a tolerance of 2 %. It was closed NOT
 * by relaxing the common tolerance (it is still 2 %) but by a narrow tolerance class of 8 %
 * for the six keys of the table `table-f1.tex`, derived from the measured spread
 * between the LU implementations on those very keys (7.267 %); the loss of strictness is
 * compensated by the extension of the F1 coverage in `baseline-eh.tsv` from 4 to 54 keys
 * (tolerance 1e-9). The details are in the KDoc of `PublishedValuesTest.LU_PATH_DEPENDENT_TOLERANCE`
 * and in `docs/baseline-changes.md`.
 */
tasks.test {
    useJUnitPlatform()
    excludeGeneratorTools()
    // The path to the interpreter is passed to the test explicitly: the test must not look for it
    // on its own — otherwise it would find different interpreters on different machines.
    // There is deliberately NO dependency on `setupScipyVerification` here: an ordinary run
    // must not require Python and network access. The property `scipy.required` is NOT set here:
    // without an environment the cross-check reports a skip (the soft behaviour).
    systemProperty("scipy.python", scipyPython)
    systemProperty("numerics.backend", numericsBackend)
}

// The stack of the test JVM. The native LAPACK (OpenBLAS in CI on ubuntu) in a multi-threaded dgetrf
// puts large working arrays on the stack of the calling thread; with the standard -Xss
// this is a SIGSEGV/SIGBUS without a stack trace (exit 139). In CI there is additionally OPENBLAS_NUM_THREADS=4.
tasks.withType<Test>().configureEach {
    jvmArgs("-Xss8m")
}

/**
 * THE COMPOSITION OF `check` (and, following it, of `build`): `fastTest` + `characterizationTest` +
 * `extraCharacterizationTest` + `slowTest` (plus `koverVerify`, added by the plugin).
 *
 * Why `test` is excluded from `check`. It duplicates `fastTest` and `slowTest` together:
 * the most expensive classes would be run twice and the composition of the gate would become implicit.
 *
 * WHY `slowTest` IS NOW PART OF `check` (a change of stage 8.6).
 * Before stage 8.6 it was RED (the F1 discrepancy with the publication) and was therefore kept out of
 * `check`, out of CI and out of the Kover sources. The consequence was worse than a red test: the cross-check against
 * the publication was executed in NO automatic check, that is, the
 * limiter of the width of the relaxation could be widened silently. A set that is green but is run
 * nowhere is not a gate but a decoration.
 *
 * THE MECHANISM of the inclusion is the EXPLICIT `dependsOn("slowTest")` below. The alternative through
 * `disabledForTestTasks` of the `kover` block (removing `slowTest` from there, so that it becomes a
 * source of coverage and thereby a dependency of `koverVerify`) was tried and
 * REJECTED BY FACT: `./gradlew check --rerun-tasks` fails after 7 min 9 s with
 * `Process 'Gradle Test Executor' finished with non-zero exit value 137` (a SIGKILL on
 * memory) — the Kover instrumentation on top of matrices up to n = 64 does not fit into memory,
 * although the same `slowTest` ON ITS OWN passes 23/23. For that reason `slowTest` REMAINED
 * in `disabledForTestTasks` (that is, without instrumentation), while it enters `check` directly.
 * The price of the choice: the lines covered ONLY by the slow classes do not appear in the coverage report —
 * but THE CHECKS THEMSELVES are executed, and that is the main thing. Verified with `check --dry-run`:
 * `:slowTest` is in the graph.
 *
 * THE PRICE: `check` grows from ~1 min to ~9 min (`slowTest` takes 8 min 39 s, of which
 * `PublishedValuesTest` ~190 s). The price is accepted deliberately: `check`/`build` is a
 * pre-commit gate rather than an edit-check cycle. For a fast cycle there are
 * `./gradlew fastTest` (a few seconds) and the triple
 * `fastTest characterizationTest extraCharacterizationTest` (~1 min) — which is what `check`
 * contained before this change.
 *
 * ADDITIONALLY THROUGH CI: the job `full` in `.github/workflows/ci.yml` runs `slowTest`
 * as a separate job on every push and PR — in parallel with the rest, so that
 * the feedback of the job `fast` stays within seconds.
 *
 * INDEPENDENTLY OF ALL THIS, the resource checks of `PublishedValuesTest` —
 * `publishedValuesResourceIsWellFormed` and `luPathDependentToleranceCoversExactlyTheDeclaredKeys`
 * — are marked AT THE METHOD LEVEL with the tag `fast` and are therefore executed also in
 * `fastTest`, that is, in the fast cycle and in the job `fast`. They parse only the resource
 * (a few milliseconds) and catch a silent widening of the relaxation IMMEDIATELY rather than after
 * eight minutes.
 *
 * `convergenceOrderTest` is NOT part of `check` (it stays in `disabledForTestTasks`): its
 * fast subset is already in `fastTest`, while the full matrix is run by the job
 * `characterization` and is part of `slowTest`. `scipyVerify` requires Python and network access.
 *
 * A remark on the interaction with Kover: `koverVerify` is already part of `check` and pulls in
 * `fastTest` by itself — but this must not be relied upon: the composition of the coverage sources may
 * change, while the requirement "`check` runs the fast set" does not depend on that.
 * For that reason the dependency is declared explicitly.
 */
tasks.named("check") {
    // What is removed is PRECISELY `test`, keeping everything else (including `koverVerify`)
    // that the plugins added: a rigid `setDependsOn(listOf("fastTest", ...))` would silently
    // throw out future checks as well.
    //
    // The filter looks for NEITHER a string NOR a `TaskProvider`: the `java` plugin in Gradle 8.9 binds
    // the test suite to `check` as a `Provider<JvmTestSuite>` (verified by type inference:
    // `NamedDomainObjectCreatingProvider :: provider(TestSuite 'test', ...)`), and a check
    // by task name does not catch it.
    setDependsOn(
        dependsOn.filterNot { dependency ->
            val resolved = if (dependency is Provider<*>) dependency.orNull else dependency
            resolved is JvmTestSuite && resolved.name == "test"
        },
    )
    // `characterizationTest` and `extraCharacterizationTest` are listed EXPLICITLY, although both
    // classes are now run as part of `slowTest` as well (which entered `check` through Kover).
    // Why they were not removed as duplicates: the inclusion of `slowTest` rests on the list
    // `disabledForTestTasks` of the `kover` block — an implicit mechanism of a third-party plugin.
    // It may change with a version of Kover or be edited for the sake of coverage, and then
    // the two MAIN numerical gates would silently drop out of `check` together with it. An explicit
    // dependency guarantees their execution independently of the behaviour of Kover; a repeated
    // run of the same two classes costs 46 s against the background of the eight-minute `slowTest` and
    // is worth this insurance.
    //
    // `slowTest` is THE MAIN CHANGE of stage 8.6: without it the only cross-check against an EXTERNAL
    // source of truth (`PublishedValuesTest`, 708 published numbers) and its
    // limiter of the width of the relaxation were executed in NO automatic
    // check (excluded from `check`, commented out in CI) — that is, they were green
    // only because they were not run. See the KDoc above on the price (~9 min) and on
    // why the inclusion was made by an explicit `dependsOn` rather than through the Kover sources.
    dependsOn("fastTest", "characterizationTest", "extraCharacterizationTest", "slowTest", "verifyArtifactDependencies")
}

tasks.register<Test>("fastTest") {
    group = "verification"
    description = "The fast test set (tag fast): the check after every edit"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("fast") }
    excludeGeneratorTools()
    systemProperty("numerics.backend", numericsBackend)
}

tasks.register<Test>("slowTest") {
    group = "verification"
    description = "The slow test set (tag slow): a run over large grids before a merge"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    // The tag `machine` is excluded ONLY under -PmachineDependentGates=false (CI on a foreign
    // architecture). Locally the set is complete: the single F1 method from
    // `PublishedValuesTest` drops out — the remaining 613 published values of the same class
    // are cross-checked in CI as usual. The characterization classes are not governed by this
    // flag: they have a task of their own, and in CI they are run unconditionally.
    useJUnitPlatform {
        includeTags("slow")
        if (!machineDependentGates) excludeTags("machine")
    }
    excludeGeneratorTools()
    systemProperty("numerics.backend", numericsBackend)
}

/**
 * THE MAIN GATE OF THE NUMERICAL NEUTRALITY of a refactoring.
 *
 * Why a separate task rather than merely a part of `slowTest`. `EhCharacterizationTest` is the only
 * protection against a corruption of the numbers: 1366 values against
 * `src/test/resources/characterization/baseline-eh.tsv` with a tolerance of 1e-9. By time
 * (measured at ~27 s) it is assigned to the tag `slow`, while the whole `slowTest` takes 8 min 39 s. A separate
 * task gives a fast signal on the main gate without waiting for the whole slow set and,
 * most importantly, does not depend on whether `slowTest` stays in `check`: a failure of precisely
 * this gate is visible by the name of the task rather than drowning in a common run.
 * The tag of the class is not changed by this: it stays `slow` and is part of `slowTest` as well.
 */
tasks.register<Test>("characterizationTest") {
    group = "verification"
    description = "The numerical-neutrality gate: 1366 values of E_h against the baseline (classes portable/sensitive)"
    // PORTABLE BETWEEN LU ROUTES: verified by a run on `-Dnumerics.backend=java` and `native`.
    // The 52 F1 keys that broke the former gate were separated into the class `sensitive` and are compared against the
    // bound `2*cond*max(omega,eps)*||u||inf`, computed in the same run.
    // For that reason `skipEntirelyIfMachineGatesDisabled()` is NO LONGER here, and the task runs in CI.
    gateOverTestClass("characterization.EhCharacterizationTest")
}

/**
 * AN ADDITIONAL GATE of the numerical neutrality — what is not covered by `baseline-eh.tsv`.
 *
 * It closes three holes of the main gate before the common solver code is extracted
 * (stage 4 of the specification): the schemes `combinedNystrom`/`iteratedCombinedNystrom` (the solvers have
 * DIFFERENT stopping criteria, and a merge of the bodies will change the numbers), the non-uniform grids
 * `quasiUniform`/`geometric`/`graded` and an interval other than `[0,1]`.
 *
 * Why a SEPARATE task rather than an extension of `characterizationTest`. `characterizationTest`
 * is the protocol of the proof of neutrality around the INVIOLABLE baseline
 * `baseline-eh.tsv` (5 tests, 1366 values); adding a test with another baseline to it
 * would mix two independent gates, and a failure of one could not be
 * told from a failure of the other by the name of the task. The baselines and the sets are separated deliberately.
 *
 * The tag of the class is `slow` (the actual run takes ~19 s, which does not fit the budget of the fast set),
 * so the test is part of `slowTest` as well; this task makes it executable separately, as does
 * `characterizationTest`: the whole `slowTest` costs 8 min 39 s, whereas here a fast signal on a
 * particular gate is needed. BOTH tasks are part of `check`: this one (explicitly, as insurance)
 * and `slowTest` in full — see the KDoc of `check` above on the reasons for that duplication.
 */
tasks.register<Test>("extraCharacterizationTest") {
    group = "verification"
    description = "The gate of combinedNystrom and of the non-uniform grids against baseline-extra.tsv (classes portable/residual/exact)"
    // The measurement: 0 failures on both backends, that is, there was no binding to a machine here at all.
    gateOverTestClass("characterization.ExtraCharacterizationTest")
}

/**
 * CLASSIFICATION OF THE BASELINE: the third TSV column is COMPUTED rather than written by hand.
 *
 * It captures both matrices TWICE — on `-Dnumerics.backend=java` (netlib F2J, pure Java) and on
 * `native` (netlib + the system LAPACK) — and compares the snapshots: a key that coincides within
 * the `portable` rule receives that class; a diverging one receives `sensitive`, BUT only
 * if it has a system `(I-M)c=g` (the schemes `base`/`sloan` of the problem F1). A diverging key
 * of any other scheme BREAKS the task: no bound exists for it, and a silent
 * widening of the class would mean switching the gate off. The result is
 * `build/baseline/classified/baseline-{eh,extra}.tsv`, which after an inspection is copied
 * into `src/test/resources/characterization/`.
 *
 * WHY TWO SEPARATE RUNS: `numericsBackend` is computed once per configuration of the
 * project, so the backends cannot be spread over tasks within a single build — the task
 * invokes `./gradlew` again, twice, with different values of `-Dnumerics.backend`.
 * The cost of both captures is about one and a half minutes.
 */
/**
 * Capture of both matrices on TWO backends in succession — a prerequisite of the classification.
 *
 * A separate task type rather than a `doLast` in an ordinary one: `ExecOperations` is available only
 * through injection (in Gradle 8.9 `project.exec` is deprecated and incompatible with the
 * configuration cache). `numericsBackend` is computed once per configuration, so the
 * backends cannot be spread over tasks within a single build — the task invokes `./gradlew`
 * again, twice, with different `-Dnumerics.backend`.
 */
abstract class CaptureBothBackends @Inject constructor(
    private val execOperations: ExecOperations,
) : DefaultTask() {

    @get:Internal
    abstract val gradlewPath: Property<String>

    @get:Internal
    abstract val projectDirectory: Property<String>

    @get:OutputDirectory
    abstract val javaDirectory: DirectoryProperty

    @get:OutputDirectory
    abstract val nativeDirectory: DirectoryProperty

    @TaskAction
    fun capture() {
        for (target in listOf(javaDirectory.get().asFile to "java", nativeDirectory.get().asFile to "native")) {
            val (dir, backend) = target
            dir.mkdirs()
            execOperations.exec {
                commandLine(
                    gradlewPath.get(), "--console=plain", "captureBaseline", "captureExtraBaseline",
                    "-Dnumerics.backend=$backend", "-Dbaseline.output.dir=${dir.absolutePath}",
                )
                workingDir = File(projectDirectory.get())
            }
        }
    }
}

tasks.register<CaptureBothBackends>("captureBaselineBothBackends") {
    group = "verification"
    description = "Capture both matrices on -Dnumerics.backend=java and native (a prerequisite of classifyBaseline)"
    dependsOn("testClasses")
    gradlewPath.set(rootDir.resolve("gradlew").absolutePath)
    projectDirectory.set(projectDir.absolutePath)
    javaDirectory.set(layout.buildDirectory.dir("baseline/java"))
    nativeDirectory.set(layout.buildDirectory.dir("baseline/native"))
    outputs.upToDateWhen { false }
}

/**
 * CLASSIFICATION OF THE BASELINE: the third TSV column is COMPUTED rather than written by hand.
 *
 * It compares the snapshots of the two backends (see `captureBaselineBothBackends`): a key that coincides
 * within the `portable` rule receives that class; the keys `*.iters` receive `exact` and the keys
 * `*.residual` receive `residual`; a diverging key receives `sensitive`, BUT only if
 * it has a system `(I-M)c=g` (the schemes `base`/`sloan` of the problem F1). A diverging key of any
 * other scheme BREAKS the task with an explicit message: no bound `cond*omega` exists for it,
 * and a silent widening of the class would mean switching the gate off on those keys.
 *
 * The result is `build/baseline/classified/baseline-{eh,extra}.tsv` with the values of the NATIVE
 * run; after an inspection it is copied into `src/test/resources/characterization/` by the protocol of
 * `docs/baseline-changes.md`. The full cost is about one and a half minutes (two captures).
 */
tasks.register<JavaExec>("classifyBaseline") {
    group = "verification"
    description = "COMPUTE the class column from the snapshots of both backends (build/baseline/classified)"
    dependsOn("captureBaselineBothBackends")
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("characterization.BaselineClassifier")
    args(
        layout.buildDirectory.dir("baseline/java").get().asFile.absolutePath,
        layout.buildDirectory.dir("baseline/native").get().asFile.absolutePath,
        layout.buildDirectory.dir("baseline/classified").get().asFile.absolutePath,
    )
}

/**
 * THE CONVERGENCE-ORDER GATE.
 *
 * Why a SEPARATE task rather than merely the tag `slow`. The task itself is NOT part of `check`
 * (see below), while the whole `slowTest` costs 8 min 39 s — without a task of its own the check of the orders
 * would be executable only manually or in full together with the whole slow set.
 * The tag of the class is not changed by this: the method of the full matrix stays `slow` and is part
 * of `slowTest`.
 *
 * WHERE IT IS RUN: by the step `Numerical-neutrality and convergence-order gates` in the job `characterization`
 * (`.github/workflows/ci.yml`) — explicitly, on every push and PR. A task invoked neither from `check`
 * nor from CI is not a gate but a manual tool that gets forgotten.
 *
 * ITS PLACE IN THE GRAPH: the task is NOT part of `check`, and this was verified with `check --dry-run`.
 * An explicit action is required: Kover makes a source of coverage out of ANY task of type `Test`
 * not listed in `disabledForTestTasks`, and thereby a dependency of `koverVerify`,
 * which is already in `check`. While the task was not listed there, it silently added one and a half to two
 * minutes to `check`, although it added NO COVERAGE at all: the full matrix
 * executes the same lines of the solvers as the fast subset of the same class, only
 * on a larger number of grids and systems. There is no sense in paying that time on every build for a zero
 * gain in coverage, so the task was entered into `disabledForTestTasks`
 * of the `kover` block. The full matrix is nevertheless still executed in `check` — as part of
 * `slowTest`, which is included there explicitly.
 *
 * What is NOT lost by this: a degradation of the order is caught by the fast subset (tag `fast`,
 * 32 combinations), which runs as part of `fastTest` on every edit: a mutation
 * check (a return of the piecewise-linear reconstruction in `kulkarniQuasi`) breaks it on
 * 4 combinations out of 32. This task is the full matrix before a merge, on a par with
 * `characterizationTest` (under the same mutation 24 combinations out of 168 fail).
 *
 * THE MEASUREMENT OF THE FULL MATRIX (`--rerun`, the multik backend, the developer machine): 168 combinations
 * up to n = 64. Two successive measurements: 91 s and 105 s wall-clock for the whole task (of which the
 * method of the full matrix itself took 79 s and 99 s). The spread between the runs is about 20 % — this is
 * a property of the measurement (JIT and the thermal regime of the CPU), so the RANGE "1.5-2 min" is stated everywhere
 * rather than a single exact number: an exact number would create a false precision here.
 */
tasks.register<Test>("convergenceOrderTest") {
    group = "verification"
    description = "The full matrix of convergence orders: 168 combinations on the grids 8/16/32/64"
    // The same argument as for the other numerical gates: the table of orders was captured on multik.
    gateOverTestClass("convergence.ConvergenceOrderTest")
}

tasks.register<Test>("scipyVerify") {
    group = "verification"
    description = "External cross-check against SciPy/NumPy (tag scipy): it requires a Python environment"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("scipy") }
    excludeGeneratorTools()
    // The Python environment and the cross-check artifacts are mandatory prerequisites of precisely this
    // task: the test reads the export from build/verification/.
    dependsOn("setupScipyVerification", "dumpVerificationArtifacts")
    systemProperty("scipy.python", scipyPython)
    systemProperty("numerics.backend", numericsBackend)
    // THE STRICT MODE. Here the cross-check is requested explicitly and the environment is prepared
    // by the dependencies, so an inoperative environment is a DEFECT rather than a circumstance.
    // Without this a broken venv (an interpreter is present, SciPy is not) would give 9 SKIPs and a GREEN
    // build — that is, the only external piece of evidence would quietly disappear.
    systemProperty("scipy.required", "true")
}

// Capture of the reference snapshot of the numerical results (into build/baseline/*.tsv).
// It is run manually before a justified change of an algorithm, in order to
// record the old and the new behaviour.
tasks.register<Test>("captureBaseline") {
    group = "verification"
    description = "Capture the reference snapshot of E_h of all the schemes into build/baseline/"
    // The output directory is set from outside: `classifyBaseline` captures the matrix twice in succession.
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("characterization.BaselineSnapshotTool") }
    // The output directory is set from outside: `classifyBaseline` captures the matrix twice in succession.
    providers.systemProperty("baseline.output.dir").orNull?.let { systemProperty("baseline.output.dir", it) }
    outputs.upToDateWhen { false }
    // The snapshot must be captured on the same backend against which it is later compared.
    systemProperty("numerics.backend", numericsBackend)
}

// Capture of the ADDITIONAL snapshot (combined Nyström, non-uniform grids,
// the interval [0,2]) into build/baseline/baseline-extra.tsv. Unlike `captureBaseline`,
// the name of the file is deterministic and the file is overwritten: a repeated run gives the same
// file, suitable for a byte-for-byte comparison.
tasks.register<Test>("captureExtraBaseline") {
    group = "verification"
    description = "Capture the additional snapshot into build/baseline/baseline-extra.tsv"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("characterization.ExtraBaselineSnapshotTool") }
    providers.systemProperty("baseline.output.dir").orNull?.let { systemProperty("baseline.output.dir", it) }
    outputs.upToDateWhen { false }
    systemProperty("numerics.backend", numericsBackend)
}

// Export of the internal artifacts (the quadrature nodes, the values of the basis, the assembled
// matrices, the images of the operators, the right-hand sides, E_h) into build/verification/ for an
// INDEPENDENT external cross-check by the script tools/verify_with_scipy.py.
// This is not a check but a data generator, so it is not part of an ordinary `test`;
// it is run by the task `scipyVerify`, which needs the export in substance.
// Generation of the tables of §5--6 of the article on minimal splines (the model problems M1--M5,
// frequency-tuned generating systems) into build/sec4/. Like the baseline snapshots,
// this is a GENERATOR OF NUMBERS rather than a check: it has no PASS/FAIL criteria, so it is not part of
// `test`, `fastTest`, `slowTest` or `scipyVerify`.
// The number of nodes of the composite quadrature is set by `-Dsec4.quad` (8 by default);
// doubling the nodes serves as a control of the stability of the quantities cited in the article.
tasks.register<Test>("sec4Tables") {
    group = "verification"
    description = "Generate the tables of §5--6 of the article into build/sec4/ (-Dsec4.quad=8|16)"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("verification.Sec4VerificationTool") }
    outputs.upToDateWhen { false }
    // The numbers are compared between runs, so the backend is pinned, as it is for the snapshots.
    systemProperty("numerics.backend", numericsBackend)
    providers.systemProperty("sec4.quad").orNull?.let { systemProperty("sec4.quad", it) }
}

tasks.register<Test>("dumpVerificationArtifacts") {
    group = "verification"
    description = "Export the internal artifacts into build/verification/ for the cross-check against SciPy"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter { includeTestsMatching("verification.VerificationArtifactDumpTool") }
    outputs.upToDateWhen { false }
    // The artifacts are exported for an external cross-check, that is, they are also numbers
    // compared with a third-party reference — the backend is pinned in the same way.
    systemProperty("numerics.backend", numericsBackend)
}

// --- Configuration of the coverage measurement -------------------------------
// The aim is a high coverage of the computational core and of the logic of the solvers.
// The demonstrations, the benchmark and the number formatter live in the separate source set `demo`
// and do not appear in the report at all, so no exclusions by class name are required:
// the formerly excluded `numerics.Fmt` was moved into `demo.format.Fmt`.
kover {
    currentProject {
        instrumentation {
            // Kover collects coverage from ALL tasks of type `Test` and makes each of them a
            // dependency of the report generation, and the report is part of `check`/`build`. The list
            // below is the only lever in Kover 0.8.3 that at once selects the
            // source of coverage and removes tasks from the graph.
            //
            // THE SOURCE OF COVERAGE = `fastTest`, and it alone. The choice was made BY MEASUREMENT
            // rather than by taste — three options were tried:
            //
            //   (a) the source `test`  — the graph `slowTest koverXmlReport` contains BOTH `slowTest`
            //       AND `test`: the most expensive classes are run TWICE, and there is no report at all;
            //   (b) the sources `fastTest` + `slowTest` — TRIED AT STAGE 8.6 and REJECTED
            //       BY FACT. After the F1 discrepancy was closed, `slowTest` is green and the option
            //       started to look usable — but the run `./gradlew check --rerun-tasks`
            //       FAILS after 7 min 9 s: `Process 'Gradle Test Executor' finished with
            //       non-zero exit value 137`, that is, a SIGKILL on memory. The reason: under
            //       the Kover instrumentation the same `slowTest` that ON ITS OWN passes
            //       23/23 in 8 min 39 s stops fitting into memory: it already needs
            //       matrices up to n = 64 plus 1366 baseline values;
            //   (c) `fastTest` only — the CHOSEN option: the report is always built and the CI job
            //       publishes it. The price: the lines covered only by the slow classes do not appear in the
            //       report. The price is accepted: the slow classes themselves are executed in `check`
            //       DIRECTLY through `dependsOn("slowTest")`, that is, THE GATE is there and only the
            //       COVERAGE is lost, not the check. Coverage as an aim is secondary in
            //       comparison with a working build.
            //
            // IMPORTANT WHEN EDITING: `slowTest` MUST NOT be removed from here without an increase of the
            // `maxHeapSize` of the task — the build fails on OOM (option (b) above).
            //
            // Excluded from the sources of coverage:
            //   test                      — the full set, it duplicates fastTest+slowTest;
            //   slowTest                  — it fails under the instrumentation (exit 137, see (b));
            //                               it enters `check` by an explicit `dependsOn` rather than through Kover;
            //   characterizationTest      — a subset of slowTest;
            //   extraCharacterizationTest — the same;
            //   convergenceOrderTest      — the same (the fast subset is already in fastTest);
            //   scipyVerify               — it requires Python and need not be part of a build;
            //   captureBaseline, captureExtraBaseline, dumpVerificationArtifacts, sec4Tables —
            //                               data generators rather than checks. `sec4Tables` is MANDATORY
            //                               here: without it `koverXmlReport` pulled it into the job
            //                               `fast` of CI (~18 min of generating the tables of the article), and the job
            //                               failed on the 20-minute timeout — this was the case in the original
            //                               monorepository as well after that task appeared.
            disabledForTestTasks.addAll(
                "test",
                "slowTest",
                "characterizationTest",
                "extraCharacterizationTest",
                "convergenceOrderTest",
                "scipyVerify",
                "captureBaseline",
                "captureExtraBaseline",
                "dumpVerificationArtifacts",
                "sec4Tables",
            )
        }
        sources {
            // Performance measurements are not solver code and do not participate in the coverage.
            excludedSourceSets.add("benchmark")
        }
    }
    reports {
        // THE COVERAGE BAR held by `koverVerify` (it is part of `check`).
        // Before this block `koverVerify` in `check` passed IDLY: there were no rules
        // at all, that is, the task was green under any coverage.
        //
        // The numbers are MEASURED rather than assigned: the source of coverage is `fastTest` only
        // (see `disabledForTestTasks` above), on which the lines are 69.2 % (1379 of 1993) and the
        // branches 74.5 % (502 of 674). The threshold is the measured value minus 2 points: a margin for the
        // machine-dependent branches (the selection of the backend) and for rounding.
        //
        // WHY THE BAR IS LOWER THAN IN minimal-splines (94/89). There the source of coverage
        // is the whole test set, here it is the fast set only: the classes covered
        // ONLY by the slow tests (the cross-check against the publication, the characterization, the full matrix
        // of orders) do not appear in the report, although the tests themselves are executed in `check`.
        // Raising the bar to the level of minimal-splines is possible only together with enabling
        // the instrumentation for `slowTest`, and that fails under Kover on memory (exit 137,
        // see above). The bar is not an aim but a LATCH: it catches a removal of tests and a collapse of the
        // coverage, so it is more honest to keep it at the measured level than at the desired one.
        verify {
            rule("Line coverage") {
                minBound(67)
            }
            rule("Branch coverage") {
                bound {
                    minValue = 72
                    coverageUnits = CoverageUnit.BRANCH
                }
            }
        }
    }
}

// Demonstration runs: every solver prints its own convergence tables.
tasks.register<JavaExec>("runFredholm") {
    group = "application"
    description = "Demonstration: convergence tables for the Fredholm equation"
    mainClass.set("demo.fredholm.FredholmDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

tasks.register<JavaExec>("runVolterra") {
    group = "application"
    description = "Demonstration: convergence tables for the Volterra equation"
    mainClass.set("demo.volterra.VolterraDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

tasks.register<JavaExec>("runUryson") {
    group = "application"
    description = "Demonstration: convergence tables for the Uryson equation"
    mainClass.set("demo.uryson.UrysonDemoKt")
    classpath = sourceSets["demo"].runtimeClasspath
}

// The benchmark lives in its OWN source set `src/benchmark` (as in minimal-splines): it is not a
// demonstration (there are no convergence tables) and not a test (there is no PASS/FAIL criterion) but a measurement
// that must neither enter the coverage nor break `compileDemoKotlin`.
tasks.register<JavaExec>("runBenchmark") {
    group = "application"
    description = "Performance benchmark (time as a function of N, scalability over threads)"
    mainClass.set("demo.bench.BenchmarkKt")
    classpath = sourceSets["benchmark"].runtimeClasspath
}
