package conventions

import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Tags
import java.io.File
import java.lang.reflect.Method
import java.lang.reflect.Modifier
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * A GUARD TEST OF THE TAG CONVENTION.
 *
 * Motivation. No test task of the build runs "all the tests": `fastTest`
 * selects `includeTags("fast")`, `slowTest` `includeTags("slow")`,
 * `scipyVerify` `includeTags("scipy")`. Therefore a test class WITHOUT a time tag
 * gets into NO suite at all: it compiles but is never executed. The failure
 * is silent — the build is green, the report `build/test-results` simply does not
 * contain such a class, and the loss can be noticed only by manually comparing the list
 * of tags with the list of files. This is exactly what the check here does.
 *
 * What exactly is checked: every test class has exactly ONE time tag from
 * [TIME_TAGS]. Zero — the class drops out of all the suites; more than one — the class
 * is executed twice (in two tasks), and that is either a typo or a hidden
 * doubling of the build time. The other tags (`machine`) are ADDITIONAL: they
 * control the exclusion of machine-dependent gates in CI and do not affect the count
 * of the time tags.
 *
 * TWO ways of placing the tag are allowed:
 *  * a tag on the class — the main case (33 classes out of 34 are marked this way);
 *  * a tag on EVERY test method — this is done in
 *    [convergence.ConvergenceOrderTest], where the methods are deliberately split by
 *    cost: the full matrix of orders goes into `slowTest`, and the two fast
 *    checks into `fastTest`. The invariant is the same (no method stays outside the
 *    suites), so such a class is NOT considered a violation.
 *
 * The detection of the classes goes by the COMPILED output and not by the sources:
 * a text search would not distinguish a tag on a class from a tag inside a string or
 * a comment and would not see the inheritance of annotations. The check requires no external
 * dependencies (`classgraph`, `reflections`) — the class directory
 * is walked directly.
 *
 * The tag of the class itself is `fast`: the check must run in CI after every edit and
 * passes its own check on a par with the rest.
 */
@Tag("fast")
class TagConventionTest {

    private companion object {

        /**
         * The time tags: each corresponds to ONE test task of the build
         * (`fastTest`, `slowTest`, `scipyVerify`). The list must coincide with
         * `includeTags(...)` in `build.gradle.kts`.
         */
        val TIME_TAGS = setOf("fast", "slow", "scipy")

        /**
         * THE DATA GENERATORS are not tests: they have no PASS/FAIL criteria, they
         * shoot baselines and dump artifacts, and use `@Test` only as
         * a way of being run from Gradle. They do not need time tags, because
         * `build.gradle.kts` excludes them BY THESE VERY NAMES from all the suites
         * (`generatorTestClasses` + `excludeGeneratorTools()`).
         *
         * TWO LISTS MUST BE KEPT IN SYNC: when a fifth generator appears, its name
         * is added both there and here. Otherwise either it will silently get into `fastTest`
         * (the build script was forgotten), or the check here will require a tag from it
         * (this list was forgotten).
         */
        val GENERATOR_CLASSES = setOf(
            "characterization.BaselineSnapshotTool",
            "characterization.ExtraBaselineSnapshotTool",
            "verification.VerificationArtifactDumpTool",
            "verification.Sec4VerificationTool",
        )

        /**
         * The lower bound on the number of test classes found.
         *
         * PROTECTION AGAINST DEGENERATION: if the directory walk breaks (the compiler output
         * moves, the naming scheme changes), the check will find zero classes and
         * will pass "successfully", having stopped guarding anything. At the time of writing there
         * are 35 classes; the threshold is taken with a margin below.
         */
        const val MIN_TEST_CLASSES = 30

        /** The directory of the compiled tests, in case it could not be determined from self. */
        const val FALLBACK_CLASSES_DIR = "build/classes/kotlin/test"

        /**
         * The annotations by which a class is recognized as a test one. `kotlin.test.Test` on
         * the JVM is an alias of `org.junit.jupiter.api.Test`, so it needs no separate name.
         * The optional ones (`ParameterizedTest` from `junit-jupiter-params`,
         * `RepeatedTest`) are resolved CAREFULLY: the absence of a type in the classpath must
         * not break the walk.
         */
        val TEST_ANNOTATION_NAMES = listOf(
            "org.junit.jupiter.api.Test",
            "org.junit.jupiter.api.TestFactory",
            "org.junit.jupiter.api.RepeatedTest",
            "org.junit.jupiter.params.ParameterizedTest",
        )
    }

    /**
     * Every test class has exactly one time tag.
     *
     * The error message lists ALL the violations by name with an indication of the
     * cause: a counter ("3 violations") would force one to look for the culprits by hand.
     */
    @Test
    fun everyTestClassHasExactlyOneTimeTag() {
        val testAnnotations = resolveTestAnnotations()
        assertTrue(
            testAnnotations.isNotEmpty(),
            "Not a single test annotation from $TEST_ANNOTATION_NAMES was found — the walk is certainly empty",
        )

        val root = testClassesRoot()
        assertTrue(root.isDirectory, "The directory of the compiled tests is not found: ${root.absolutePath}")

        val testClasses = root.walkTopDown()
            .filter { it.isFile && it.extension == "class" }
            .mapNotNull { binaryName(root, it) }
            // Nested and synthetic classes (a `$` in the name) are skipped: the tag
            // is inherited from the outer class, they need no separate marking.
            .filter { '$' !in it }
            .mapNotNull { loadClassOrNull(it) }
            .filter { !it.isInterface && !Modifier.isAbstract(it.modifiers) }
            .filter { c -> c.declaredMethods.any { m -> isTestMethod(m, testAnnotations) } }
            .filter { it.name !in GENERATOR_CLASSES }
            .sortedBy { it.name }
            .toList()

        assertTrue(
            testClasses.size >= MIN_TEST_CLASSES,
            "The walk found only ${testClasses.size} test classes in ${root.absolutePath}, " +
                "at least $MIN_TEST_CLASSES were expected — the search is broken, not the convention",
        )

        val violations = testClasses.mapNotNull { c -> violationOf(c, testAnnotations)?.let { c.name to it } }

        assertTrue(
            violations.isEmpty(),
            buildString {
                append("The tag convention is violated (classes checked: ${testClasses.size}).\n")
                append("Every test class must carry exactly one time tag from $TIME_TAGS ")
                append("on the class or on every test method, otherwise it will get into no suite.\n")
                violations.forEach { (name, reason) -> append("  * $name: $reason\n") }
            },
        )
    }

    /** The reason for the violation, or `null` if the class is fine. */
    private fun violationOf(c: Class<*>, testAnnotations: List<Class<out Annotation>>): String? {
        val classTags = timeTagsOf(c.declaredTags())
        if (classTags.size == 1) return null
        if (classTags.size > 1) return "several time tags: ${classTags.sorted().joinToString(", ")}"

        // There is no tag on the class — legitimate if EVERY test method is marked.
        val methods = c.declaredMethods.filter { isTestMethod(it, testAnnotations) }.sortedBy { it.name }
        val badMethods = methods.mapNotNull { m ->
            val tags = timeTagsOf(m.declaredTags())
            when {
                tags.size == 1 -> null
                tags.isEmpty() -> "${m.name}: no time tag"
                else -> "${m.name}: several time tags: ${tags.sorted().joinToString(", ")}"
            }
        }
        if (badMethods.isEmpty()) return null
        return "no time tag on the class, and not all the methods are marked individually " +
            "(${badMethods.joinToString("; ")})"
    }

    /**
     * The tags declared DIRECTLY on the element.
     *
     * Two or more `@Tag`s compile into a container `@Tags`, a single one stays
     * a standalone annotation; both representations are taken into account.
     */
    private fun java.lang.reflect.AnnotatedElement.declaredTags(): List<String> {
        val container = getDeclaredAnnotation(Tags::class.java)
        if (container != null) return container.value.map { it.value }
        return getDeclaredAnnotationsByType(Tag::class.java).map { it.value }
    }

    /** Of all the tags keeps only the time ones: `machine` and others are admissible as additional. */
    private fun timeTagsOf(tags: List<String>): List<String> = tags.filter { it in TIME_TAGS }.distinct()

    private fun isTestMethod(m: Method, testAnnotations: List<Class<out Annotation>>): Boolean =
        testAnnotations.any { m.isAnnotationPresent(it) }

    /**
     * The root of the compiled tests.
     *
     * It is determined FROM SELF (by the location of this very class) — this way the check
     * does not depend on the working directory of the Gradle task; the fallback path
     * [FALLBACK_CLASSES_DIR] is left for the case of a run from a jar-like source.
     */
    private fun testClassesRoot(): File {
        val self = TagConventionTest::class.java.protectionDomain?.codeSource?.location
            ?.takeIf { it.protocol == "file" }
            ?.let { runCatching { File(it.toURI()) }.getOrNull() }
            ?.takeIf { it.isDirectory }
        return self ?: File(FALLBACK_CLASSES_DIR)
    }

    /** The path to a `.class` → the binary name of the class; `null` for files outside the package tree. */
    private fun binaryName(root: File, classFile: File): String? {
        val relative = classFile.relativeTo(root).invariantSeparatorsPath
        if (relative.startsWith("META-INF/")) return null
        return relative.removeSuffix(".class").replace('/', '.')
    }

    /**
     * Loads a class WITHOUT initialization (`initialize = false`): the static
     * initializers of the tests read resources and build solvers — the walk does not
     * need this and it would cost minutes. Unresolvable references (a `NoClassDefFoundError` from
     * optional dependencies) are silently skipped: such a class will not
     * run in any suite anyway.
     */
    private fun loadClassOrNull(name: String): Class<*>? =
        try {
            Class.forName(name, false, TagConventionTest::class.java.classLoader)
        } catch (_: Throwable) {
            null
        }

    /** Resolves the names of the test annotations, skipping those absent from the classpath. */
    private fun resolveTestAnnotations(): List<Class<out Annotation>> =
        TEST_ANNOTATION_NAMES.mapNotNull { name ->
            try {
                @Suppress("UNCHECKED_CAST")
                Class.forName(name, false, TagConventionTest::class.java.classLoader) as? Class<out Annotation>
            } catch (_: Throwable) {
                null
            }
        }
}
