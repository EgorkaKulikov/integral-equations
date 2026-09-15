package conventions

import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Tags
import java.io.File
import java.lang.reflect.Method
import java.lang.reflect.Modifier
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * СТОРОЖЕВОЙ ТЕСТ СОГЛАШЕНИЯ О ТЕГАХ.
 *
 * Мотивация. Ни одна тестовая задача сборки не запускает «все тесты»: `fastTest`
 * отбирает `includeTags("fast")`, `slowTest` — `includeTags("slow")`,
 * `scipyVerify` — `includeTags("scipy")`. Поэтому тестовый класс БЕЗ тега времени
 * не попадает НИ В ОДИН набор: он компилируется, но никогда не выполняется. Отказ
 * при этом молчаливый — сборка зелёная, отчёт `build/test-results` просто не
 * содержит такого класса, и заметить пропажу можно только вручную сверив список
 * тегов со списком файлов. Ровно это и делает здешняя проверка.
 *
 * Что именно проверяется: у каждого тестового класса ровно ОДИН тег времени из
 * [TIME_TAGS]. Ноль — класс выпадает из всех наборов; больше одного — класс
 * выполняется дважды (в двух задачах), а это либо опечатка, либо скрытое
 * удвоение времени сборки. Прочие теги (`machine`) — ДОПОЛНИТЕЛЬНЫЕ: они
 * управляют исключением машинно-зависимых гейтов в CI и на счёт тегов времени
 * не влияют.
 *
 * Допускается ДВА способа расстановки:
 *  * тег на классе — основной случай (так помечены 33 класса из 34);
 *  * тег на КАЖДОМ тестовом методе — так сделано в
 *    [convergence.ConvergenceOrderTest], где методы сознательно разнесены по
 *    стоимости: полная матрица порядков идёт в `slowTest`, а две быстрые
 *    проверки — в `fastTest`. Инвариант тот же (ни один метод не остаётся вне
 *    наборов), поэтому такой класс нарушением НЕ считается.
 *
 * Обнаружение классов идёт по КОМПИЛИРОВАННОМУ выводу, а не по исходникам:
 * поиск по тексту не отличил бы тег на классе от тега внутри строки или
 * комментария и не увидел бы наследование аннотаций. Внешних зависимостей
 * (`classgraph`, `reflections`) проверка не требует — каталог классов
 * обходится напрямую.
 *
 * Тег самого класса — `fast`: проверка обязана идти в CI после каждой правки и
 * проходит собственную проверку наравне с остальными.
 */
@Tag("fast")
class TagConventionTest {

    private companion object {

        /**
         * Теги времени: каждый соответствует ОДНОЙ тестовой задаче сборки
         * (`fastTest`, `slowTest`, `scipyVerify`). Список должен совпадать с
         * `includeTags(...)` в `build.gradle.kts`.
         */
        val TIME_TAGS = setOf("fast", "slow", "scipy")

        /**
         * ГЕНЕРАТОРЫ ДАННЫХ — не тесты: критериев PASS/FAIL у них нет, они
         * снимают эталоны и выгружают артефакты, а `@Test` используют лишь как
         * способ запуска из Gradle. Теги времени им не нужны, потому что
         * `build.gradle.kts` исключает их ПО ЭТИМ ЖЕ ИМЕНАМ из всех наборов
         * (`generatorTestClasses` + `excludeGeneratorTools()`).
         *
         * ДВА СПИСКА НАДО ДЕРЖАТЬ СИНХРОННО: появится пятый генератор — его имя
         * добавляется и туда, и сюда. Иначе либо он молча попадёт в `fastTest`
         * (забыли build-скрипт), либо здешняя проверка потребует от него тега
         * (забыли этот список).
         */
        val GENERATOR_CLASSES = setOf(
            "characterization.BaselineSnapshotTool",
            "characterization.ExtraBaselineSnapshotTool",
            "verification.VerificationArtifactDumpTool",
            "verification.Sec4VerificationTool",
        )

        /**
         * Нижняя граница числа найденных тестовых классов.
         *
         * ЗАЩИТА ОТ ВЫРОЖДЕНИЯ: если обход каталога сломается (переедет вывод
         * компилятора, изменится схема имён), проверка найдёт ноль классов и
         * пройдёт «успешно», перестав что-либо стеречь. На момент написания
         * классов 35; порог взят с запасом вниз.
         */
        const val MIN_TEST_CLASSES = 30

        /** Каталог компилированных тестов на случай, если его не удалось определить по себе. */
        const val FALLBACK_CLASSES_DIR = "build/classes/kotlin/test"

        /**
         * Аннотации, по которым класс признаётся тестовым. `kotlin.test.Test` на
         * JVM — псевдоним `org.junit.jupiter.api.Test`, поэтому отдельного имени
         * не требует. Необязательные (`ParameterizedTest` из `junit-jupiter-params`,
         * `RepeatedTest`) разрешаются ОСТОРОЖНО: отсутствие типа в classpath не
         * должно ронять обход.
         */
        val TEST_ANNOTATION_NAMES = listOf(
            "org.junit.jupiter.api.Test",
            "org.junit.jupiter.api.TestFactory",
            "org.junit.jupiter.api.RepeatedTest",
            "org.junit.jupiter.params.ParameterizedTest",
        )
    }

    /**
     * У каждого тестового класса ровно один тег времени.
     *
     * Сообщение об ошибке перечисляет ВСЕ нарушения поимённо с указанием
     * причины: счётчик («нарушений: 3») заставил бы искать виновных вручную.
     */
    @Test
    fun everyTestClassHasExactlyOneTimeTag() {
        val testAnnotations = resolveTestAnnotations()
        assertTrue(
            testAnnotations.isNotEmpty(),
            "Не найдена ни одна аннотация теста из $TEST_ANNOTATION_NAMES — обход заведомо пуст",
        )

        val root = testClassesRoot()
        assertTrue(root.isDirectory, "Каталог компилированных тестов не найден: ${root.absolutePath}")

        val testClasses = root.walkTopDown()
            .filter { it.isFile && it.extension == "class" }
            .mapNotNull { binaryName(root, it) }
            // Вложенные и синтетические классы (`$` в имени) пропускаем: тег
            // наследуется от внешнего класса, отдельной пометки они не требуют.
            .filter { '$' !in it }
            .mapNotNull { loadClassOrNull(it) }
            .filter { !it.isInterface && !Modifier.isAbstract(it.modifiers) }
            .filter { c -> c.declaredMethods.any { m -> isTestMethod(m, testAnnotations) } }
            .filter { it.name !in GENERATOR_CLASSES }
            .sortedBy { it.name }
            .toList()

        assertTrue(
            testClasses.size >= MIN_TEST_CLASSES,
            "Обход нашёл всего ${testClasses.size} тестовых классов в ${root.absolutePath}, " +
                "ожидалось не меньше $MIN_TEST_CLASSES — сломан поиск, а не соглашение",
        )

        val violations = testClasses.mapNotNull { c -> violationOf(c, testAnnotations)?.let { c.name to it } }

        assertTrue(
            violations.isEmpty(),
            buildString {
                append("Нарушено соглашение о тегах (проверено классов: ${testClasses.size}).\n")
                append("Каждый тестовый класс обязан нести ровно один тег времени из $TIME_TAGS ")
                append("на классе либо на каждом тестовом методе, иначе он не попадёт ни в один набор.\n")
                violations.forEach { (name, reason) -> append("  * $name: $reason\n") }
            },
        )
    }

    /** Причина нарушения или `null`, если с классом всё в порядке. */
    private fun violationOf(c: Class<*>, testAnnotations: List<Class<out Annotation>>): String? {
        val classTags = timeTagsOf(c.declaredTags())
        if (classTags.size == 1) return null
        if (classTags.size > 1) return "несколько тегов времени: ${classTags.sorted().joinToString(", ")}"

        // Тега на классе нет — законно, если помечен КАЖДЫЙ тестовый метод.
        val methods = c.declaredMethods.filter { isTestMethod(it, testAnnotations) }.sortedBy { it.name }
        val badMethods = methods.mapNotNull { m ->
            val tags = timeTagsOf(m.declaredTags())
            when {
                tags.size == 1 -> null
                tags.isEmpty() -> "${m.name}: нет тега времени"
                else -> "${m.name}: несколько тегов времени: ${tags.sorted().joinToString(", ")}"
            }
        }
        if (badMethods.isEmpty()) return null
        return "нет тега времени на классе, и не все методы помечены поимённо " +
            "(${badMethods.joinToString("; ")})"
    }

    /**
     * Теги, объявленные ПРЯМО на элементе.
     *
     * Два и более `@Tag` компилируются в контейнер `@Tags`, один — остаётся
     * самостоятельной аннотацией; учитываются оба представления.
     */
    private fun java.lang.reflect.AnnotatedElement.declaredTags(): List<String> {
        val container = getDeclaredAnnotation(Tags::class.java)
        if (container != null) return container.value.map { it.value }
        return getDeclaredAnnotationsByType(Tag::class.java).map { it.value }
    }

    /** Из всех тегов оставляет только теги времени: `machine` и прочие допустимы как дополнительные. */
    private fun timeTagsOf(tags: List<String>): List<String> = tags.filter { it in TIME_TAGS }.distinct()

    private fun isTestMethod(m: Method, testAnnotations: List<Class<out Annotation>>): Boolean =
        testAnnotations.any { m.isAnnotationPresent(it) }

    /**
     * Корень компилированных тестов.
     *
     * Определяется ПО СЕБЕ (по расположению собственного класса) — так проверка
     * не зависит от рабочего каталога задачи Gradle; резервный путь
     * [FALLBACK_CLASSES_DIR] оставлен на случай запуска из jar-подобного источника.
     */
    private fun testClassesRoot(): File {
        val self = TagConventionTest::class.java.protectionDomain?.codeSource?.location
            ?.takeIf { it.protocol == "file" }
            ?.let { runCatching { File(it.toURI()) }.getOrNull() }
            ?.takeIf { it.isDirectory }
        return self ?: File(FALLBACK_CLASSES_DIR)
    }

    /** Путь к `.class` → двоичное имя класса; `null` для файлов вне дерева пакетов. */
    private fun binaryName(root: File, classFile: File): String? {
        val relative = classFile.relativeTo(root).invariantSeparatorsPath
        if (relative.startsWith("META-INF/")) return null
        return relative.removeSuffix(".class").replace('/', '.')
    }

    /**
     * Загружает класс БЕЗ инициализации (`initialize = false`): статические
     * инициализаторы тестов читают ресурсы и строят решатели — обходу это не
     * нужно и стоило бы минуты. Неразрешимые ссылки (`NoClassDefFoundError` от
     * необязательных зависимостей) молча пропускаются: такой класс всё равно не
     * запустится ни в одном наборе.
     */
    private fun loadClassOrNull(name: String): Class<*>? =
        try {
            Class.forName(name, false, TagConventionTest::class.java.classLoader)
        } catch (_: Throwable) {
            null
        }

    /** Разрешает имена аннотаций тестов, пропуская отсутствующие в classpath. */
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
