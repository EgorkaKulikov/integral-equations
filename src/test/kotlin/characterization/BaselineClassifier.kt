package characterization

import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

/**
 * ИНСТРУМЕНТ КЛАССИФИКАЦИИ ЭТАЛОНА: колонка `класс` ВЫЧИСЛЯЕТСЯ, а не пишется рукой.
 *
 * Запуск: `./gradlew classifyBaseline`. Задача снимает обе матрицы ДВАЖДЫ — на
 * `-Dnumerics.backend=java` (netlib F2J, чистая Java) и на `native` (netlib + системная
 * LAPACK, на машине разработчика — Apple Accelerate), — после чего этот инструмент
 * сравнивает снимки и пишет `build/baseline/classified/baseline-{eh,extra}.tsv`
 * со значениями НАТИВНОГО прогона и третьей колонкой класса.
 *
 * ПРАВИЛО КЛАССИФИКАЦИИ (в порядке применения):
 *  1. ключ `*.iters` — [BaselineClass.EXACT]. Счётчик целочисленный, и измерение
 *     разрешает строгость: между бэкендами он не разошёлся ни разу (0 из 336).
 *     Если разойдётся — инструмент ПАДАЕТ: это регрессия, а не новый класс;
 *  2. ключ `*.residual` — [BaselineClass.RESIDUAL];
 *  3. значения java и native согласованы по правилу `portable` — [BaselineClass.PORTABLE];
 *  4. иначе — [BaselineClass.SENSITIVE], НО только если ключ поддержан механикой
 *     границы ([F1SystemConditioning.supports], то есть схемы `base`/`sloan` задачи F1).
 *     Ключ любой другой схемы, вышедший за правило `portable`, — ДЕФЕКТ: инструмент
 *     падает с перечислением таких ключей. Для `kulkarni`/`nystrom`/Урысона границы
 *     не существует без выставления наружу матрицы соответствующей схемы, и молчаливое
 *     расширение класса означало бы отключение гейта на этих ключах.
 *
 * ПРОДВИЖЕНИЕ ГРУППЫ. Класс назначается ТРОЙКЕ (система, семейство, n) целиком: если
 * разошёлся хотя бы один ключ `base`/`sloan` группы, `sensitive` получают оба. Иначе
 * состав `sensitive` зависел бы от того, какой из двух ключей сегодня чуть ближе к
 * границе, — то есть от шума, а не от свойства системы.
 */
object BaselineClassifier {

    private const val EH_NAME = "baseline-eh.tsv"
    private const val EXTRA_NAME = "baseline-extra.tsv"

    private val EH_HEADER = """
        # ХАРАКТЕРИЗАЦИОННЫЙ ЭТАЛОН E_h (снимок собственного поведения реализации).
        #
        # Сверяется тестом characterization.EhCharacterizationTest.
        # ФОРМАТ СТРОКИ:  ключ <TAB> значение <TAB> класс.
        #   значение — 17 значащих цифр (%.17g): это МИНИМУМ, при котором десятичная запись
        #              double восстанавливается побитово; при 12 цифрах сама запись эталона
        #              вносила погрешность хранения ~1e-12 — грубее измеренного расхождения
        #              реализаций BLAS (<= 1.0e-14 на не-F1 ключах);
        #   класс    — режим сравнения, ВЫЧИСЛЕННЫЙ инструментом `./gradlew classifyBaseline`
        #              по двум снимкам (-Dnumerics.backend=java против native). Рукой он не
        #              проставляется: рукописный список повторил бы «что сегодня падает».
        #     portable  — отн. 1e-9 при поле 6e-13 = 10^3*eps*||u||inf. Расхождение путей LU
        #                 на этих ключах <= 1.0e-14, то есть запас 60x;
        #     sensitive — |dlt| <= 2*cond_1*max(omega,eps)*||u||inf, где cond и omega ВЫЧИСЛЯЮТСЯ
        #                 в том же прогоне и на том же бэкенде (characterization.F1SystemConditioning).
        #                 Константу здесь хранить нельзя: она вернула бы привязку к машине;
        #     exact     — строгое совпадение строк (целочисленные счётчики итераций);
        #     residual  — отн. 1e-3 при шуме 6e-15.
        # Строки, начинающиеся с #, и пустые — комментарии. ЛЮБАЯ иная строка обязана
        # разбиваться ровно на три колонки, иначе разбор ПАДАЕТ (прежний парсер молча
        # считал такую строку комментарием, и ключ становился «отсутствующим в эталоне»).
        #
        # ОКРУЖЕНИЕ СНЯТИЯ (существенно: числа к нему привязаны).
        #   бэкенд линейной алгебры : netlib + системная LAPACK (Apple Accelerate), -Dnumerics.backend=native
        #   библиотеки              : numerical-core 1.1.0, minimal-splines 1.1.0
        #   JDK                     : 21 (jvmToolchain(21))
        #   платформа               : macOS aarch64 (Apple silicon)
        # ПЕРЕНОСИМОСТЬ ПРОВЕРЕНА: гейт зелёный и на `native`, и на `java` (другой путь LU).
        # Ровно поэтому тег `machine` с гейта снят, и он гоняется в CI на ubuntu/OpenBLAS.
        #
        # КЛЮЧИ F1.*.sloan ПРИВЯЗАНЫ К ПОРЯДКУ СУММИРОВАНИЯ, а не только к бэкенду.
        # F1 — уравнение первого рода, решаемое регуляризацией Вазваза с alpha = 1e-10, то есть
        # c_L = -1/alpha = -1e10. Решение Слоана есть fEff(t) + c_L*applyNodes(t, .), где ОБА
        # слагаемых имеют порядок 1.38e10, а их сумма — порядок 2.7: сокращение в 5e9 раз,
        # теряется около 9.7 из 16 значащих цифр. Три МАТЕМАТИЧЕСКИ ЭКВИВАЛЕНТНЫХ порядка
        # суммирования дают E_h, различающиеся на 4.3–39.9 % (медиана 28 %). Именно эти ключи
        # и получили класс sensitive — их сверка идёт с вычисляемой границей, а не с допуском.
        #
        # ИСТОРИЯ ИЗМЕНЕНИЙ ЭТАЛОНА: docs/baseline-changes.md.
    """.trimIndent()

    private val EXTRA_HEADER = """
        # ДОПОЛНИТЕЛЬНЫЙ ХАРАКТЕРИЗАЦИОННЫЙ ЭТАЛОН: комбинированный Nyström,
        # неравномерные сетки, отрезок [0,2] — то, что НЕ покрыто baseline-eh.tsv.
        #
        # Сверяется тестом characterization.ExtraCharacterizationTest.
        # ФОРМАТ СТРОКИ:  ключ <TAB> значение <TAB> класс — тот же, что и у baseline-eh.tsv.
        #   значение — 17 значащих цифр (%.17g, round-trip для double) либо маркер
        #              NaN / Infinity / -Infinity / ERROR:<класс>, сравниваемый ПОДСТРОЧНО:
        #              превращение отказа в число — такое же изменение поведения, как и
        #              изменение самого числа;
        #   класс    — ВЫЧИСЛЕН инструментом `./gradlew classifyBaseline` по двум снимкам
        #              (-Dnumerics.backend=java против native), см. шапку baseline-eh.tsv.
        #              Здесь встречаются portable (значения E_h), residual (ключи *.residual,
        #              отн. 1e-3 при шуме 6e-15) и exact (счётчики *.iters, строгое равенство).
        #              Класса sensitive в этом файле НЕТ: измерение дало 0 падений на обоих
        #              бэкендах, то есть привязки к пути LU у дополнительной матрицы нет.
        # Прежний выбор режима по СУФФИКСУ ключа в коде теста этим и заменён: режим берётся
        # из данных, и добавление схемы не требует правки цепочки `if`.
        #
        # ОКРУЖЕНИЕ СНЯТИЯ (существенно: числа к нему привязаны).
        #   бэкенд линейной алгебры : netlib + системная LAPACK (Apple Accelerate), -Dnumerics.backend=native
        #   библиотеки              : numerical-core 1.1.0, minimal-splines 1.1.0
        #   JDK                     : 21 (jvmToolchain(21))
        #   платформа               : macOS aarch64 (Apple silicon)
        # ПЕРЕНОСИМОСТЬ ПРОВЕРЕНА: гейт зелёный и на `native`, и на `java`; тег `machine` снят,
        # гейт гоняется в CI на ubuntu/OpenBLAS.
        #
        # ИСТОРИЯ ИЗМЕНЕНИЙ ЭТАЛОНА: docs/baseline-changes.md.
    """.trimIndent()

    @JvmStatic
    fun main(args: Array<String>) {
        val javaDir = File(args.getOrElse(0) { "build/baseline/java" })
        val nativeDir = File(args.getOrElse(1) { "build/baseline/native" })
        val outDir = File(args.getOrElse(2) { "build/baseline/classified" }).apply { mkdirs() }
        val problems = mutableListOf<String>()
        var total = 0
        for ((name, header) in listOf(EH_NAME to EH_HEADER, EXTRA_NAME to EXTRA_HEADER)) {
            total += classifyFile(File(javaDir, name), File(nativeDir, name), File(outDir, name), header, problems)
        }
        if (problems.isNotEmpty()) {
            System.err.println("КЛАССИФИКАЦИЯ НЕ ВЫПОЛНЕНА (${problems.size} шт.):")
            problems.take(60).forEach { System.err.println("  $it") }
            exitProcess(1)
        }
        println("Классифицировано $total ключей -> ${outDir.absolutePath}")
    }

    private fun readSnapshot(file: File): Map<String, String> {
        check(file.isFile) { "Снимок не найден: ${file.absolutePath} (нужен прогон captureBaseline* обоих бэкендов)" }
        return file.readLines().mapNotNull { line ->
            val parts = line.trim().split('\t')
            if (parts.size >= 2 && !line.startsWith("#")) parts[0] to parts[1] else null
        }.toMap()
    }

    private fun classifyFile(
        javaFile: File,
        nativeFile: File,
        target: File,
        header: String,
        problems: MutableList<String>,
    ): Int {
        val javaRows = readSnapshot(javaFile)
        val nativeRows = readSnapshot(nativeFile)
        val onlyJava = javaRows.keys - nativeRows.keys
        val onlyNative = nativeRows.keys - javaRows.keys
        if (onlyJava.isNotEmpty() || onlyNative.isNotEmpty()) {
            problems += "${target.name}: составы снимков различаются (только java: ${onlyJava.size}, " +
                "только native: ${onlyNative.size}) — сравнивать нечего"
            return 0
        }
        val classes = LinkedHashMap<String, BaselineClass>()
        for (key in nativeRows.keys.sorted()) {
            val j = javaRows.getValue(key)
            val n = nativeRows.getValue(key)
            val cls = when {
                key.endsWith(BaselineFormat.ITERATIONS_SUFFIX) -> {
                    if (j != n) {
                        problems += "$key: счётчик итераций разошёлся между бэкендами (java=$j, native=$n) — " +
                            "это регрессия критерия останова, а не новый класс"
                    }
                    BaselineClass.EXACT
                }
                key.endsWith(ExtraCharacterizationMatrix.RESIDUAL_SUFFIX) -> BaselineClass.RESIDUAL
                portableAgrees(j, n) -> BaselineClass.PORTABLE
                F1SystemConditioning.supports(key) -> BaselineClass.SENSITIVE
                else -> {
                    problems += "$key: значение зависит от пути LU (java=$j, native=$n), но схема НЕ base/sloan " +
                        "задачи F1 — границы cond*omega для неё не существует. Это ДЕФЕКТ, а не класс: " +
                        "найдите причину расхождения, а не помечайте ключ sensitive"
                    BaselineClass.PORTABLE
                }
            }
            classes[key] = cls
        }
        promoteGroups(classes)
        target.writeText(
            buildString {
                append(header).append('\n')
                for ((key, cls) in classes) {
                    append(key).append('\t').append(nativeRows.getValue(key)).append('\t').append(cls.tag).append('\n')
                }
            },
        )
        val counts = classes.values.groupingBy { it.tag }.eachCount().toSortedMap()
        println("${target.name}: ${classes.size} строк, классы $counts")
        return classes.size
    }

    /** Согласованы ли значения по правилу класса [BaselineClass.PORTABLE]. */
    private fun portableAgrees(expected: String, actual: String): Boolean {
        val e = expected.toDoubleOrNull()
        val a = actual.toDoubleOrNull()
        if (e == null || a == null) return expected == actual
        if (e.isNaN() || a.isNaN()) return e.isNaN() == a.isNaN()
        val difference = abs(a - e)
        if (difference <= BaselineFormat.PORTABLE_ABSOLUTE_FLOOR) return true
        return difference / maxOf(abs(e), BaselineFormat.PORTABLE_ABSOLUTE_FLOOR) <=
            BaselineFormat.PORTABLE_RELATIVE_TOLERANCE
    }

    /** Класс назначается тройке (система, семейство, n) целиком — см. KDoc объекта. */
    private fun promoteGroups(classes: MutableMap<String, BaselineClass>) {
        val sensitiveGroups = classes.filterValues { it == BaselineClass.SENSITIVE }
            .keys.mapNotNull { key -> F1SystemConditioning.parseKey(key)?.let { it } }
            .toSet()
        if (sensitiveGroups.isEmpty()) return
        for (key in classes.keys.toList()) {
            val group = F1SystemConditioning.parseKey(key) ?: continue
            if (group in sensitiveGroups) classes[key] = BaselineClass.SENSITIVE
        }
    }
}
