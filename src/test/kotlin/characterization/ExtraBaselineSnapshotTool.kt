package characterization

import java.io.File
import kotlin.test.Test

/**
 * Служебный инструмент: снимает ДОПОЛНИТЕЛЬНЫЙ характеризационный снимок
 * (комбинированный Nyström, неравномерные сетки, отрезок, отличный от `[0,1]`).
 * Состав матрицы описан в [ExtraCharacterizationMatrix].
 *
 * Это НЕ проверочный тест: он ничего не утверждает и всегда завершается успешно.
 * Результат — файл `build/baseline/baseline-extra.tsv`, который после осмотра
 * копируется в `src/test/resources/characterization/baseline-extra.tsv` и становится
 * эталоном для [ExtraCharacterizationTest].
 *
 * Запуск: `./gradlew captureExtraBaseline`.
 *
 * УСТРОЙСТВО ЗАПИСИ (с 2026-09-15 такое же у [BaselineSnapshotTool] — именно отсюда оно
 * туда и перенесено; до того это были три различия со старым инструментом):
 *  - имя файла ДЕТЕРМИНИРОВАНО и не содержит имени потока JUnit, иначе оно зависело бы
 *    от планировщика, а сравнение снимков приходилось бы делать шаблоном;
 *  - файл ПЕРЕЗАПИСЫВАЕТСЯ, а не дописывается: при `appendText` повторный запуск
 *    без ручной очистки каталога удваивает содержимое;
 *  - строки ОТСОРТИРОВАНЫ по ключу (сортировкой в [ExtraCharacterizationMatrix.collect]),
 *    поэтому дифф двух снимков показывает изменение чисел, а не перестановку строк.
 */
class ExtraBaselineSnapshotTool {

    /** Снимает всю дополнительную матрицу одним файлом. */
    @Test
    fun captureExtraSnapshot() {
        val rows = ExtraCharacterizationMatrix.collect()
        // См. [BaselineSnapshotTool]: каталог задаётся свойством ради двух снимков подряд.
        val dir = File(System.getProperty("baseline.output.dir")?.takeIf { it.isNotBlank() } ?: "build/baseline").apply { mkdirs() }
        val target = File(dir, "baseline-extra.tsv")
        // Одна операция записи вместо тысячи дозаписей: и быстрее, и исключает
        // частично записанный файл при падении посреди снятия.
        target.writeText(rows.joinToString(separator = "") { (key, value) -> "$key\t$value\n" })
        println("Снимок дополнительной матрицы: ${rows.size} строк -> ${target.absolutePath}")
    }
}
