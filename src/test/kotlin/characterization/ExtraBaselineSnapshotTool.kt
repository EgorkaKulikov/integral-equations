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
 * ОТЛИЧИЯ ОТ [BaselineSnapshotTool] (сделаны намеренно, это исправление известных дефектов):
 *  - имя файла ДЕТЕРМИНИРОВАНО. Старый инструмент подставляет в имя название потока
 *    JUnit (`snapshot-<поток>.tsv`), поэтому имя зависит от планировщика, а сравнение
 *    снимков приходится делать шаблоном `snapshot-*.tsv`;
 *  - файл ПЕРЕЗАПИСЫВАЕТСЯ, а не дописывается. Старый инструмент использует
 *    `appendText`, и повторный запуск без ручной очистки каталога удваивает содержимое;
 *  - строки ОТСОРТИРОВАНЫ по ключу (сортировкой в [ExtraCharacterizationMatrix.collect]),
 *    поэтому дифф двух снимков показывает изменение чисел, а не перестановку строк.
 */
class ExtraBaselineSnapshotTool {

    /** Снимает всю дополнительную матрицу одним файлом. */
    @Test
    fun captureExtraSnapshot() {
        val rows = ExtraCharacterizationMatrix.collect()
        val dir = File("build/baseline").apply { mkdirs() }
        val target = File(dir, "baseline-extra.tsv")
        // Одна операция записи вместо тысячи дозаписей: и быстрее, и исключает
        // частично записанный файл при падении посреди снятия.
        target.writeText(rows.joinToString(separator = "") { (key, value) -> "$key\t$value\n" })
        println("Снимок дополнительной матрицы: ${rows.size} строк -> ${target.absolutePath}")
    }
}
