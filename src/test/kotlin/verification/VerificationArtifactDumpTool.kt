package verification

import kotlin.test.Test

/**
 * ГЕНЕРАТОР ДАННЫХ ДЛЯ РУЧНОГО РАЗБОРА (не проверка).
 *
 * Выгружает внутренние артефакты вычислений в `build/verification/` задачей Gradle
 * `dumpVerificationArtifacts`, не запуская сверку. Нужен, когда расхождение уже
 * обнаружено и требуется исследовать выгруженные числа вручную (или прогнать по ним
 * собственный скрипт), не дожидаясь тестового прогона.
 *
 * Для АВТОМАТИЧЕСКОЙ сверки этот класс не нужен: смоук-тест
 * [ScipyCrossVerificationTest] готовит артефакты сам через [VerificationArtifacts],
 * поэтому не зависит от того, запускалась ли перед ним данная задача.
 *
 * Из обычного `test` исключён (см. `build.gradle.kts`): это не проверка, а генератор.
 */
class VerificationArtifactDumpTool {

    @Test
    fun dumpAllArtifacts() {
        val files = VerificationArtifacts.dumpAll()
        println("Выгружено файлов: ${files.size}")
        for (file in files) println("  ${file.absolutePath} (${file.length()} байт)")
    }
}
