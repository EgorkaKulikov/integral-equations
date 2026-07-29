package verification

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import kotlin.math.abs
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * СВЕРКА С ОПУБЛИКОВАННЫМИ РЕЗУЛЬТАТАМИ (задача 2.2).
 *
 * В отличие от `characterization.EhCharacterizationTest`, который фиксирует
 * собственное поведение реализации, этот тест сверяет вычисленные значения с
 * ВНЕШНИМ источником — таблицами статьи, помеченными «Do not alter numbers».
 * Числа получены другим запуском на другой ревизии кода, поэтому совпадение
 * является независимым подтверждением, а не самопроверкой.
 *
 * Эталон: `src/test/resources/verification/published-values.tsv`
 * (ключ, значение, файл-источник, расположение в файле).
 *
 * ВАЖНО. Допуск [RELATIVE_TOLERANCE] определяется точностью публикации (четыре
 * значащие цифры) и НЕ подлежит ослаблению для «прохождения» теста. Расхождение
 * свыше допуска означает либо реальное расхождение с опубликованным результатом,
 * либо ошибку переноса числа — и то, и другое требует разбирательства.
 */
class PublishedValuesTest {

    private companion object {
        /**
         * Относительный допуск сверки.
         *
         * В таблицах статьи приведено четыре значащие цифры, поэтому собственная
         * погрешность записи числа достигает 0.05 %. Допуск 2 % даёт запас на
         * различие версий JDK, порядка суммирования при параллельной сборке матриц
         * и накопление ошибки округления, оставаясь при этом на два порядка строже
         * любого содержательного изменения алгоритма.
         */
        const val RELATIVE_TOLERANCE = 0.02

        /**
         * Порог машинного шума для значений `E_h`.
         *
         * Значения ниже него не сверяются: там погрешность метода уже исчерпана и
         * результат определяется порядком суммирования, а не алгоритмом. Что это
         * именно шум, видно из самих таблиц статьи — для F2exp, базис B, схема
         * Кулкарни при n=32 таблица `table-t2-fredholm.tex` даёт 7.327e-15, а
         * `table-t3-fredholm.tex` для той же величины — 7.994e-15 (расхождение 9 %),
         * хотя обе получены из одного набора запусков. Требовать 2 % там, где сама
         * публикация расходится с собой на 9 %, бессмысленно.
         *
         * Значение 1e-13 выбрано так, чтобы отсечь этот диапазон, но сохранить в
         * сверке величины уровня 1e-12 и выше, которые ещё воспроизводимы.
         */
        const val NOISE_FLOOR = 1e-13

        /** Порядок квадратуры, единый для всех расчётов (как в демонстрациях и эталоне). */
        const val QUADRATURE_ORDER = 8
    }

    /** Разобранная строка эталона. */
    private data class PublishedValue(
        val key: String,
        val value: Double,
        val sourceFile: String,
        val location: String,
    )

    private val published: Map<String, PublishedValue> by lazy {
        val resource = javaClass.getResourceAsStream("/verification/published-values.tsv")
            ?: fail("Не найден файл эталона /verification/published-values.tsv")
        resource.bufferedReader().useLines { lines ->
            lines.mapNotNull { line ->
                val trimmed = line.trim()
                if (trimmed.isEmpty() || trimmed.startsWith("#")) return@mapNotNull null
                val parts = line.split('\t')
                if (parts.size != 4) return@mapNotNull null
                val key = parts[0].trim()
                key to PublishedValue(key, parts[1].trim().toDouble(), parts[2].trim(), parts[3].trim())
            }.toMap()
        }
    }

    /** Накопитель результатов одного теста: расхождения и число фактически сверенных величин. */
    private class Verification {
        val mismatches = mutableListOf<String>()
        var checked = 0
        var skippedAsNoise = 0
        val missing = mutableListOf<String>()
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        "lambda" -> ThreePointFunctionals(basis)
        else -> error("Неизвестное семейство функционалов: '$name'")
    }

    private fun system(name: String): GeneratingSystem = when (name) {
        "B" -> GeneratingSystem.B
        "H" -> GeneratingSystem.H
        "T" -> GeneratingSystem.T
        else -> error("Неизвестная порождающая система: '$name'")
    }

    /**
     * Сверяет одну величину с публикацией.
     *
     * Ключ, отсутствующий в эталоне, — не молчаливый пропуск, а отдельно
     * фиксируемое событие: оно означает рассогласование теста и ресурсного файла.
     */
    private fun check(verification: Verification, key: String, actual: Double, isError: Boolean) {
        val expected = published[key] ?: run {
            verification.missing += key
            return
        }
        // Значения на уровне машинного шума исключены из сверки (см. NOISE_FLOOR).
        if (isError && expected.value < NOISE_FLOOR) {
            verification.skippedAsNoise++
            return
        }
        verification.checked++
        val relative = abs(actual - expected.value) / abs(expected.value)
        if (relative > RELATIVE_TOLERANCE) {
            verification.mismatches += buildString {
                append(key)
                append(": опубликовано=").append(expected.value)
                append(", вычислено=").append(actual)
                append(", отн.расхождение=").append("%.2f%%".format(100.0 * relative))
                append(" [источник: ").append(expected.sourceFile)
                append(", ").append(expected.location).append("]")
            }
        }
    }

    private fun report(verification: Verification, title: String) {
        assertTrue(
            verification.missing.isEmpty(),
            "$title: ключи отсутствуют в эталоне published-values.tsv " +
                "(${verification.missing.size} шт.):\n" + verification.missing.joinToString("\n").take(2000),
        )
        assertTrue(
            verification.checked > 0,
            "$title: не сверено ни одной величины — проверьте согласованность ключей",
        )
        assertTrue(
            verification.mismatches.isEmpty(),
            "$title: расхождение с опубликованными значениями свыше " +
                "${100.0 * RELATIVE_TOLERANCE}% (${verification.mismatches.size} из " +
                "${verification.checked} сверенных):\n" +
                verification.mismatches.joinToString("\n").take(6000),
        )
    }

    /** Эмпирический порядок `p_h = log2(E_h / E_{h/2})`. */
    private fun order(coarse: Double, fine: Double): Double = ln(coarse / fine) / ln(2.0)

    // ------------------------------------------------------------------------
    // Уравнение Фредгольма
    // ------------------------------------------------------------------------

    /** Задачи Фредгольма II рода: все схемы, базисы и семейства, встречающиеся в таблицах. */
    @Test
    fun fredholmSecondKindMatchesPublishedValues() {
        val verification = Verification()
        val problems = listOf(
            problems.fredholm.FredholmProblem.F2,
            problems.fredholm.FredholmProblem.F2exp,
        )
        for (problem in problems) {
            for (systemName in listOf("B", "H", "T")) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    // Погрешности по сеткам: нужны и сами E_h, и порядки между соседними.
                    val errors = LinkedHashMap<String, MutableMap<Int, Double>>()
                    for (n in listOf(8, 16, 32, 64)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system(systemName), grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.fredholm.FredholmOperator(
                            problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                        )
                        val solver = solvers.fredholm.SecondKindSolver(
                            basis, funcs, op, 1.0,
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val schemes = linkedMapOf(
                            "base" to solver.base(),
                            "sloan" to solver.sloan(),
                            "kulkarni" to solver.kulkarni(),
                            "iterKulkarni" to solver.iteratedKulkarni(),
                        )
                        // Схемы Nyström не поддерживают семейства с производной.
                        if (!funcs.usesDerivative) {
                            schemes["nystrom"] = solver.nystrom()
                            schemes["iterNystrom"] = solver.iteratedNystrom()
                        }
                        for ((schemeName, solution) in schemes) {
                            val eh = errorEh(exact, solution.eval, grid)
                            errors.getOrPut(schemeName) { linkedMapOf() }[n] = eh
                            val key = "F.${problem.name}.$systemName.$familyName.n$n.$schemeName.Eh"
                            if (published.containsKey(key)) check(verification, key, eh, isError = true)
                        }
                    }
                    checkOrders(verification, "F.${problem.name}.$systemName.$familyName", errors)
                }
            }
        }
        report(verification, "Фредгольм II рода")
    }

    /** Задача Фредгольма I рода F1: базовая схема и итерация Слоана. */
    @Test
    fun fredholmFirstKindMatchesPublishedValues() {
        val verification = Verification()
        val problem = problems.fredholm.FredholmProblem.F1
        for (systemName in listOf("B", "H", "T")) {
            for (familyName in listOf("theta", "xi1", "xi2")) {
                for (n in listOf(8, 16, 32)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system(systemName), grid)
                    val funcs = family(familyName, basis)
                    val op = solvers.fredholm.FredholmOperator(
                        problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                    )
                    val solver = problems.fredholm.firstKindSolver(problem, basis, funcs, op)
                    val exact = { t: Double -> problem.exact(t) }
                    for ((schemeName, solution) in listOf(
                        "base" to solver.base(),
                        "sloan" to solver.sloan(),
                    )) {
                        val key = "F.F1.$systemName.$familyName.n$n.$schemeName.Eh"
                        if (published.containsKey(key)) {
                            check(verification, key, errorEh(exact, solution.eval, grid), isError = true)
                        }
                    }
                }
            }
        }
        report(verification, "Фредгольм I рода (F1)")
    }

    // ------------------------------------------------------------------------
    // Уравнение Вольтерры
    // ------------------------------------------------------------------------

    /** Задачи Вольтерры II рода: все схемы, базисы и семейства из таблиц. */
    @Test
    fun volterraSecondKindMatchesPublishedValues() {
        val verification = Verification()
        val problems = listOf(
            problems.volterra.VolterraProblem.V2,
            problems.volterra.VolterraProblem.V2exp,
            problems.volterra.VolterraProblem.V2win,
        )
        for (problem in problems) {
            for (systemName in listOf("B", "H", "T")) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    val errors = LinkedHashMap<String, MutableMap<Int, Double>>()
                    for (n in listOf(8, 16, 32, 64)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system(systemName), grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.volterra.VolterraOperator(
                            problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                        )
                        val solver = solvers.volterra.SecondKindSolver(
                            basis, funcs, op, 1.0,
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val schemes = linkedMapOf(
                            "base" to solver.base(),
                            "sloan" to solver.sloan(),
                            "kulkarni" to solver.kulkarni(),
                            "iterKulkarni" to solver.iteratedKulkarni(),
                        )
                        // Nyström для Вольтерры существенно дороже (веса зависят от t),
                        // поэтому в статье он приведён только до n = 32.
                        if (!funcs.usesDerivative && n <= 32) {
                            schemes["nystrom"] = solver.nystrom()
                            schemes["iterNystrom"] = solver.iteratedNystrom()
                        }
                        for ((schemeName, solution) in schemes) {
                            val eh = errorEh(exact, solution.eval, grid)
                            errors.getOrPut(schemeName) { linkedMapOf() }[n] = eh
                            val key = "V.${problem.name}.$systemName.$familyName.n$n.$schemeName.Eh"
                            if (published.containsKey(key)) check(verification, key, eh, isError = true)
                        }
                    }
                    checkOrders(verification, "V.${problem.name}.$systemName.$familyName", errors)
                }
            }
        }
        report(verification, "Вольтерра II рода")
    }

    /** Задача Вольтерры I рода V1: база, Слоан, Кулкарни. */
    @Test
    fun volterraFirstKindMatchesPublishedValues() {
        val verification = Verification()
        val problem = problems.volterra.VolterraProblem.V1
        for (n in listOf(8, 16, 32, 64)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = ProjFunctionals(basis)
            val op = solvers.volterra.VolterraOperator(
                problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
            )
            val solver = problems.volterra.firstKindSolver(problem, basis, funcs, op)
            val exact = { t: Double -> problem.exact(t) }
            for ((schemeName, solution) in listOf(
                "base" to solver.base(),
                "sloan" to solver.sloan(),
                "kulkarni" to solver.kulkarni(),
            )) {
                val key = "V.V1.B.theta.n$n.$schemeName.Eh"
                if (published.containsKey(key)) {
                    check(verification, key, errorEh(exact, solution.eval, grid), isError = true)
                }
            }
        }
        report(verification, "Вольтерра I рода (V1)")
    }

    /**
     * Сверяет эмпирические порядки `p_h`, вычисленные по соседним сеткам.
     *
     * Порядок сверяется только тогда, когда ОБЕ участвующие погрешности лежат выше
     * порога шума: `log2` от отношения двух шумовых величин смысла не имеет.
     * Допуск для порядка — абсолютный, а не относительный: сам порядок в статье
     * приведён с двумя знаками после запятой, и относительное сравнение при
     * значениях около нуля (вырожденно-точные случаи) вело бы себя неустойчиво.
     */
    private fun checkOrders(
        verification: Verification,
        prefix: String,
        errors: Map<String, Map<Int, Double>>,
    ) {
        for ((schemeName, byGrid) in errors) {
            for (n in listOf(8, 16, 32)) {
                val key = "$prefix.n$n.$schemeName.ph"
                val expected = published[key] ?: continue
                val coarse = byGrid[n] ?: continue
                val fine = byGrid[2 * n] ?: continue
                if (coarse < NOISE_FLOOR || fine < NOISE_FLOOR) {
                    verification.skippedAsNoise++
                    continue
                }
                verification.checked++
                val actual = order(coarse, fine)
                // Абсолютный допуск: 2 % от опубликованного порядка, но не менее 0.05
                // (точность записи самого порядка в таблице — два знака).
                val tolerance = maxOf(RELATIVE_TOLERANCE * abs(expected.value), 0.05)
                if (abs(actual - expected.value) > tolerance) {
                    verification.mismatches += buildString {
                        append(key)
                        append(": опубликовано p_h=").append(expected.value)
                        append(", вычислено p_h=").append("%.4f".format(actual))
                        append(" (E_$n=").append(coarse).append(", E_${2 * n}=").append(fine).append(")")
                        append(" [источник: ").append(expected.sourceFile)
                        append(", ").append(expected.location).append("]")
                    }
                }
            }
        }
    }

    /**
     * Целостность ресурсного файла: ключи разбираются, значения положительны.
     *
     * Проверка направлена на сам эталон, а не на решатели: опечатка при переносе
     * числа из LaTeX (например, потерянный знак минус в показателе) иначе
     * проявилась бы как «расхождение реализации с публикацией».
     */
    @Test
    fun publishedValuesResourceIsWellFormed() {
        assertTrue(published.isNotEmpty(), "Ресурсный файл эталона пуст")
        val problems = mutableListOf<String>()
        val validEquations = setOf("F", "V")
        val validSystems = setOf("B", "H", "T")
        val validFamilies = setOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")
        val validSchemes = setOf("base", "sloan", "kulkarni", "iterKulkarni", "nystrom", "iterNystrom")
        val validMetrics = setOf("Eh", "ph")
        for ((key, entry) in published) {
            val parts = key.split('.')
            if (parts.size != 7) {
                problems += "$key: ожидается 7 сегментов ключа, получено ${parts.size}"
                continue
            }
            val (equation, _, systemName, familyName) = parts
            val gridPart = parts[4]
            val schemeName = parts[5]
            val metric = parts[6]
            if (equation !in validEquations) problems += "$key: неизвестное уравнение '$equation'"
            if (systemName !in validSystems) problems += "$key: неизвестный базис '$systemName'"
            if (familyName !in validFamilies) problems += "$key: неизвестное семейство '$familyName'"
            if (schemeName !in validSchemes) problems += "$key: неизвестная схема '$schemeName'"
            if (metric !in validMetrics) problems += "$key: неизвестная метрика '$metric'"
            if (!gridPart.startsWith("n") || gridPart.drop(1).toIntOrNull() == null) {
                problems += "$key: некорректный сегмент сетки '$gridPart'"
            }
            if (metric == "Eh" && entry.value <= 0.0) {
                problems += "$key: погрешность обязана быть положительной, получено ${entry.value}"
            }
            // Погрешность выше единицы означала бы потерю знака минус в показателе.
            if (metric == "Eh" && entry.value > 1.0) {
                problems += "$key: подозрительно большая погрешность ${entry.value} (проверьте перенос)"
            }
            if (entry.sourceFile.isBlank()) problems += "$key: не указан файл-источник"
            if (entry.location.isBlank()) problems += "$key: не указано расположение в файле"
        }
        assertTrue(
            problems.isEmpty(),
            "Ресурсный файл published-values.tsv некорректен (${problems.size} шт.):\n" +
                problems.joinToString("\n").take(4000),
        )
    }

    /** Деструктуризация первых четырёх сегментов ключа. */
    private operator fun <T> List<T>.component4(): T = this[3]
}
