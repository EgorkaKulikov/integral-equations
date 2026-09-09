package characterization

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraSecondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * ХАРАКТЕРИЗАЦИОННЫЙ ТЕСТ (сеть безопасности от регрессий).
 *
 * Не проверяет математическую правильность — он фиксирует ТЕКУЩЕЕ поведение всех
 * сочетаний «задача x порождающая система x семейство функционалов x схема», чтобы
 * последующий рефакторинг (перенос файлов, переименования, вынос общего кода) не
 * изменил численные результаты незаметно.
 *
 * Эталон хранится в `src/test/resources/characterization/baseline-eh.tsv` и снят
 * инструментом [BaselineSnapshotTool] на исходном состоянии репозитория.
 *
 * Допуск [RELATIVE_TOLERANCE] = 1e-9 (относительный) выбран так, чтобы:
 *  - пропускать несущественные различия последних битов, возможные из-за иного
 *    порядка суммирования при параллельной сборке матриц;
 *  - ловить любое реальное изменение алгоритма (оно меняет результат на много
 *    порядков больше).
 *
 * ВАЖНО: если изменение алгоритма ОБОСНОВАНО (исправление ошибки), эталон следует
 * пересnimать осознанно, зафиксировав старое и новое значения в отчёте, а не
 * «подгонять» допуск.
 *
 * ТЕГ `machine` — ЭТАЛОН ПРИВЯЗАН К МАШИНЕ, НА КОТОРОЙ СНЯТ.
 * Шапка `baseline-eh.tsv` фиксирует окружение снятия: multik/OpenBLAS, JDK 21,
 * macOS aarch64. На другой архитектуре CPU нативный BLAS выбирает другие ядра
 * (NEON против AVX), а с ними — другой порядок блочного суммирования. Разница
 * возникает в последних битах, но допуск здесь 1e-9, и на плохо обусловленных
 * задачах F1 (`cond(I - M) ~ 1e10`) она усиливается на много порядков: у схемы
 * `sloan` два слагаемых порядка 1.38e10 сокращаются до O(1), и три математически
 * эквивалентных порядка суммирования дают `E_h` с разбросом 4.3-39.9 %
 * (измерено, см. `docs/baseline-changes.md`).
 *
 * Поэтому тест — ЛОКАЛЬНЫЙ гейт (он же в `./gradlew check`), а в CI на чужой
 * архитектуре он исключается флагом `-PmachineDependentGates=false`. Сверка
 * эталона на другом железе не делает гейт строже — она делает его вечно красным,
 * а красный гейт перестают читать. Подробности — `docs/TESTING.md`.
 */
@Tag("slow")
@Tag("machine")
class EhCharacterizationTest {

    private companion object {
        /** Относительный допуск сравнения с эталоном. */
        const val RELATIVE_TOLERANCE = 1e-9

        /**
         * Абсолютный «пол» сравнения: расхождение |actual − expected| не выше него —
         * шум округления, относительным допуском не проверяется.
         *
         * Значение — `10³·ε·‖u‖∞` при `‖u‖∞ ≈ e` (решения задач эталона имеют порядок
         * `e^t` на `[0,1]`): `E_h = max|u − u_h|` есть разность величин порядка `e`, и
         * накопленная ошибка округления при её вычислении (сборка `u_h` из `O(n)`
         * слагаемых, LU-разложение, квадратуры) составляет десятки–сотни `ε·‖u‖`.
         * Отклонения ниже этого уровня не отличимы от шума и не являются изменением
         * метода. Измерено при смене реализации LAPACK (numerical-core 1.0.0:
         * multik/OpenBLAS → netlib с системной библиотекой): 442 из 1312 не-F1 ключей
         * сдвинулись, максимум `4.0e-15` абс. (`≈ 7·ε·‖u‖`) — на два порядка ниже пола;
         * при прежнем поле `1e-11` (применялся лишь к паре «оба значения ниже пола»)
         * 93 из них падали с отн. расхождением до `3.2e-5` при `E_h ≈ 1e-11`: допуск
         * `1e-9` требовал там абсолютного согласия `1e-20`, то есть побитовой
         * воспроизводимости LU.
         *
         * Чего пол НЕ делает: ключи F1 (`alpha = 1e-10`, `cond₁ ≈ 2·10¹⁰`) при той же
         * смене LAPACK сдвинулись на `4e-8 … 4e-4` абс. — на 4–9 порядков выше пола, и
         * гейт их ловит; их пересъём оформлен отдельно (`docs/baseline-changes.md`,
         * запись от 2026-09-09).
         */
        const val ABSOLUTE_FLOOR = 6e-13
    }

    /** Пара «ключ эталона -> зафиксированное значение E_h». */
    private val baseline: Map<String, Double> by lazy {
        val resource = javaClass.getResourceAsStream("/characterization/baseline-eh.tsv")
            ?: fail("Не найден файл эталона /characterization/baseline-eh.tsv")
        resource.bufferedReader().useLines { lines ->
            lines.mapNotNull { line ->
                val parts = line.trim().split('\t')
                if (parts.size == 2) parts[0] to parts[1].toDouble() else null
            }.toMap()
        }
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> splines.functionals.ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** Сверяет вычисленное значение с эталоном по ключу. */
    private fun check(key: String, actual: Double, mismatches: MutableList<String>) {
        val expected = baseline[key] ?: run {
            mismatches += "$key: отсутствует в эталоне (вычислено $actual)"
            return
        }
        val difference = abs(actual - expected)
        // Расхождение на уровне шума округления (см. KDoc [ABSOLUTE_FLOOR]) — не изменение метода.
        if (difference <= ABSOLUTE_FLOOR) return
        val relative = difference / maxOf(abs(expected), ABSOLUTE_FLOOR)
        if (relative > RELATIVE_TOLERANCE) {
            mismatches += "$key: эталон=$expected, получено=$actual, отн.расхождение=$relative"
        }
    }

    private fun reportIfAny(mismatches: List<String>) {
        assertTrue(
            mismatches.isEmpty(),
            "Обнаружено изменение численного поведения (${mismatches.size} шт.):\n" +
                mismatches.joinToString("\n").take(4000),
        )
    }

    /** Все схемы Фредгольма на всех базисах и семействах функционалов. */
    @Test
    fun fredholmMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val problems = listOf(
            problems.fredholm.FredholmProblem.F2span,
            problems.fredholm.FredholmProblem.F2,
            problems.fredholm.FredholmProblem.F2exp,
        )
        for (problem in problems) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    for (n in listOf(8, 16)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                        val solver = FredholmSecondKindSolver(
                            basis, funcs, op, 1.0,
                            RhsWithDerivatives(
                                { t -> problem.rhsExact(t, op) },
                                { t -> problem.rhsExactDeriv(t, op) },
                                { t -> problem.rhsExactDeriv2(t, op) },
                            ),
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "F.${problem.name}.${system.name}.$familyName.n$n"
                        check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                        check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                        check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                        check(
                            "$prefix.iterKulkarni",
                            errorEh(exact, solver.iteratedKulkarni().eval, grid),
                            mismatches,
                        )
                        if (!funcs.usesDerivative) {
                            check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                            check(
                                "$prefix.iterNystrom",
                                errorEh(exact, solver.iteratedNystrom().eval, grid),
                                mismatches,
                            )
                        }
                    }
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** Все схемы Вольтерры на всех базисах и семействах функционалов. */
    @Test
    fun volterraMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val problems = listOf(
            problems.volterra.VolterraProblem.V2span,
            problems.volterra.VolterraProblem.V2,
            problems.volterra.VolterraProblem.V2exp,
            problems.volterra.VolterraProblem.V2win,
        )
        for (problem in problems) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (familyName in listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")) {
                    for (n in listOf(8, 16)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = family(familyName, basis)
                        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
                        val solver = VolterraSecondKindSolver(
                            basis, funcs, op, 1.0,
                            RhsWithDerivatives(
                                { t -> problem.rhsExact(t, op) },
                                { t -> problem.rhsExactDeriv(t, op) },
                                { t -> problem.rhsExactDeriv2(t, op) },
                            ),
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "V.${problem.name}.${system.name}.$familyName.n$n"
                        check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                        check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                        check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                        check(
                            "$prefix.iterKulkarni",
                            errorEh(exact, solver.iteratedKulkarni().eval, grid),
                            mismatches,
                        )
                        if (!funcs.usesDerivative) {
                            check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                            check(
                                "$prefix.iterNystrom",
                                errorEh(exact, solver.iteratedNystrom().eval, grid),
                                mismatches,
                            )
                        }
                    }
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** Нелинейные схемы Урысона (II род) на всех базисах. */
    @Test
    fun urysonMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        for (problem in listOf(problems.uryson.UrysonProblem.A, problems.uryson.UrysonProblem.B)) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (n in listOf(8, 16)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system, grid)
                    val funcs = splines.functionals.ProjFunctionals(basis)
                    val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                    val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = problems.uryson.secondKindSolver(problem, basis, funcs, space, op)
                    val exact = { t: Double -> problem.exact(t) }
                    val prefix = "U.${problem.name}.${system.name}.n$n"
                    check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
                    check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
                    check("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid), mismatches)
                    check("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid), mismatches)
                }
            }
        }
        reportIfAny(mismatches)
    }

    /** Решатели уравнений I рода (Фредгольм — Wazwaz, Вольтерра — дифференцирование). */
    @Test
    fun firstKindMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = splines.functionals.ProjFunctionals(basis)

            val fp = problems.fredholm.FredholmProblem.F1
            val fop = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val fSolver = problems.fredholm.firstKindSolver(fp, basis, funcs, fop)
            check("F1.B.theta.n$n.base", errorEh({ t -> fp.exact(t) }, fSolver.base().eval, grid), mismatches)
            check("F1.B.theta.n$n.sloan", errorEh({ t -> fp.exact(t) }, fSolver.sloan().eval, grid), mismatches)

            val vp = problems.volterra.VolterraProblem.V1
            val vop = solvers.volterra.VolterraOperator(vp.kernel, grid, GaussLegendre(8))
            val vSolver = problems.volterra.firstKindSolver(vp, basis, funcs, vop)
            check("V1.B.theta.n$n.base", errorEh({ t -> vp.exact(t) }, vSolver.base().eval, grid), mismatches)
            check("V1.B.theta.n$n.sloan", errorEh({ t -> vp.exact(t) }, vSolver.sloan().eval, grid), mismatches)
        }
        reportIfAny(mismatches)
    }

    /**
     * РАСШИРЕННОЕ ПОКРЫТИЕ F1 — КОМПЕНСАЦИЯ широкого допуска сверки (этап 8.6).
     *
     * Зачем он есть. `verification.PublishedValuesTest` сверяет 42 величины F1 с
     * публикацией с допуском 2 %, а шесть ключей из `table-f1.tex` — с 12 %
     * (таблица снята на другом пути LU, см. KDoc там). ЗАМЕРЕНО (этап 8.6):
     * при таких допусках огрубление квадратуры `GaussLegendre(8) → 6` и `→ 4`
     * проходит сверку НЕЗАМЕЧЕННЫМ на всех 42 ключах. Причина не в допуске:
     * у F1 `E_h ≈ 8e-5` определяется регуляризацией (`alpha = 1e-10`), а не
     * дискретизацией — порядок сходимости ≈ 0, и сама величина к качеству
     * квадратуры малочувствительна.
     *
     * Здесь же допуск [RELATIVE_TOLERANCE] = 1e-9, и та же мутация ловится
     * немедленно. Состав покрытия — НАДМНОЖЕСТВО того, что сверяется с
     * публикацией, поэтому ни одна ослабленная там величина не остаётся без
     * строгого гейта. Перечисление сочетаний взято из [BaselineSnapshotTool.F1_COVERAGE]
     * — одно и то же для гейта и для инструмента снятия, так что состав
     * эталона и состав проверки разойтись не могут.
     *
     * О СХЕМЕ `sloan` ДЛЯ F1 — важное свойство, а НЕ дефект. Ключи `F1.*.sloan`
     * ПРИВЯЗАНЫ К ПОРЯДКУ ОБХОДА УЗЛОВ в `FredholmOperator.applyNodes`. Причина:
     * решение Слоана есть `fEff(t) + c_L·applyNodes(t, ·)`, где `c_L = -1/alpha = -1e10`,
     * и ОБА слагаемых имеют порядок 1.38e10, а их сумма — порядок 2.7
     * (измерено, этап 8.6). То есть происходит сокращение в 5·10^9 раз, и теряется
     * около 9.7 из 16 значащих цифр. ЗАМЕР: три МАТЕМАТИЧЕСКИ ЭКВИВАЛЕНТНЫХ
     * порядка суммирования той же формулы (прямой, обратный, компенсированный
     * Кэхэна—Неймана) дают `E_h`, различающиеся на 4.3–39.9 % (медиана 28 %) —
     * больше, чем расхождение с публикацией и больше, чем расхождение бэкендов.
     *
     * Практическое следствие для будущего рефакторинга: ЛЮБАЯ перестановка
     * цикла суммирования в `applyNodes` (обратный обход, блочное или параллельное
     * суммирование, компенсированное сложение) сломает этот гейт НА КЛЮЧАХ
     * `F1.*.sloan`, НЕ ИЗМЕНИВ МАТЕМАТИКИ. Такое падение НЕ свидетельствует о
     * ошибке в новом коде — но и не может быть просто заглушено: требуется
     * осознанный пересъём этих ключей с записью в `docs/baseline-changes.md`.
     */
    @Test
    fun firstKindExtendedFredholmMatchesBaseline() {
        val mismatches = mutableListOf<String>()
        val fp = problems.fredholm.FredholmProblem.F1
        for ((system, familyName, n) in BaselineSnapshotTool.F1_COVERAGE) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val funcs = family(familyName, basis)
            val op = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val solver = problems.fredholm.firstKindSolver(fp, basis, funcs, op)
            val exact = { t: Double -> fp.exact(t) }
            val prefix = "F1.${system.name}.$familyName.n$n"
            check("$prefix.base", errorEh(exact, solver.base().eval, grid), mismatches)
            check("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid), mismatches)
        }
        reportIfAny(mismatches)
    }
}
