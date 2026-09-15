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
import java.util.Locale
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
 * РЕЖИМ СРАВНЕНИЯ БЕРЁТСЯ ИЗ ДАННЫХ — третья колонка эталона (`класс`), а не из
 * констант этого класса. Классы и их измеренное обоснование — [BaselineClass];
 * разбор и сравнение — [BaselineFormat]; вычисление колонки — `./gradlew classifyBaseline`.
 * Коротко: `portable` — отн. 1e-9 при поле 6e-13 (ловит любое изменение алгоритма,
 * прощает последние биты при ином порядке суммирования), `sensitive` — граница
 * `2*cond*max(omega,eps)*||u||inf`, ВЫЧИСЛЯЕМАЯ в прогоне ([F1SystemConditioning]).
 *
 * ВАЖНО: если изменение алгоритма ОБОСНОВАНО (исправление ошибки), эталон следует
 * пересматривать осознанно, зафиксировав старое и новое значения в отчёте, а не
 * «подгонять» допуск.
 *
 * ТЕГА `machine` БОЛЬШЕ НЕТ — и это результат ИЗМЕРЕНИЯ, а не смягчения требований.
 * Снятие обеих матриц на `-Dnumerics.backend=java` (netlib F2J) и на `native`
 * (netlib + Apple Accelerate) показало: из 1366 ключей текущий гейт валили ровно 52,
 * ВСЕ `F1.*`; у остальных 1312 расхождение путей LU не превышает 1.0e-14 при поле
 * 6e-13 — запас 60x. Ключи F1 выделены в класс `sensitive` и сверяются с границей,
 * вычисляемой на том же бэкенде, поэтому гейт зелёный на ОБОИХ путях LU и гоняется
 * в CI (ubuntu/OpenBLAS — третий независимый путь). Подробности — `docs/TESTING.md`
 * и `docs/baseline-changes.md`.
 */
@Tag("slow")
class EhCharacterizationTest {

    private companion object {
        /** Путь ресурса эталона; общий с инструментом снятия и сторожевым тестом. */
        const val RESOURCE_PATH = "/characterization/baseline-eh.tsv"
    }

    /** Пара «ключ эталона -> зафиксированное значение и класс сравнения». */
    private val baseline: Map<String, BaselineEntry> by lazy {
        val resource = javaClass.getResourceAsStream(RESOURCE_PATH)
            ?: fail("Не найден файл эталона $RESOURCE_PATH")
        resource.bufferedReader().useLines { BaselineFormat.parse(it, RESOURCE_PATH) }
    }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> splines.functionals.ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /** Сверяет вычисленное значение с эталоном по правилу КЛАССА этого ключа. */
    private fun check(key: String, actual: Double, mismatches: MutableList<String>) {
        val expected = baseline[key] ?: run {
            mismatches += "$key: отсутствует в эталоне (вычислено $actual)"
            return
        }
        // Значение приводится к тому же представлению, в котором оно хранится (17 цифр,
        // round-trip для double): иначе сравнение зависело бы от способа печати.
        val formatted = String.format(Locale.ROOT, "%.17g", actual)
        BaselineFormat.compare(key, expected, formatted)?.let { mismatches += it }
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
     * Здесь же правило класса `sensitive` — граница `2*cond*max(omega,eps)*||u||inf`
     * порядка 3.8e-5 при значениях `E_h ~ 8e-5`, и та же мутация ловится
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
