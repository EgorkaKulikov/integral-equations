package characterization

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
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
 */
class EhCharacterizationTest {

    private companion object {
        /** Относительный допуск сравнения с эталоном. */
        const val RELATIVE_TOLERANCE = 1e-9

        /**
         * Абсолютный «пол» сравнения: значения ниже него считаются нулевыми.
         * Нужен для задач вида F2span/V2span, где решение лежит в span порождающей
         * системы и ошибка близка к машинному нулю — там относительное сравнение
         * бессмысленно (шум последних битов даёт разброс в разы).
         */
        const val ABSOLUTE_FLOOR = 1e-11
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
        "theta" -> numerics.functionals.ProjFunctionals(basis)
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
        if (abs(expected) < ABSOLUTE_FLOOR && abs(actual) < ABSOLUTE_FLOOR) return
        val relative = abs(actual - expected) / maxOf(abs(expected), ABSOLUTE_FLOOR)
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
                        val solver = solvers.fredholm.SecondKindSolver(
                            basis, funcs, op, 1.0,
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
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
                        val solver = solvers.volterra.SecondKindSolver(
                            basis, funcs, op, 1.0,
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
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
                    val funcs = numerics.functionals.ProjFunctionals(basis)
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
            val funcs = numerics.functionals.ProjFunctionals(basis)

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
}
