package characterization

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.DiscreteDeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import java.io.File
import kotlin.test.Test

/**
 * Служебный инструмент: печатает ЭТАЛОННЫЙ СНИМОК величин E_h по всем сочетаниям
 * «задача x порождающая система x семейство функционалов x схема» для линейных
 * решателей (Фредгольм, Вольтерра) и нелинейного (Урысон).
 *
 * Это НЕ проверочный тест: он ничего не утверждает и всегда завершается успешно.
 * Его единственное назначение — получить машинно-читаемый список значений, которые
 * затем дословно переносятся в характеризационные тесты (`CharacterizationTest`)
 * как сеть безопасности перед рефакторингом.
 *
 * Запуск: `./gradlew captureBaseline` (вывод смотреть в отчёте либо с флагом `-i`).
 * Через `test --tests ...` запустить НЕЛЬЗЯ: этот класс намеренно исключён из
 * всех проверочных задач фильтром, и Gradle ответит `No tests found for given includes`.
 *
 * Формат строки вывода: `SNAP<TAB>ключ<TAB>значение`, где значение печатается с 12
 * значащими цифрами — этого достаточно, чтобы отличить «то же самое вычисление»
 * от «изменившейся арифметики», но с запасом на недетерминизм параллельной сборки.
 */
class BaselineSnapshotTool {

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> numerics.functionals.ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "xitilde1" -> DiscreteDeBoorFixFunctionals(basis, 1)
        "xitilde2" -> DiscreteDeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        else -> ThreePointFunctionals(basis)
    }

    /**
     * Пишет пару «ключ-значение» в файл `build/baseline/<поток>.tsv`.
     *
     * Вывод идёт ТОЛЬКО в файл, а не в stdout: печать более тысячи строк ломает
     * формирование XML-отчё1та Gradle. Каталог `build/` не попадает в систему
     * контроля версий, что для временного артефакта и требуется.
     */
    private fun emit(key: String, value: Double) {
        sink.appendText("$key\t%.12g".format(value) + "\n")
    }

    private val sink: File by lazy {
        val dir = File("build/baseline").apply { mkdirs() }
        File(dir, "snapshot-${Thread.currentThread().name.replace(Regex("[^A-Za-z0-9]"), "_")}.tsv")
    }

    /** Снимок линейного решателя Фредгольма. */
    @Test
    fun snapshotFredholm() {
        val problems = listOf(
            problems.fredholm.FredholmProblem.F2span,
            problems.fredholm.FredholmProblem.F2,
            problems.fredholm.FredholmProblem.F2exp,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val families = listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")
        for (problem in problems) {
            for (system in systems) {
                for (familyName in families) {
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
                        emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                        emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                        emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                        emit("$prefix.iterKulkarni", errorEh(exact, solver.iteratedKulkarni().eval, grid))
                        if (!funcs.usesDerivative) {
                            emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                            emit("$prefix.iterNystrom", errorEh(exact, solver.iteratedNystrom().eval, grid))
                        }
                    }
                }
            }
        }
    }

    /** Снимок линейного решателя Вольтерры. */
    @Test
    fun snapshotVolterra() {
        val problems = listOf(
            problems.volterra.VolterraProblem.V2span,
            problems.volterra.VolterraProblem.V2,
            problems.volterra.VolterraProblem.V2exp,
            problems.volterra.VolterraProblem.V2win,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val families = listOf("theta", "xi0", "xi1", "xi2", "mu", "lambda")
        for (problem in problems) {
            for (system in systems) {
                for (familyName in families) {
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
                        emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                        emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                        emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                        emit("$prefix.iterKulkarni", errorEh(exact, solver.iteratedKulkarni().eval, grid))
                        if (!funcs.usesDerivative) {
                            emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                            emit("$prefix.iterNystrom", errorEh(exact, solver.iteratedNystrom().eval, grid))
                        }
                    }
                }
            }
        }
    }

    /** Снимок нелинейного решателя Урысона (II род). */
    @Test
    fun snapshotUryson() {
        val urysonProblems = listOf(
            problems.uryson.UrysonProblem.A,
            problems.uryson.UrysonProblem.B,
        )
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        for (problem in urysonProblems) {
            for (system in systems) {
                for (n in listOf(8, 16)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system, grid)
                    val funcs = numerics.functionals.ProjFunctionals(basis)
                    val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                    val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = problems.uryson.secondKindSolver(problem, basis, funcs, space, op)
                    val exact = { t: Double -> problem.exact(t) }
                    val prefix = "U.${problem.name}.${system.name}.n$n"
                    emit("$prefix.base", errorEh(exact, solver.base().eval, grid))
                    emit("$prefix.sloan", errorEh(exact, solver.sloan().eval, grid))
                    emit("$prefix.kulkarni", errorEh(exact, solver.kulkarni().eval, grid))
                    emit("$prefix.nystrom", errorEh(exact, solver.nystrom().eval, grid))
                }
            }
        }
    }

    /** Снимок решателей уравнений I рода (Фредгольм — Wazwaz, Вольтерра — дифференцирование). */
    @Test
    fun snapshotFirstKind() {
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = numerics.functionals.ProjFunctionals(basis)

            val fp = problems.fredholm.FredholmProblem.F1
            val fop = solvers.fredholm.FredholmOperator(fp.kernel, grid, GaussLegendre(8))
            val fSolver = problems.fredholm.firstKindSolver(fp, basis, funcs, fop)
            emit("F1.B.theta.n$n.base", errorEh({ t -> fp.exact(t) }, fSolver.base().eval, grid))
            emit("F1.B.theta.n$n.sloan", errorEh({ t -> fp.exact(t) }, fSolver.sloan().eval, grid))

            val vp = problems.volterra.VolterraProblem.V1
            val vop = solvers.volterra.VolterraOperator(vp.kernel, grid, GaussLegendre(8))
            val vSolver = problems.volterra.firstKindSolver(vp, basis, funcs, vop)
            emit("V1.B.theta.n$n.base", errorEh({ t -> vp.exact(t) }, vSolver.base().eval, grid))
            emit("V1.B.theta.n$n.sloan", errorEh({ t -> vp.exact(t) }, vSolver.sloan().eval, grid))
        }
    }
}
