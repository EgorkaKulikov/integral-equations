package demo.uryson

import demo.format.Fmt
import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import numerics.functionals.orders
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import solvers.uryson.FirstKindSolution
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import solvers.uryson.UrysonSecondKindSolver
import kotlin.math.abs

/**
 * Демонстрационная печать таблиц сходимости для нелинейного уравнения Урысона.
 *
 * Это не часть библиотеки, а иллюстрация её применения: код строит решатели на
 * последовательности сгущающихся сеток, вычисляет погрешность `E_h`, наблюдаемый
 * порядок `p_h` и выводит результат в консоль, а также в виде строк LaTeX.
 */
object Tables {
    /** Последовательность сеток: каждая следующая вдвое мельче предыдущей. */
    private val GRID_SIZES = listOf(8, 16, 32, 64, 128)

    /**
     * Фиксированное зерно генератора шума для задач первого рода.
     *
     * Явная константа делает численный эксперимент полностью воспроизводимым:
     * повторный запуск даёт те же зашумлённые данные и те же таблицы.
     */
    const val SEED = 20240517L

    /** Квадратура для всех демонстраций: порядок заведомо выше порядка аппроксимации. */
    private val quad = GaussLegendre(8)

    /** Ожидаемый порядок сходимости базовой схемы — используется для оценки константы `C_h`. */
    private const val EXPECTED_ORDER = 3.0

    /** Собирает решатель второго рода для заданной задачи, базиса и числа интервалов. */
    private fun makeSolver(
        problem: UrysonProblem,
        system: GeneratingSystem,
        n: Int,
    ): UrysonSecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, space, op)
    }

    /** Сходимость базовой схемы для задачи A на полиномиальном и гиперболическом базисах. */
    fun tableSecondKindOrder() {
        println("\n--- Задача A (второго рода): порядок сходимости, равномерная сетка ---")
        val errorsB = ArrayList<Double>()
        val errorsH = ArrayList<Double>()
        val steps = ArrayList<Double>()
        for (n in GRID_SIZES) {
            val solverB = makeSolver(UrysonProblem.A, GeneratingSystem.B, n)
            val solverH = makeSolver(UrysonProblem.A, GeneratingSystem.H, n)
            steps.add(solverB.grid.h)
            val exact = { t: Double -> UrysonProblem.A.exact(t) }
            errorsB.add(errorEh(exact, solverB.base().eval, solverB.grid))
            errorsH.add(errorEh(exact, solverH.base().eval, solverH.grid))
        }
        val ordersB = orders(errorsB)
        val ordersH = orders(errorsH)
        println("   n |    h    |   Eh(B)   | ph(B) |   Eh(H)   | ph(H)")
        for (i in GRID_SIZES.indices) {
            println(
                "%4d | %s | %s | %5s | %s | %5s".format(
                    GRID_SIZES[i], Fmt.h(steps[i]),
                    Fmt.e(errorsB[i]), Fmt.p(ordersB[i]),
                    Fmt.e(errorsH[i]), Fmt.p(ordersH[i]),
                ),
            )
        }
        println("   LaTeX:")
        for (i in GRID_SIZES.indices) {
            println(
                "   $${GRID_SIZES[i]}$ & $${Fmt.h(steps[i])}$ & $${Fmt.tex(errorsB[i])}$ & " +
                    "$${Fmt.p(ordersB[i])}$ & $${Fmt.tex(errorsH[i])}$ & $${Fmt.p(ordersH[i])}$ \\\\",
            )
        }
    }

    /** Сравнение трёх порождающих систем на задаче B. */
    fun tableGeneratingSystems() {
        println("\n--- Задача B (второго рода): три порождающие системы, равномерная сетка ---")
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val errors = systems.map { ArrayList<Double>() }
        val steps = ArrayList<Double>()
        for (n in GRID_SIZES) {
            for ((index, system) in systems.withIndex()) {
                val solver = makeSolver(UrysonProblem.B, system, n)
                if (index == 0) steps.add(solver.grid.h)
                errors[index].add(
                    errorEh({ t -> UrysonProblem.B.exact(t) }, solver.base().eval, solver.grid),
                )
            }
        }
        val observedOrders = systems.indices.map { orders(errors[it]) }
        println("   n |    h    | [B: Eh ph Ch] | [H: Eh ph Ch] | [T: Eh ph Ch]")
        for (i in GRID_SIZES.indices) {
            val columns = systems.indices.joinToString(" | ") { si ->
                "%s %5s %s".format(
                    Fmt.e(errors[si][i]),
                    Fmt.p(observedOrders[si][i]),
                    Fmt.e(errors[si][i] / Math.pow(steps[i], EXPECTED_ORDER)),
                )
            }
            println("%4d | %s | %s".format(GRID_SIZES[i], Fmt.h(steps[i]), columns))
        }
    }

    /** Сравнение четырёх схем второго рода на задаче B для двух базисов. */
    fun tableSchemes() {
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
            println("\n--- Задача B: сравнение схем, базис ${system.name} ---")
            val names = listOf("базовая", "Слоан", "Кулкарни", "Nystrom")
            val errors = names.map { ArrayList<Double>() }
            val steps = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val solver = makeSolver(UrysonProblem.B, system, n)
                steps.add(solver.grid.h)
                val exact = { t: Double -> UrysonProblem.B.exact(t) }
                errors[0].add(errorEh(exact, solver.base().eval, solver.grid))
                errors[1].add(errorEh(exact, solver.sloan().eval, solver.grid))
                errors[2].add(errorEh(exact, solver.kulkarni().eval, solver.grid))
                errors[3].add(errorEh(exact, solver.nystrom().eval, solver.grid))
            }
            val observedOrders = names.indices.map { orders(errors[it]) }
            println("   n |    h    | " + names.joinToString(" | ") { "$it: Eh ph" })
            for (i in GRID_SIZES.indices) {
                val columns = names.indices.joinToString(" | ") { mi ->
                    "%s %5s".format(Fmt.e(errors[mi][i]), Fmt.p(observedOrders[mi][i]))
                }
                println("%4d | %s | %s".format(GRID_SIZES[i], Fmt.h(steps[i]), columns))
            }
        }
    }

    /**
     * Регуляризованное решение задачи C первого рода при разных уровнях шума.
     *
     * Используется квазиравномерная сетка: на ней неравномерность шагов заметнее
     * проявляет влияние выбора порождающей системы.
     */
    fun tableFirstKindNoiseLevels() {
        println("\n--- Задача C (первого рода, регуляризация): базис H, квазиравномерная сетка ---")
        val deltas = listOf(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        println("  delta |  n  |   alpha   |    Eh     |  Eh_rel   |   res     |   Omega")
        for (i in deltas.indices) {
            val delta = deltas[i]
            val n = GRID_SIZES[i]
            val grid = Grid.quasiUniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
            val solver = firstKindSolver(basis, funcs, space, op)
            val thetaFDelta =
                noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, delta, SEED)
            val solution = solver.solveMorozov(thetaFDelta, delta)
            val error = errorEh({ t -> UrysonProblem.C.exact(t) }, solution.eval, grid)
            val exactNorm = (0..100 * n).maxOf { abs(UrysonProblem.C.exact(it.toDouble() / (100 * n))) }
            println(
                "  %s | %3d | %s | %s | %s | %s | %s".format(
                    Fmt.e(delta), n, Fmt.e(solution.alpha), Fmt.e(error),
                    Fmt.e(error / exactNorm), Fmt.e(solution.resid), Fmt.e(solution.omega),
                ),
            )
        }
    }

    /** Сравнение базисов B и H на задаче D первого рода при фиксированном уровне шума. */
    fun tableFirstKindBasisComparison() {
        println("\n--- Задача D (первого рода): базис B против H, delta = 1e-3, квазиравномерная сетка ---")
        val delta = 1e-3
        println("   n  |   alpha   |  Eh(B)   |  res(B)   |  Eh(H)   |  res(H)   | Eh_B/Eh_H")
        for (n in GRID_SIZES) {
            fun solveWith(system: GeneratingSystem): FirstKindSolution {
                val grid = Grid.quasiUniform(n)
                val basis = MinimalSplineBasis(system, grid)
                val funcs = ProjFunctionals(basis)
                val space = SplineSpace(basis, quad)
                val op = UrysohnOperator(UrysonProblem.D.kernel, grid, quad)
                val solver = firstKindSolver(basis, funcs, space, op)
                val thetaFDelta =
                    noisyThetaCoefficients(UrysonProblem.D, solver, op, grid, quad, delta, SEED)
                return solver.solveMorozov(thetaFDelta, delta)
            }
            val solutionB = solveWith(GeneratingSystem.B)
            val solutionH = solveWith(GeneratingSystem.H)
            val grid = Grid.quasiUniform(n)
            val exact = { t: Double -> UrysonProblem.D.exact(t) }
            val errorB = errorEh(exact, solutionB.eval, grid)
            val errorH = errorEh(exact, solutionH.eval, grid)
            println(
                "  %3d | %s | %s | %s | %s | %s | %s".format(
                    n, Fmt.e(solutionB.alpha), Fmt.e(errorB), Fmt.e(solutionB.resid),
                    Fmt.e(errorH), Fmt.e(solutionH.resid), Fmt.p(errorB / errorH),
                ),
            )
        }
    }
}

/** Точка входа демонстрации: печатает таблицы сходимости для уравнения Урысона. */
fun main() {
    println("=".repeat(72))
    println("Нелинейное уравнение Урысона: таблицы сходимости")
    println("Зерно генератора шума (задачи первого рода): ${Tables.SEED}")
    println("=".repeat(72))
    Tables.tableSecondKindOrder()
    Tables.tableGeneratingSystems()
    Tables.tableSchemes()
    Tables.tableFirstKindNoiseLevels()
    Tables.tableFirstKindBasisComparison()
    println("\nРасчёт завершён.")
}
