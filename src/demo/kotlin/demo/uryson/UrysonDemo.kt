package demo.uryson

import demo.format.Fmt
import numerics.GaussLegendre
import numerics.orders
import problems.uryson.NoiseNorm
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import solvers.uryson.FirstKindSolution
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import solvers.uryson.UrysonSecondKindSolver

/**
 * Демонстрационная печать таблиц сходимости для нелинейного уравнения Урысона.
 *
 * Это не часть библиотеки, а иллюстрация её применения: код строит решатели на
 * последовательности сгущающихся сеток, вычисляет погрешность `E_h`, наблюдаемый
 * порядок `p_h` и печатает результат в консоль и в виде строк LaTeX в формате
 * таблиц статьи new-01 (`booktabs` + `siunitx`, столбцы `S[table-format=1.3e-2]`).
 *
 * Правило достоверности (как в new-01): значения `E_h < 1e-12` определяются
 * округлением и печатаются прочерком, как и порядки `p_h`, у которых хотя бы одна
 * из двух соседних погрешностей ниже порога или следующей сетки нет.
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

    /** Порог достоверности: ниже него погрешность определяется округлением (new-01, sec:protocol). */
    private const val RELIABILITY_FLOOR = 1e-12

    /** Отношение соседних шагов неравномерной сетки (new-01, eq:alt-mesh: mu = 2). */
    private const val GRADED_RATIO = 2.0

    /** Нормировка шума для задач первого рода: в `C[a,b]`, согласованно с условием (VII). */
    private val NOISE_NORM = NoiseNorm.SUP

    // ------------------------------------------------------------------ формат LaTeX

    /** Прочерк в ячейке `S`-столбца. */
    private const val DASH = "\\multicolumn{1}{c}{---}"

    /** Погрешность в формате `1.014e-4` либо прочерк ниже порога. */
    private fun texE(x: Double): String =
        if (x.isNaN() || x < RELIABILITY_FLOOR) DASH else "%.3e".format(x).replace("e-0", "e-").replace("e+0", "e")

    /**
     * Порядок `p_h` по паре соседних погрешностей: прочерк, если следующей сетки нет
     * либо одна из погрешностей ниже порога.
     */
    private fun texP(errs: List<Double>, i: Int): String {
        if (i + 1 >= errs.size) return DASH
        if (errs[i] < RELIABILITY_FLOOR || errs[i + 1] < RELIABILITY_FLOOR) return DASH
        return "%.2f".format(Math.log(errs[i] / errs[i + 1]) / Math.log(2.0))
    }

    private fun gridOf(kind: String, n: Int): Grid =
        if (kind == "uniform") Grid.uniform(n) else Grid.graded(n, ratio = GRADED_RATIO)

    /** Собирает решатель второго рода для заданной задачи, базиса и сетки. */
    private fun makeSolver(problem: UrysonProblem, system: GeneratingSystem, grid: Grid): UrysonSecondKindSolver {
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, space, op)
    }

    // ------------------------------------------------------------------ второй род

    /** Сходимость базовой схемы для задачи A на базисах B и H, равномерная и неравномерная сетки. */
    fun tableSecondKindOrder() {
        for (kind in listOf("uniform", "graded")) {
            println("\n--- Задача A (второго рода): базовая схема, сетка $kind ---")
            val errorsB = ArrayList<Double>()
            val errorsH = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val grid = gridOf(kind, n)
                val exact = { t: Double -> UrysonProblem.A.exact(t) }
                errorsB.add(errorEh(exact, makeSolver(UrysonProblem.A, GeneratingSystem.B, grid).base().eval, grid))
                errorsH.add(errorEh(exact, makeSolver(UrysonProblem.A, GeneratingSystem.H, grid).base().eval, grid))
            }
            val ordersB = orders(errorsB)
            val ordersH = orders(errorsH)
            println("   n |   Eh(B)   | ph(B) |   Eh(H)   | ph(H)")
            for (i in GRID_SIZES.indices) {
                println(
                    "%4d | %s | %5s | %s | %5s".format(
                        GRID_SIZES[i], Fmt.e(errorsB[i]), Fmt.p(ordersB[i]), Fmt.e(errorsH[i]), Fmt.p(ordersH[i]),
                    ),
                )
            }
            println("   LaTeX (S-столбцы):")
            for (i in GRID_SIZES.indices) {
                println("    ${GRID_SIZES[i]} & ${texE(errorsB[i])} & ${texP(errorsB, i)} & ${texE(errorsH[i])} & ${texP(errorsH, i)} \\\\")
            }
        }
    }

    /** Сравнение трёх порождающих систем на задаче B (базовая схема). */
    fun tableGeneratingSystems() {
        println("\n--- Задача B (второго рода): три порождающие системы, равномерная сетка ---")
        val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val errors = systems.map { ArrayList<Double>() }
        val steps = ArrayList<Double>()
        for (n in GRID_SIZES) {
            val grid = Grid.uniform(n)
            steps.add(grid.h)
            for ((index, system) in systems.withIndex()) {
                val solver = makeSolver(UrysonProblem.B, system, grid)
                errors[index].add(errorEh({ t -> UrysonProblem.B.exact(t) }, solver.base().eval, grid))
            }
        }
        val observedOrders = systems.indices.map { orders(errors[it]) }
        println("   n |    h    | [B: Eh ph Ch] | [H: Eh ph Ch] | [T: Eh ph Ch]")
        for (i in GRID_SIZES.indices) {
            val columns = systems.indices.joinToString(" | ") { si ->
                "%s %5s %s".format(
                    Fmt.e(errors[si][i]), Fmt.p(observedOrders[si][i]),
                    Fmt.e(errors[si][i] / Math.pow(steps[i], EXPECTED_ORDER)),
                )
            }
            println("%4d | %s | %s".format(GRID_SIZES[i], Fmt.h(steps[i]), columns))
        }
        println("   LaTeX (S-столбцы: n, [Eh ph Ch] x3):")
        for (i in GRID_SIZES.indices) {
            val cells = systems.indices.joinToString(" & ") { si ->
                val e = errors[si][i]
                val ch = if (e < RELIABILITY_FLOOR) DASH else "%.3e".format(e / Math.pow(steps[i], EXPECTED_ORDER)).replace("e-0", "e-")
                "${texE(e)} & ${texP(errors[si], i)} & $ch"
            }
            println("    ${GRID_SIZES[i]} & $cells \\\\")
        }
    }

    /** Все схемы второго рода на задаче B для базисов B и H. */
    fun tableSchemes() {
        val names = listOf("база", "Слоан", "Кулкарни", "итер. Кулкарни", "Nyström", "комб. Nyström", "итер. Nyström")
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H)) {
            println("\n--- Задача B: сравнение схем, базис ${system.name}, равномерная сетка ---")
            val errors = names.map { ArrayList<Double>() }
            for (n in GRID_SIZES) {
                val grid = Grid.uniform(n)
                val solver = makeSolver(UrysonProblem.B, system, grid)
                val exact = { t: Double -> UrysonProblem.B.exact(t) }
                errors[0].add(errorEh(exact, solver.base().eval, grid))
                errors[1].add(errorEh(exact, solver.sloan().eval, grid))
                errors[2].add(errorEh(exact, solver.kulkarni().eval, grid))
                errors[3].add(errorEh(exact, solver.iteratedKulkarni().eval, grid))
                errors[4].add(errorEh(exact, solver.nystrom().eval, grid))
                errors[5].add(errorEh(exact, solver.combinedNystrom().eval, grid))
                errors[6].add(errorEh(exact, solver.iteratedNystrom().eval, grid))
            }
            val observedOrders = names.indices.map { orders(errors[it]) }
            println("   n | " + names.joinToString(" | ") { "$it: Eh ph" })
            for (i in GRID_SIZES.indices) {
                val columns = names.indices.joinToString(" | ") { mi -> "%s %5s".format(Fmt.e(errors[mi][i]), Fmt.p(observedOrders[mi][i])) }
                println("%4d | %s".format(GRID_SIZES[i], columns))
            }
            println("   LaTeX (S-столбцы: n, [Eh ph] x7 в порядке: ${names.joinToString(", ")}):")
            for (i in GRID_SIZES.indices) {
                val cells = names.indices.joinToString(" & ") { mi -> "${texE(errors[mi][i])} & ${texP(errors[mi], i)}" }
                println("    ${GRID_SIZES[i]} & $cells \\\\")
            }
        }
    }

    // ------------------------------------------------------------------ первый род

    private fun solveFirstKind(problem: UrysonProblem, system: GeneratingSystem, grid: Grid, delta: Double): FirstKindSolution {
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta = noisyThetaCoefficients(problem, solver, op, grid, quad, delta, SEED, NOISE_NORM)
        return solver.solveMorozov(thetaFDelta, delta)
    }

    /**
     * Регуляризованное решение задачи C первого рода при согласованном убывании уровня
     * шума и шага. Сетка — неравномерная с чередующимися шагами (new-01, eq:alt-mesh);
     * шум задан в `C[a,b]` (`||xi||_infty = delta`).
     */
    fun tableFirstKindNoiseLevels() {
        println("\n--- Задача C (первого рода, регуляризация): базис H, сетка graded(2), шум SUP ---")
        val deltas = listOf(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
        println("  delta |  n  |   alpha   |    Eh     |   res     |   Omega")
        val rows = ArrayList<String>()
        for (i in deltas.indices) {
            val delta = deltas[i]
            val n = GRID_SIZES[i]
            val grid = Grid.graded(n, ratio = GRADED_RATIO)
            val solution = solveFirstKind(UrysonProblem.C, GeneratingSystem.H, grid, delta)
            val error = errorEh({ t -> UrysonProblem.C.exact(t) }, solution.eval, grid)
            println(
                "  %s | %3d | %s | %s | %s | %s".format(
                    Fmt.e(delta), n, Fmt.e(solution.alpha), Fmt.e(error), Fmt.e(solution.resid), Fmt.e(solution.omega),
                ),
            )
            rows.add("    ${texE(delta)} & $n & ${texE(solution.alpha)} & ${texE(error)} & ${texE(solution.resid)} & ${"%.3f".format(solution.omega)} \\\\")
        }
        println("   LaTeX (S-столбцы: delta, n, alpha, Eh, res, Omega):")
        rows.forEach { println(it) }
    }

    /** Сравнение базисов B и H на задаче D первого рода при фиксированном уровне шума. */
    fun tableFirstKindBasisComparison() {
        println("\n--- Задача D (первого рода): базис B против H, delta = 1e-3 (SUP), сетка graded(2) ---")
        val delta = 1e-3
        println("   n  |   alpha   |  Eh(B)   |  res(B)   |  Eh(H)   |  res(H)")
        val rows = ArrayList<String>()
        for (n in GRID_SIZES) {
            val grid = Grid.graded(n, ratio = GRADED_RATIO)
            val solutionB = solveFirstKind(UrysonProblem.D, GeneratingSystem.B, grid, delta)
            val solutionH = solveFirstKind(UrysonProblem.D, GeneratingSystem.H, grid, delta)
            val exact = { t: Double -> UrysonProblem.D.exact(t) }
            val errorB = errorEh(exact, solutionB.eval, grid)
            val errorH = errorEh(exact, solutionH.eval, grid)
            println(
                "  %3d | %s | %s | %s | %s | %s".format(
                    n, Fmt.e(solutionB.alpha), Fmt.e(errorB), Fmt.e(solutionB.resid), Fmt.e(errorH), Fmt.e(solutionH.resid),
                ),
            )
            rows.add("    $n & ${texE(solutionB.alpha)} & ${texE(errorB)} & ${texE(solutionB.resid)} & ${texE(solutionH.alpha)} & ${texE(errorH)} & ${texE(solutionH.resid)} \\\\")
        }
        println("   LaTeX (S-столбцы: n, alpha(B), Eh(B), res(B), alpha(H), Eh(H), res(H)):")
        rows.forEach { println(it) }
    }
}

/** Точка входа демонстрации: печатает таблицы сходимости для уравнения Урысона. */
fun main() {
    println("=".repeat(72))
    println("Нелинейное уравнение Урысона: таблицы сходимости")
    println("Зерно генератора шума (задачи первого рода): ${Tables.SEED}; нормировка шума: SUP")
    println("Квадратура: Гаусс--Лежандр, 8 узлов на ячейку; порог достоверности 1e-12")
    println("=".repeat(72))
    Tables.tableSecondKindOrder()
    Tables.tableGeneratingSystems()
    Tables.tableSchemes()
    Tables.tableFirstKindNoiseLevels()
    Tables.tableFirstKindBasisComparison()
    println("\nРасчёт завершён.")
}
