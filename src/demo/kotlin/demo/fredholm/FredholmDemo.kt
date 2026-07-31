package demo.fredholm

import demo.format.Fmt
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
import numerics.functionals.orders
import problems.fredholm.FredholmProblem
import problems.fredholm.firstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver

/**
 * Демонстрационная печать таблиц сходимости для линейных уравнений Фредгольма.
 *
 * Это не часть библиотеки, а иллюстрация её применения: код собирает решатели на
 * последовательности сгущающихся сеток, вычисляет погрешность `E_h` и наблюдаемый
 * порядок `p_h` и выводит результат в консоль.
 */
object Tables {
    /** Последовательность сеток: каждое следующее значение вдвое мельче предыдущего. */
    private val GRID_SIZES = listOf(8, 16, 32, 64)

    /** Квадратура для всех демонстраций: порядок заведомо выше порядка аппроксимации. */
    private val quad = GaussLegendre(8)

    private fun makeSolver(
        problem: FredholmProblem,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ): Pair<FredholmSecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        val op = FredholmOperator(problem.kernel, grid, quad)
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        return solver to grid
    }

    /**
     * Создаёт семейство функционалов по его краткому имени.
     * @throws IllegalArgumentException при неизвестном имени (защита от опечаток).
     */
    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi", "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        "lambda" -> ThreePointFunctionals(basis)
        else -> throw IllegalArgumentException("Неизвестное семейство функционалов: '$name'")
    }

    /** Сходимость трёх семейств де Бура–Фикса (r = 0, 1, 2) на базисах B, H, T. */
    fun tableDeBoorFix(problem: FredholmProblem) {
        println("\n--- ${problem.name}: функционалы де Бура--Фикса xi<0>, xi<1>, xi<2>, базисы B/H/T ---")
        val schemes = listOf("база", "Слоан", "Кулк", "ит.Кулк")
        for (familyName in listOf("xi0", "xi1", "xi2")) {
            println("  семейство $familyName:")
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val errors = schemes.map { ArrayList<Double>() }
                for (n in GRID_SIZES) {
                    val (solver, grid) = makeSolver(problem, system, familyName, n)
                    val exact = { t: Double -> problem.exact(t) }
                    errors[0].add(errorEh(exact, solver.base().eval, grid))
                    errors[1].add(errorEh(exact, solver.sloan().eval, grid))
                    errors[2].add(errorEh(exact, solver.kulkarni().eval, grid))
                    errors[3].add(errorEh(exact, solver.iteratedKulkarni().eval, grid))
                }
                val observedOrders = schemes.indices.map { orders(errors[it]) }
                println("   базис ${system.name}:")
                for (i in GRID_SIZES.indices) {
                    println(
                        "     n=%4d | ".format(GRID_SIZES[i]) +
                            schemes.indices.joinToString(" | ") { s ->
                                "%s:%s(%s)".format(schemes[s], Fmt.e(errors[s][i]), Fmt.p(observedOrders[s][i]))
                            },
                    )
                }
            }
        }
    }

    /** Базовая схема с функционалами theta на трёх порождающих системах: E_h, p_h, C_h. */
    fun tablePhi(problem: FredholmProblem) {
        println("\n--- ${problem.name}: theta, базисы B/H/T (E_h, p_h, C_h) ---")
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val errors = ArrayList<Double>()
            val steps = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val (solver, grid) = makeSolver(problem, system, "theta", n)
                steps.add(grid.h)
                errors.add(errorEh({ t -> problem.exact(t) }, solver.base().eval, grid))
            }
            val observedOrders = orders(errors)
            println("  базис ${system.name}:")
            for (i in GRID_SIZES.indices) {
                // C_h = E_h / h^3: асимптотическая константа при теоретическом порядке 3.
                val constant = errors[i] / Math.pow(steps[i], 3.0)
                println(
                    "   n=%4d h=%s E_h=%s p_h=%s C_h=%s".format(
                        GRID_SIZES[i], Fmt.h(steps[i]), Fmt.e(errors[i]),
                        Fmt.p(observedOrders[i]), Fmt.e(constant),
                    ),
                )
            }
        }
    }

    /** Сравнение схем: базовая, Слоан, Кулкарни, итерированный Кулкарни. */
    fun tableMethods(problem: FredholmProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: базис ${system.name}, theta: " +
                "база/Слоан/Кулкарни/итер.Кулкарни (E_h, p_h) ---",
        )
        val names = listOf("база", "Слоан", "Кулк", "ит.Кулк")
        val errors = names.map { ArrayList<Double>() }
        for (n in GRID_SIZES) {
            val (solver, grid) = makeSolver(problem, system, "theta", n)
            val exact = { t: Double -> problem.exact(t) }
            errors[0].add(errorEh(exact, solver.base().eval, grid))
            errors[1].add(errorEh(exact, solver.sloan().eval, grid))
            errors[2].add(errorEh(exact, solver.kulkarni().eval, grid))
            errors[3].add(errorEh(exact, solver.iteratedKulkarni().eval, grid))
        }
        printComparison(names, errors)
    }

    /** Сравнение семейств функционалов theta / xi / mu / lambda на базовой схеме. */
    fun tableFamilies(problem: FredholmProblem, system: GeneratingSystem) {
        println("\n--- ${problem.name}: базис ${system.name}, базовая схема, семейства (E_h, p_h) ---")
        for (familyName in listOf("theta", "xi", "mu", "lambda")) {
            val errors = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val (solver, grid) = makeSolver(problem, system, familyName, n)
                errors.add(errorEh({ t -> problem.exact(t) }, solver.base().eval, grid))
            }
            val observedOrders = orders(errors)
            println(
                "  %-7s: ".format(familyName) +
                    GRID_SIZES.indices.joinToString(" ") { i ->
                        "%s(%s)".format(Fmt.e(errors[i]), Fmt.p(observedOrders[i]))
                    },
            )
        }
    }

    /**
     * Сравнение схем Nyström: классическая («голая» квадратура) и комбинированный
     * оператор `L_n = P_chi L + (I - P_chi) L^N_h`, к которому и относятся
     * опубликованные оценки суперсходимости (см. `docs/REFERENCES.md`).
     */
    fun tableNystrom(problem: FredholmProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: базис ${system.name}, theta: " +
                "база/Слоан/Nyström/итер.Nyström/комб.Nyström/итер.комб. (E_h, p_h) ---",
        )
        val names = listOf("база", "Слоан", "Nyst", "ит.Nyst", "комб.Nyst", "ит.комб")
        val errors = names.map { ArrayList<Double>() }
        for (n in GRID_SIZES) {
            val (solver, grid) = makeSolver(problem, system, "theta", n)
            val exact = { t: Double -> problem.exact(t) }
            errors[0].add(errorEh(exact, solver.base().eval, grid))
            errors[1].add(errorEh(exact, solver.sloan().eval, grid))
            errors[2].add(errorEh(exact, solver.nystrom().eval, grid))
            errors[3].add(errorEh(exact, solver.iteratedNystrom().eval, grid))
            errors[4].add(errorEh(exact, solver.combinedNystrom().eval, grid))
            errors[5].add(errorEh(exact, solver.iteratedCombinedNystrom().eval, grid))
        }
        printComparison(names, errors)
    }

    /**
     * Уравнение первого рода, решаемое методом регуляризации.
     *
     * Публикуются только базовая схема и итерация Слоана: схема Кулкарни использует
     * матрицу `M2`, элементы которой растут как `alpha^{-2}`, и при `alpha = 1e-10`
     * численно неприменима.
     */
    fun tableFirstKind() {
        val alpha = 1e-10
        println("\n--- F1 (регуляризация, alpha=$alpha), базис H, theta: база/Слоан ---")
        println("    Примечание: обусловленность растёт как alpha^{-1}; alpha подобрано экспериментально.")
        val problem = FredholmProblem.F1
        for (n in listOf(8, 16, 32)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val op = FredholmOperator(problem.kernel, grid, quad)
            val solver = firstKindSolver(problem, basis, ProjFunctionals(basis), op, alpha)
            val exact = { t: Double -> problem.exact(t) }
            val baseError = errorEh(exact, solver.base().eval, grid)
            val sloanError = errorEh(exact, solver.sloan().eval, grid)
            println("   n=%4d E_h(база)=%s E_h(Слоан)=%s".format(n, Fmt.e(baseError), Fmt.e(sloanError)))
        }
    }

    /** Печатает построчное сравнение нескольких схем с их наблюдаемыми порядками. */
    private fun printComparison(names: List<String>, errors: List<List<Double>>) {
        val observedOrders = names.indices.map { orders(errors[it]) }
        for (i in GRID_SIZES.indices) {
            println(
                "   n=%4d | ".format(GRID_SIZES[i]) +
                    names.indices.joinToString(" | ") { s ->
                        "%s:%s(%s)".format(names[s], Fmt.e(errors[s][i]), Fmt.p(observedOrders[s][i]))
                    },
            )
        }
    }
}

/**
 * Точка входа демонстрации: печатает таблицы сходимости для двух задач второго рода
 * и одной задачи первого рода.
 *
 * Корректность вычислительного ядра проверяется тестами (`./gradlew fastTest`), а не
 * этой программой.
 */
fun main() {
    println("=".repeat(72))
    println("Уравнения Фредгольма: таблицы сходимости")
    println("=".repeat(72))

    // Две задачи второго рода: рациональное решение и экспоненциальное.
    val secondKindExamples = listOf(
        FredholmProblem.F2 to GeneratingSystem.B,
        FredholmProblem.F2exp to GeneratingSystem.B,
    )
    for ((problem, system) in secondKindExamples) {
        Tables.tablePhi(problem)
        Tables.tableMethods(problem, system)
        Tables.tableNystrom(problem, system)
        Tables.tableFamilies(problem, system)
        Tables.tableDeBoorFix(problem)
    }

    Tables.tableFirstKind()
    println("\nРасчёт завершён.")
}
