package demo.volterra

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
import problems.volterra.VolterraProblem
import problems.volterra.firstKindSolver
import problems.volterra.secondKindSolver
import solvers.volterra.SecondKindSolver
import solvers.volterra.VolterraOperator

/**
 * Демонстрационная печать таблиц сходимости для линейных уравнений Вольтерры.
 *
 * Это не часть библиотеки, а иллюстрация её применения: код собирает решатели на
 * последовательности сгущающихся сеток, вычисляет погрешность `E_h` и наблюдаемый
 * порядок `p_h` и выводит результат в консоль.
 */
object Tables {
    /**
     * Последовательность сеток: каждое следующее значение вдвое мельче предыдущего.
     *
     * Ограничение `n <= 64` связано со стоимостью матрицы `M2`: для оператора Вольтерры
     * она требует двойного интегрирования с переменным верхним пределом, то есть
     * `O(dim^2 * Q^2)` вычислений ядра.
     */
    private val GRID_SIZES = listOf(8, 16, 32, 64)

    /**
     * Сетки для таблицы Nyström ограничены сильнее: итерированный вариант применяет
     * точный оператор Вольтерры к приближению `u^N_h`, вычисление которого само стоит
     * `O(n)` на точку из-за зависящих от `t` весов.
     */
    private val NYSTROM_GRID_SIZES = listOf(8, 16, 32)

    /** Квадратура для всех демонстраций: порядок заведомо выше порядка аппроксимации. */
    private val quad = GaussLegendre(8)

    private fun makeSolver(
        problem: VolterraProblem,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ): Pair<SecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        val op = VolterraOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, op) to grid
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
    fun tableDeBoorFix(problem: VolterraProblem) {
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
                println("   базис ${system.name}:")
                printComparison(schemes, errors, GRID_SIZES, indent = "     ")
            }
        }
    }

    /** Погрешность, наблюдаемый порядок и константа для базовой схемы на базисах B, H, T. */
    fun tablePhi(problem: VolterraProblem) {
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
                // C_h = E_h / h^3: теоретический порядок базовой схемы для квадратичных сплайнов.
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

    /** Сравнение базовой схемы, итерации Слоана и обеих схем Кулкарни. */
    fun tableMethods(problem: VolterraProblem, system: GeneratingSystem) {
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
        printComparison(names, errors, GRID_SIZES)
    }

    /** Сравнение семейств функционалов theta, xi, mu, lambda на базовой схеме. */
    fun tableFamilies(problem: VolterraProblem, system: GeneratingSystem) {
        println("\n--- ${problem.name}: базис ${system.name}, семейства функционалов, базовая схема (E_h, p_h) ---")
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
     * Сравнение квадратурных схем Nyström: классической, итерированной, комбинированной
     * и итерированной комбинированной.
     *
     * Для уравнения Вольтерры доказанных оценок суперсходимости в известной литературе
     * нет, поэтому наблюдаемые здесь порядки следует трактовать как численный результат
     * для конкретных задач, а не как подтверждение теоремы.
     */
    fun tableNystrom(problem: VolterraProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: базис ${system.name}, theta: " +
                "база/Слоан/Nyström/итер.Nyström/комб.Nyström (E_h, p_h) ---",
        )
        val names = listOf("база", "Слоан", "Nyst", "ит.Nyst", "комб.Nyst")
        val errors = names.map { ArrayList<Double>() }
        for (n in NYSTROM_GRID_SIZES) {
            val (solver, grid) = makeSolver(problem, system, "theta", n)
            val exact = { t: Double -> problem.exact(t) }
            errors[0].add(errorEh(exact, solver.base().eval, grid))
            errors[1].add(errorEh(exact, solver.sloan().eval, grid))
            errors[2].add(errorEh(exact, solver.nystrom().eval, grid))
            errors[3].add(errorEh(exact, solver.iteratedNystrom().eval, grid))
            errors[4].add(errorEh(exact, solver.combinedNystrom().eval, grid))
        }
        printComparison(names, errors, NYSTROM_GRID_SIZES)
    }

    /**
     * Уравнение первого рода: сведение к уравнению второго рода дифференцированием.
     *
     * Задача корректна, поскольку диагональ ядра отлична от нуля (`K(t,t) = 1`),
     * что и позволяет применить однократное дифференцирование (случай `m = 1`).
     */
    fun tableFirstKind() {
        println("\n--- V1 (уравнение I рода, сведение дифференцированием), базис B, theta ---")
        val problem = VolterraProblem.V1
        for (n in GRID_SIZES) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = VolterraOperator(problem.kernel, grid, quad)
            val solver = firstKindSolver(problem, basis, ProjFunctionals(basis), op)
            val exact = { t: Double -> problem.exact(t) }
            val baseError = errorEh(exact, solver.base().eval, grid)
            val sloanError = errorEh(exact, solver.sloan().eval, grid)
            val kulkarniError = errorEh(exact, solver.kulkarni().eval, grid)
            println(
                "   n=%4d E_h(база)=%s E_h(Слоан)=%s E_h(Кулк)=%s".format(
                    n, Fmt.e(baseError), Fmt.e(sloanError), Fmt.e(kulkarniError),
                ),
            )
        }
    }

    /** Печатает построчное сравнение нескольких схем с их наблюдаемыми порядками. */
    private fun printComparison(
        names: List<String>,
        errors: List<List<Double>>,
        gridSizes: List<Int>,
        indent: String = "   ",
    ) {
        val observedOrders = names.indices.map { orders(errors[it]) }
        for (i in gridSizes.indices) {
            println(
                "%sn=%4d | ".format(indent, gridSizes[i]) +
                    names.indices.joinToString(" | ") { s ->
                        "%s:%s(%s)".format(names[s], Fmt.e(errors[s][i]), Fmt.p(observedOrders[s][i]))
                    },
            )
        }
    }
}

/**
 * Точка входа демонстрации: печатает таблицы сходимости для трёх задач второго рода
 * и одной задачи первого рода.
 *
 * Корректность вычислительного ядра проверяется тестами (`./gradlew test`), а не
 * этой программой.
 */
fun main() {
    println("=".repeat(72))
    println("Уравнения Вольтерры: таблицы сходимости")
    println("=".repeat(72))

    // Три задачи второго рода, различающиеся поведением диагонали ядра:
    //   V2    — рациональное ядро, K(t,t) != 0;
    //   V2exp — экспоненциальное ядро, K(t,t) != 0;
    //   V2win — сглаживающее ядро K = t - s, где K(t,t) = 0.
    val secondKindExamples = listOf(
        VolterraProblem.V2 to GeneratingSystem.B,
        VolterraProblem.V2exp to GeneratingSystem.B,
        VolterraProblem.V2win to GeneratingSystem.B,
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
