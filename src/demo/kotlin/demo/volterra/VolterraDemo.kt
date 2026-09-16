package demo.volterra

import demo.format.Fmt
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import numerics.orders
import problems.volterra.VolterraProblem
import problems.volterra.firstKindSolver
import problems.volterra.secondKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * Demonstration printout of convergence tables for linear Volterra equations.
 *
 * This is not part of the library but an illustration of its use: the code builds solvers on
 * a sequence of refined grids, computes the error `E_h` and the observed
 * order `p_h`, and prints the result to the console.
 */
object Tables {
    /**
     * Sequence of grids: each next value is twice as fine as the previous one.
     *
     * The limit `n <= 64` is due to the cost of the matrix `M2`: for the Volterra operator
     * it requires double integration with a variable upper limit, i.e.
     * `O(dim^2 * Q^2)` kernel evaluations.
     */
    private val GRID_SIZES = listOf(8, 16, 32, 64)

    /**
     * The grids for the Nyström table are restricted more tightly: the iterated variant applies
     * the exact Volterra operator to the approximation `u^N_h`, whose own evaluation costs
     * `O(n)` per point because of the `t`-dependent weights.
     */
    private val NYSTROM_GRID_SIZES = listOf(8, 16, 32)

    /** Quadrature for all demonstrations: the order is deliberately above the approximation order. */
    private val quad = GaussLegendre(8)

    private fun makeSolver(
        problem: VolterraProblem,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ): Pair<VolterraSecondKindSolver, Grid> {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        val op = VolterraOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, op) to grid
    }

    /**
     * Creates a functional family from its short name.
     * @throws IllegalArgumentException on an unknown name (protection against typos).
     */
    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi", "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "mu" -> AveragingFunctionals(basis)
        "lambda" -> ThreePointFunctionals(basis)
        else -> throw IllegalArgumentException("unknown functional family: '$name'")
    }

    /** Convergence of the three de Boor–Fix families (r = 0, 1, 2) on the bases B, H, T. */
    fun tableDeBoorFix(problem: VolterraProblem) {
        println("\n--- ${problem.name}: de Boor--Fix functionals xi<0>, xi<1>, xi<2>, bases B/H/T ---")
        val schemes = listOf("base", "Sloan", "Kulk", "it.Kulk")
        for (familyName in listOf("xi0", "xi1", "xi2")) {
            println("  family $familyName:")
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
                println("   basis ${system.name}:")
                printComparison(schemes, errors, GRID_SIZES, indent = "     ")
            }
        }
    }

    /** Error, observed order and constant for the base scheme on the bases B, H, T. */
    fun tablePhi(problem: VolterraProblem) {
        println("\n--- ${problem.name}: theta, bases B/H/T (E_h, p_h, C_h) ---")
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val errors = ArrayList<Double>()
            val steps = ArrayList<Double>()
            for (n in GRID_SIZES) {
                val (solver, grid) = makeSolver(problem, system, "theta", n)
                steps.add(grid.h)
                errors.add(errorEh({ t -> problem.exact(t) }, solver.base().eval, grid))
            }
            val observedOrders = orders(errors)
            println("  basis ${system.name}:")
            for (i in GRID_SIZES.indices) {
                // C_h = E_h / h^3: the theoretical order of the base scheme for quadratic splines.
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

    /** Comparison of the base scheme, the Sloan iteration and both Kulkarni schemes. */
    fun tableMethods(problem: VolterraProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: basis ${system.name}, theta: " +
                "base/Sloan/Kulkarni/iter.Kulkarni (E_h, p_h) ---",
        )
        val names = listOf("base", "Sloan", "Kulk", "it.Kulk")
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

    /** Comparison of the functional families theta, xi, mu, lambda on the base scheme. */
    fun tableFamilies(problem: VolterraProblem, system: GeneratingSystem) {
        println("\n--- ${problem.name}: basis ${system.name}, functional families, base scheme (E_h, p_h) ---")
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
     * Comparison of the Nyström quadrature schemes: classical, iterated, combined
     * and iterated combined.
     *
     * For the Volterra equation no proven superconvergence estimates are known in the literature,
     * so the orders observed here should be read as a numerical result
     * for the particular problems, not as a confirmation of a theorem.
     */
    fun tableNystrom(problem: VolterraProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: basis ${system.name}, theta: " +
                "base/Sloan/Nyström/iter.Nyström/comb.Nyström (E_h, p_h) ---",
        )
        val names = listOf("base", "Sloan", "Nyst", "it.Nyst", "comb.Nyst")
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
     * Equation of the first kind: reduction to an equation of the second kind by differentiation.
     *
     * The problem is well-posed because the kernel diagonal is nonzero (`K(t,t) = 1`),
     * which is what allows a single differentiation (the case `m = 1`).
     */
    fun tableFirstKind() {
        println("\n--- V1 (equation of the first kind, reduction by differentiation), basis B, theta ---")
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
                "   n=%4d E_h(base)=%s E_h(Sloan)=%s E_h(Kulk)=%s".format(
                    n, Fmt.e(baseError), Fmt.e(sloanError), Fmt.e(kulkarniError),
                ),
            )
        }
    }

    /** Prints a line-by-line comparison of several schemes with their observed orders. */
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
 * Entry point of the demonstration: prints convergence tables for three problems of the second kind
 * and one problem of the first kind.
 *
 * The correctness of the computational core is verified by tests (`./gradlew fastTest`), not
 * by this program.
 */
fun main() {
    println("=".repeat(72))
    println("Volterra equations: convergence tables")
    println("=".repeat(72))

    // Three problems of the second kind, differing in the behaviour of the kernel diagonal:
    //   V2    — rational kernel, K(t,t) != 0;
    //   V2exp — exponential kernel, K(t,t) != 0;
    //   V2win — smoothing kernel K = t - s, where K(t,t) = 0.
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
    println("\nComputation finished.")
}
