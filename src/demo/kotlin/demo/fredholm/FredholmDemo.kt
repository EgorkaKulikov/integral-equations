package demo.fredholm

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
import problems.fredholm.FredholmProblem
import problems.fredholm.firstKindSolver
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver

/**
 * Demonstration printout of convergence tables for linear Fredholm equations.
 *
 * This is not part of the library but an illustration of its use: the code builds solvers on
 * a sequence of refined grids, computes the error `E_h` and the observed
 * order `p_h`, and prints the result to the console.
 */
object Tables {
    /** Sequence of grids: each next value is twice as fine as the previous one. */
    private val GRID_SIZES = listOf(8, 16, 32, 64)

    /** Quadrature for all demonstrations: the order is deliberately above the approximation order. */
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
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
        )
        return solver to grid
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
    fun tableDeBoorFix(problem: FredholmProblem) {
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
                val observedOrders = schemes.indices.map { orders(errors[it]) }
                println("   basis ${system.name}:")
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

    /** Base scheme with the theta functionals on the three generating systems: E_h, p_h, C_h. */
    fun tablePhi(problem: FredholmProblem) {
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
                // C_h = E_h / h^3: the asymptotic constant at the theoretical order 3.
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

    /** Comparison of the schemes: base, Sloan, Kulkarni, iterated Kulkarni. */
    fun tableMethods(problem: FredholmProblem, system: GeneratingSystem) {
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
        printComparison(names, errors)
    }

    /** Comparison of the functional families theta / xi / mu / lambda on the base scheme. */
    fun tableFamilies(problem: FredholmProblem, system: GeneratingSystem) {
        println("\n--- ${problem.name}: basis ${system.name}, base scheme, families (E_h, p_h) ---")
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
     * Comparison of the Nyström schemes: the classical one ("bare" quadrature) and the combined
     * operator `L_n = P_chi L + (I - P_chi) L^N_h`, to which the published superconvergence
     * estimates apply (see `docs/REFERENCES.md`).
     */
    fun tableNystrom(problem: FredholmProblem, system: GeneratingSystem) {
        println(
            "\n--- ${problem.name}: basis ${system.name}, theta: " +
                "base/Sloan/Nyström/iter.Nyström/comb.Nyström/iter.comb (E_h, p_h) ---",
        )
        val names = listOf("base", "Sloan", "Nyst", "it.Nyst", "comb.Nyst", "it.comb")
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
     * Equation of the first kind, solved by the regularization method.
     *
     * Only the base scheme and the Sloan iteration are published: the Kulkarni scheme uses
     * the matrix `M2`, whose entries grow as `alpha^{-2}`, and at `alpha = 1e-10`
     * it is numerically inapplicable.
     */
    fun tableFirstKind() {
        val alpha = 1e-10
        println("\n--- F1 (regularization, alpha=$alpha), basis H, theta: base/Sloan ---")
        println("    Note: the conditioning grows as alpha^{-1}; alpha was tuned experimentally.")
        val problem = FredholmProblem.F1
        for (n in listOf(8, 16, 32)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val op = FredholmOperator(problem.kernel, grid, quad)
            val solver = firstKindSolver(problem, basis, ProjFunctionals(basis), op, alpha)
            val exact = { t: Double -> problem.exact(t) }
            val baseError = errorEh(exact, solver.base().eval, grid)
            val sloanError = errorEh(exact, solver.sloan().eval, grid)
            println("   n=%4d E_h(base)=%s E_h(Sloan)=%s".format(n, Fmt.e(baseError), Fmt.e(sloanError)))
        }
    }

    /** Prints a line-by-line comparison of several schemes with their observed orders. */
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
 * Entry point of the demonstration: prints convergence tables for two problems of the second kind
 * and one problem of the first kind.
 *
 * The correctness of the computational core is verified by tests (`./gradlew fastTest`), not
 * by this program.
 */
fun main() {
    println("=".repeat(72))
    println("Fredholm equations: convergence tables")
    println("=".repeat(72))

    // Two problems of the second kind: with a rational and with an exponential solution.
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
    println("\nComputation finished.")
}
