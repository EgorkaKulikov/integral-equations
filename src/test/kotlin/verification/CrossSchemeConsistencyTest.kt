package verification

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.AveragingFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraSecondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * A CROSS-CHECK OF THE SCHEMES (task 2.4).
 *
 * Different schemes approximate ONE AND THE SAME solution of one equation, so on a
 * sufficiently fine grid they must be consistent with each other: the pairwise difference
 * cannot substantially exceed the sum of their own errors.
 *
 * What the check gives. It does not prove correctness (all the schemes could err
 * consistently — because of a common error in the basis or the quadrature, say), but it
 * detects the situation where ONE scheme is out of agreement with the rest. This
 * complements the cross-check with the publication: there the numbers are checked, here the mutual
 * consistency of independently implemented computation paths.
 *
 * The difference from `PublishedValuesTest`: what is compared is not the aggregate quantities `E_h`,
 * but the VALUES OF THE SOLUTIONS at a set of points. A coincidence of the `E_h` of two schemes does not yet mean
 * that they give one function: the maximum of the error could be attained at different
 * points with a different behaviour in between.
 */
@Tag("slow")
class CrossSchemeConsistencyTest {

    private companion object {
        /**
         * The margin factor when comparing two schemes.
         *
         * The justification. From the triangle inequality
         *   |u_A(t) - u_B(t)| <= |u_A(t) - u*(t)| + |u*(t) - u_B(t)| <= E_A + E_B <= 2 max(E_A, E_B),
         * that is, the mathematically guaranteed bound equals 2. The factor 4 takes
         * a twofold margin for the fact that `E_h` is measured on a finite sample of points
         * (`errorEh` uses 100n+1 points) and may slightly underestimate the true maximum
         * of the uniform norm.
         *
         * A larger margin must not be taken: at a factor of the order of tens the check
         * would stop distinguishing consistent schemes from an inconsistent one.
         */
        const val PAIRWISE_SAFETY_FACTOR = 4.0

        /**
         * The absolute "floor" of the comparison.
         *
         * When both schemes have reached machine accuracy, their errors are determined by
         * rounding, and the requirement `|u_A - u_B| <= 4 max(E_A, E_B)` would turn
         * into a comparison of two noises. Below this threshold a discrepancy is considered
         * insubstantial regardless of the ratio of the quantities.
         */
        const val ABSOLUTE_FLOOR = 1e-12

        /** The quadrature order — the same as in the other checks and demos. */
        const val QUADRATURE_ORDER = 8

        /** The number of points of comparison of the solutions inside the interval. */
        const val COMPARISON_POINTS = 200
    }

    /** The comparison points, uniformly covering the interval of the grid (the ends included). */
    private fun comparisonPoints(grid: Grid): DoubleArray =
        DoubleArray(COMPARISON_POINTS + 1) { i ->
            grid.a + (grid.b - grid.a) * i / COMPARISON_POINTS
        }

    /** A named solution: the evaluator and its own error `E_h`. */
    private class NamedSolution(val name: String, val eval: (Double) -> Double, val error: Double)

    /**
     * Checks the pairwise consistency of all the schemes of the set.
     *
     * @param context the description of the combination "problem/basis/family/grid" for the message.
     * @param solutions the schemes solving one and the same problem.
     * @param points the points at which the values are compared.
     * @param failures the accumulator of the detected inconsistencies.
     */
    private fun checkPairwise(
        context: String,
        solutions: List<NamedSolution>,
        points: DoubleArray,
        failures: MutableList<String>,
    ) {
        for (i in solutions.indices) {
            for (j in i + 1 until solutions.size) {
                val a = solutions[i]
                val b = solutions[j]
                var worstDifference = 0.0
                var worstPoint = Double.NaN
                for (t in points) {
                    val difference = abs(a.eval(t) - b.eval(t))
                    if (difference > worstDifference) {
                        worstDifference = difference
                        worstPoint = t
                    }
                }
                val allowed = maxOf(
                    PAIRWISE_SAFETY_FACTOR * maxOf(a.error, b.error),
                    ABSOLUTE_FLOOR,
                )
                if (worstDifference > allowed) {
                    failures += buildString {
                        append(context)
                        append(": the schemes '").append(a.name).append("' and '").append(b.name)
                        append("' are inconsistent. max|u_A - u_B| = ").append(worstDifference)
                        append(" at the point t = ").append(worstPoint)
                        append(", allowed ").append(allowed)
                        append(" (= ").append(PAIRWISE_SAFETY_FACTOR).append(" * max(E_A, E_B)); ")
                        append("E(").append(a.name).append(") = ").append(a.error)
                        append(", E(").append(b.name).append(") = ").append(b.error)
                    }
                }
            }
        }
    }

    private fun families(basis: MinimalSplineBasis): List<FunctionalFamily> = listOf(
        // Only the families WITHOUT a derivative: the Nyström schemes do not support xi.
        ProjFunctionals(basis),
        AveragingFunctionals(basis),
        ThreePointFunctionals(basis),
    )

    private fun reportFailures(failures: List<String>, title: String) {
        assertTrue(
            failures.isEmpty(),
            "$title: an inconsistency of the schemes was detected (${failures.size} cases). " +
                "Different schemes approximate one solution and must converge to one limit:\n" +
                failures.joinToString("\n").take(6000),
        )
    }

    /**
     * Fredholm of the second kind, n = 64: base, sloan, kulkarni, nystrom, combinedNystrom.
     *
     * The grid is the largest of those used in the article; for the Fredholm equation all
     * the schemes are still computed on it in a reasonable time (the images of the basis splines
     * are precomputed at fixed quadrature nodes).
     */
    @Test
    fun fredholmSchemesAgreeAtFinestGrid() {
        val failures = mutableListOf<String>()
        val problem = problems.fredholm.FredholmProblem.F2
        val n = 64
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val points = comparisonPoints(grid)
            val exact = { t: Double -> problem.exact(t) }
            for (funcs in families(basis)) {
                val op = solvers.fredholm.FredholmOperator(
                    problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                )
                val solver = FredholmSecondKindSolver(
                    basis, funcs, op, 1.0,
                    RhsWithDerivatives(
                        { t -> problem.rhsExact(t, op) },
                        { t -> problem.rhsExactDeriv(t, op) },
                        { t -> problem.rhsExactDeriv2(t, op) },
                    ),
                )
                val solutions = listOf(
                    "base" to solver.base(),
                    "sloan" to solver.sloan(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                    "combinedNystrom" to solver.combinedNystrom(),
                ).map { (name, solution) ->
                    NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
                }
                checkPairwise(
                    "Fredholm ${problem.name}, basis ${system.name}, " +
                        "family ${funcs.name}, n=$n",
                    solutions, points, failures,
                )
            }
        }
        reportFailures(failures, "Fredholm of the second kind")
    }

    /**
     * Volterra of the second kind, n = 32.
     *
     * The grid is twice as coarse as for Fredholm: the Volterra operator has an integration
     * domain depending on `t`, so precomputation at fixed nodes
     * is impossible, and the Nyström weights `W_j(t)` have to be recomputed for every point.
     * For the same reason the Nyström tables for Volterra in the article are limited to n <= 32.
     */
    @Test
    fun volterraSchemesAgreeAtFinestGrid() {
        val failures = mutableListOf<String>()
        val problem = problems.volterra.VolterraProblem.V2
        val n = 32
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val points = comparisonPoints(grid)
            val exact = { t: Double -> problem.exact(t) }
            for (funcs in families(basis)) {
                val op = solvers.volterra.VolterraOperator(
                    problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                )
                val solver = VolterraSecondKindSolver(
                    basis, funcs, op, 1.0,
                    RhsWithDerivatives(
                        { t -> problem.rhsExact(t, op) },
                        { t -> problem.rhsExactDeriv(t, op) },
                        { t -> problem.rhsExactDeriv2(t, op) },
                    ),
                )
                val solutions = listOf(
                    "base" to solver.base(),
                    "sloan" to solver.sloan(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                    "combinedNystrom" to solver.combinedNystrom(),
                ).map { (name, solution) ->
                    NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
                }
                checkPairwise(
                    "Volterra ${problem.name}, basis ${system.name}, " +
                        "family ${funcs.name}, n=$n",
                    solutions, points, failures,
                )
            }
        }
        reportFailures(failures, "Volterra of the second kind")
    }

    /**
     * The consistency on a problem whose solution lies in the span of the generating system.
     *
     * Here the errors of the schemes are close to machine accuracy, and the check degenerates into
     * the requirement "all the schemes give practically one and the same function". The case is valuable
     * in that the comparison threshold is determined by the absolute floor and not by the own
     * errors of the schemes: an inconsistency of any of them is visible immediately.
     */
    @Test
    fun schemesAgreeOnSpanProblem() {
        val failures = mutableListOf<String>()
        val problem = problems.fredholm.FredholmProblem.F2span
        val n = 16
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val points = comparisonPoints(grid)
        val exact = { t: Double -> problem.exact(t) }
        val funcs = ProjFunctionals(basis)
        val op = solvers.fredholm.FredholmOperator(
            problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
        )
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
        )
        // The nystrom scheme is excluded on purpose: its approximation lies OUTSIDE the spline
        // space, so on a span problem it is not obliged to give machine
        // accuracy (see the KDoc of FredholmSecondKindSolver.nystrom) and its error ~5e-5
        // determines the comparison threshold, devaluing the check.
        val solutions = listOf(
            "base" to solver.base(),
            "sloan" to solver.sloan(),
            "kulkarni" to solver.kulkarni(),
            "iteratedKulkarni" to solver.iteratedKulkarni(),
        ).map { (name, solution) ->
            NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
        }
        checkPairwise(
            "Fredholm ${problem.name} (the solution is in the span), basis B, family theta, n=$n",
            solutions, points, failures,
        )
        reportFailures(failures, "A problem with the solution in the span of the generating system")
    }
}
