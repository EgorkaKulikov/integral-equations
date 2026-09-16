package verification

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * CROSS-CHECKING THE PROJECT SCHEMES AGAINST THE INDEPENDENT BASELINE [ReferenceNystromSolver].
 *
 * Why. Checks comparing the project schemes with each other do not detect an error
 * common to all the schemes: it cancels out. Here the baseline is an implementation using
 * NOT A SINGLE element of the numerical core of the project (see the KDoc of
 * [ReferenceNystromSolver]): no splines, no functionals, no quadrature, no
 * linear solver. Therefore a discrepancy points at a defect, while an agreement is evidence
 * not closed on the code under test.
 *
 * What exactly is checked, in two steps — otherwise the result cannot be interpreted:
 *
 *  1. THE SUITABILITY OF THE BASELINE. The baseline must reproduce the exact solution of a model
 *     problem at a level orders of magnitude better than the project schemes. If it does not do so,
 *     it cannot be cross-checked against, and step 2 is meaningless ([referenceReproducesExactSolution]).
 *  2. THE CROSS-CHECK OF THE SCHEMES. The project solutions deviate from the baseline by no more than from the exact
 *     solution (with a margin for the baseline itself not being ideal) —
 *     [projectSchemesAgreeWithReferenceNystrom].
 *
 * The `fast` tag: the baseline on 64 nodes is one inversion of a 64x64 matrix, the project schemes are taken
 * at n = 8..32, the whole class fits into fractions of a second.
 */
@Tag("fast")
class ReferenceNystromCrossCheckTest {

    private companion object {
        /** The interval of the problem; coincides with the domain of the model problems. */
        const val A = 0.0
        const val B = 1.0

        /** The grids on which the project schemes are taken. */
        val GRID_SIZES = listOf(8, 16, 32)

        /** The comparison points; 41 points cover the interval more finely than the coarsest grid. */
        const val SAMPLE_SIZE = 41
        val SAMPLE: List<Double> = (0 until SAMPLE_SIZE).map { A + (B - A) * it / (SAMPLE_SIZE - 1.0) }

        /**
         * The tolerance on the suitability of the baseline.
         *
         * The justification of the value: the Gauss-Legendre quadrature on a smooth kernel converges
         * exponentially, so 64 nodes give an accuracy at the machine level. The threshold 1e-12
         * leaves a margin of order 100 ulp for the accumulation of the error of a 64x64 linear solve —
         * and is at the same time six orders stricter than the error of the project schemes (1e-5..1e-8),
         * so the baseline is certainly more accurate than the quantity being measured.
         */
        const val TOL_REFERENCE_FITNESS = 1e-12

        /**
         * The ABSOLUTE thresholds of the deviation from the baseline: the key is `problem/system/scheme/n`.
         *
         * WHY ABSOLUTE and not "with a margin over the deviation from the exact solution".
         * The previous variant compared `versusReference <= 1.5 * versusExact + 1e-12`
         * and was IDENTICALLY TRUE. The proof: by the triangle inequality
         * `versusReference <= versusExact + |baseline - exact|`, and the second term is already
         * bounded by [TOL_REFERENCE_FITNESS] by the test [referenceReproducesExactSolution].
         * Hence `versusReference <= versusExact + 1e-12 <= 1.5 * versusExact + 1e-12`
         * held FOR ANY `versusExact`, including one degraded by thousands of times:
         * the test could not fail in principle and checked only the arithmetic of the formulas themselves.
         *
         * An absolute threshold is free of this: it does not depend on the solution under test.
         * The second variant considered (`|versusReference - versusExact| <= 2 * TOL`)
         * was rejected: it checks the closeness of TWO MEASUREMENTS of one and the same error
         * (that is, the same triangle inequality, only in two directions), and on a
         * degradation of the scheme BOTH measurements grow consistently, so the difference
         * would stay small — the degradation would pass again. Only a requirement
         * external to the solution is meaningful.
         *
         * Where the numbers come from: a doubling of the actually measured `versusReference`
         * (the summary is printed by the test itself) — the same principle as for the table
         * `EH_LIMITS` in `tools/verify_with_scipy.py`. The x2 margin catches a drop of the convergence
         * order on any of the three grids (a transition O(h^3) -> O(h^2) at n = 8 gives
         * an eightfold growth), while there is no spread between runs here: the quantities
         * are deterministic bit for bit.
         *
         * The special case `F2exp/H`: the solution `exp(t)` lies in the span of the system `H`, so
         * only rounding remains (1e-13...1e-11), growing with the size of the linear system. The thresholds
         * there follow the same rule (x2 of the fact) — and it is exactly they that check that the property
         * "the solution is in the span" is not lost.
         */
        val REFERENCE_LIMITS: Map<String, Double> = mapOf(
            "F2/B/base/n=8" to 2.03e-04, "F2/B/base/n=16" to 2.49e-05, "F2/B/base/n=32" to 3.05e-06,
            "F2/B/sloan/n=8" to 9.62e-06, "F2/B/sloan/n=16" to 5.00e-07, "F2/B/sloan/n=32" to 2.51e-08,
            "F2/H/base/n=8" to 1.70e-04, "F2/H/base/n=16" to 2.09e-05, "F2/H/base/n=32" to 2.55e-06,
            "F2/H/sloan/n=8" to 8.62e-06, "F2/H/sloan/n=16" to 4.54e-07, "F2/H/sloan/n=32" to 2.29e-08,
            "F2/T/base/n=8" to 2.37e-04, "F2/T/base/n=16" to 2.90e-05, "F2/T/base/n=32" to 3.55e-06,
            "F2/T/sloan/n=8" to 1.07e-05, "F2/T/sloan/n=16" to 5.45e-07, "F2/T/sloan/n=32" to 2.72e-08,
            "F2exp/B/base/n=8" to 9.82e-05, "F2exp/B/base/n=16" to 1.13e-05, "F2exp/B/base/n=32" to 1.37e-06,
            "F2exp/B/sloan/n=8" to 1.29e-05, "F2exp/B/sloan/n=16" to 6.39e-07, "F2exp/B/sloan/n=32" to 3.50e-08,
            // The solution is in the span of the system H: only rounding remains (see above).
            "F2exp/H/base/n=8" to 1.00e-12, "F2exp/H/base/n=16" to 1.00e-11, "F2exp/H/base/n=32" to 2.40e-11,
            "F2exp/H/sloan/n=8" to 2.20e-13, "F2exp/H/sloan/n=16" to 2.20e-12, "F2exp/H/sloan/n=32" to 3.10e-12,
            "F2exp/T/base/n=8" to 1.97e-04, "F2exp/T/base/n=16" to 2.27e-05, "F2exp/T/base/n=32" to 2.74e-06,
            "F2exp/T/sloan/n=8" to 2.57e-05, "F2exp/T/sloan/n=16" to 1.28e-06, "F2exp/T/sloan/n=32" to 7.00e-08,
        )
    }

    /** The model problems; for each one an independently written out kernel and exact solution. */
    private fun problems() = listOf(FredholmProblem.F2, FredholmProblem.F2exp)

    /**
     * The baseline is built ONLY from the statement of the problem: the kernel, the right-hand side, the interval.
     * The right-hand side is taken exact (`f = u - Ku`, the integral is computed by the baseline itself
     * on its own quadrature), so as not to draw the project operator into the baseline.
     */
    private fun reference(problem: FredholmProblem): ReferenceNystromSolver {
        val kernel = { t: Double, s: Double -> problem.kernel.k(t, s) }
        // f(t) = u(t) - ∫ K(t,s) u(s) ds. The integral is taken by the INDEPENDENT quadrature
        // of the baseline and NOT by the project operator (`problem.rhsExact` requires
        // a `FredholmOperator` and would draw the quadrature under test into the baseline).
        val rhs = { t: Double ->
            problem.exact(t) - ReferenceNystromSolver.integrate(A, B) { s -> kernel(t, s) * problem.exact(s) }
        }
        return ReferenceNystromSolver(kernel, rhs, lambda = 1.0, a = A, b = B, nodeCount = 64)
    }

    /**
     * Step 1: the baseline reproduces the exact solution of the model problems.
     *
     * Without this check step 2 makes no sense: a cross-check against an unfit baseline either
     * misses defects, or reports false ones.
     */
    @Test
    fun referenceReproducesExactSolution() {
        for (problem in problems()) {
            val solver = reference(problem)
            val deviation = SAMPLE.maxOf { t -> abs(solver.eval(t) - problem.exact(t)) }
            assertTrue(
                deviation <= TOL_REFERENCE_FITNESS,
                "The baseline Nystrom does not reproduce the exact solution of the problem ${problem.name}: " +
                    "the deviation $deviation > the tolerance $TOL_REFERENCE_FITNESS. Such a " +
                    "baseline cannot be cross-checked against — find the cause first, do not loosen the tolerance.",
            )
        }
    }

    /**
     * Step 2: the solutions of the project schemes agree with the independent baseline.
     *
     * Both schemes (the base one and the Sloan iteration) are checked on all three generating
     * systems and three grids — that is, the same coverage as the dumped
     * `solution-errors.tsv`, but the comparison is POINTWISE against an independent solution,
     * and not only by the aggregate `E_h`.
     */
    @Test
    fun projectSchemesAgreeWithReferenceNystrom() {
        // A summary of the worst deviations: printed always, and not only on a failure.
        // The reason: a green test without numbers does not let one distinguish "the schemes are accurate" from
        // "the tolerance is too wide", and it is exactly this distinction that is the content of the cross-check.
        val summary = linkedMapOf<String, Double>()
        for (problem in problems()) {
            val referenceSolver = reference(problem)
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n, a = A, b = B)
                    val basis = MinimalSplineBasis(system, grid)
                    val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = FredholmSecondKindSolver(
                        basis, ProjFunctionals(basis), op, 1.0,
                        RhsWithDerivatives(
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
                        ),
                    )
                    for ((scheme, evaluate) in listOf(
                        "base" to solver.base().eval,
                        "sloan" to solver.sloan().eval,
                    )) {
                        val versusReference = SAMPLE.maxOf { t -> abs(evaluate(t) - referenceSolver.eval(t)) }
                        val tag = "${problem.name}/${system.name}/$scheme/n=$n"
                        summary[tag] = versusReference
                        // A non-finite deviation is a separate failure: the comparison `NaN <= limit`
                        // is false, so without the check the message would be about exceeding the threshold,
                        // which leads the diagnosis away from the real cause.
                        assertTrue(
                            versusReference.isFinite(),
                            "$tag: the deviation from the baseline is NON-FINITE ($versusReference): the scheme " +
                                "returned NaN or an infinity",
                        )
                        val limit = REFERENCE_LIMITS[tag]
                        assertTrue(
                            limit != null,
                            "$tag: there is no threshold for this combination in REFERENCE_LIMITS. A new combination " +
                                "must either be cross-checked or break the test — but not pass silently",
                        )
                        assertTrue(
                            versusReference <= limit!!,
                            "$tag: the deviation from the INDEPENDENT baseline $versusReference exceeds " +
                                "the absolute threshold $limit. The baseline does not use the project code (see the KDoc of " +
                                "ReferenceNystromSolver), so the discrepancy points at a defect of the scheme, " +
                                "not of the baseline. Do NOT loosen the threshold — find the cause first.",
                        )
                    }
                }
            }
        }
        // Completeness in the OPPOSITE direction: every threshold must be used. Otherwise
        // a disappearance of a combination from the loop (a narrowing of GRID_SIZES, say) would reduce
        // the volume of the cross-check unnoticed — the test would stay green.
        val unused = REFERENCE_LIMITS.keys - summary.keys
        assertTrue(
            unused.isEmpty(),
            "The REFERENCE_LIMITS thresholds were left UNUSED: ${unused.sorted()}. Hence " +
                "the cross-check stopped covering part of the combinations, and the test would not notice it",
        )
        println("The deviation of the project schemes from the INDEPENDENT Nystrom baseline (max over $SAMPLE_SIZE points):")
        for ((tag, deviation) in summary) {
            println("  %-26s %.3e  (threshold %.3e)".format(tag, deviation, REFERENCE_LIMITS[tag]))
        }
    }
}
