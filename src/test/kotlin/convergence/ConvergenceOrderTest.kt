package convergence

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import solvers.core.SolutionFunc
import splines.functionals.AveragingFunctionals
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import numerics.orders
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.volterra.VolterraProblem
import solvers.fredholm.FredholmOperator
import solvers.volterra.VolterraOperator
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * THE CONVERGENCE ORDER as a checkable property (the means "Po" in `docs/REFERENCES.md`).
 *
 * WHY THIS TEST EXISTS when there is already `baseline-eh.tsv` with 1366 values.
 * The characterization baseline records the NUMBERS on the grids n = 8 and 16. A regression that
 * cuts the order from 4 to 3 but barely changes the values on coarse grids passes
 * it silently — and that is exactly how the `kulkarniQuasi` defect (a piecewise linear reconstruction
 * of the iterate, cutting the order from ~3.8 to ~1.9) lived unnoticed. Here what is checked
 * is not the value but the RATE of its decrease: `p = log2(E_n / E_2n)`.
 *
 * WHAT IS MEASURED. For every combination "equation x scheme x generating system x
 * functional family" a sequence of errors `E_h` is built on the grids
 * [GRID_SIZES] and from it the empirical orders. The expected value is given by the EXPLICIT
 * TABLE [expected] with the tolerance [ORDER_TOLERANCE]; the table was shot by an actual
 * run and reconciled with `docs/REFERENCES.md` (see below).
 *
 * SUITES AND TAGS. The full matrix (168 combinations up to n = 64) has the `slow` tag and a separate
 * task `./gradlew convergenceOrderTest` (measurements of two runs: 91 s and 105 s wall-clock,
 * that is 1.5-2 min — a range, not an exact number). The reduced subset
 * (the basis B, the families `theta`/`mu`, grids up to n = 32) and the check of exactness on span have the tag
 * `fast` (measurement: 3.0-3.5 s and 1.8-2.2 s), they are run on every edit.
 *
 * RECONCILIATION WITH `docs/REFERENCES.md` (section 3, "A remark on the convergence orders").
 * Discrepancies above 0.5 between the fact and the documentation were NOT FOUND:
 *  * the classical Nyström "does not raise the order", the documentation gives ~4.2 for
 *    Fredholm with the basis B and the family theta — measured 4.21;
 *  * the combined Nyström (Fredholm) — the published estimate `O(h^7)`,
 *    ~7.0 in the documentation — measured 7.04; the iterated variant `O(h^8)` — measured 8.13;
 *  * the combined Nyström (Volterra) — "no gain is observed, about 3.8" —
 *    measured 3.90 (a discrepancy of 0.10, within the tolerance);
 *  * the Sloan iteration (Fredholm) — "the observed 4.2...4.3" in the documentation — measured 4.26;
 *  * the base collocation — the theoretical 3 — measured 3.0...3.1.
 * For the Kulkarni scheme on quasi-interpolants (`mu`, `lambda`) there is NO published order
 * (in `REFERENCES.md` it is marked "without a source, a numerical observation"), so
 * the table here records exactly the observation and does not confirm a theory.
 *
 * ### RECONCILIATION WITH THE PUBLISHED TABLES (an external source, not our own run)
 *
 * The reconciliation is done against `src/test/resources/verification/published-values.tsv` (the keys `*.ph`,
 * extracted by a script from the .tex tables of the article). The outcome: of the 56 rows of the table [expected]
 * 24 are CONFIRMED by the external source; there are NO discrepancies above 0.5 (the worst is 0.44).
 *
 * HOW THE COMPARISON WAS DONE. A published `p_h` with the key `nN` is the order on a CONCRETE
 * pair `N -> 2N` and not an asymptotic value, so the finest pair was taken whose BOTH errors
 * lie above the trust threshold OF THE ROW. Without this condition the comparison is meaningless,
 * and this is NOT fitting: in the publication itself `F.F2.B.theta.n32.iterKulkarni.ph = 0` at
 * `Eh(n32) = Eh(n64) = 4.441e-16` — that is, the authors also ran into machine zero, and their
 * number 0 describes the arithmetic and not the scheme. The same `NOISE_FLOOR` filter is applied
 * by `verification.PublishedValuesTest.checkOrders` too.
 *
 * ROWS CONFIRMED BY THE PUBLICATION (the expectation of the test → published, deviation):
 *  * `F.theta.base` 3.00→3.02 (0.02); `F.mu.base` 2.95→2.94 (0.01);
 *    `F.lambda.base` 3.10→3.07 (0.03); `F.xi1.base` 3.00→2.96 (0.04);
 *  * `F.theta.sloan` 4.30→4.26 (0.04); `F.xi1.sloan` 3.00→2.99 (0.01);
 *  * `F.theta.kulkarni` 7.35→7.35 (0.00); `F.xi1.kulkarni` 5.90→5.88 (0.02);
 *  * `F.xi1.iterKulkarni` 5.95→5.94 (0.01);
 *    **`F.theta.iterKulkarni` 8.60→8.55 (0.05)** — it is exactly this key that proves that the row
 *    is measurable and was previously marked `Saturates` by mistake;
 *  * `F.theta.nystrom` 4.20→4.21 (0.01); `F.theta.iterNystrom` 4.20→4.20 (0.00);
 *  * `V.theta.base` 3.00→3.01; `V.mu.base` 2.95→2.94; `V.lambda.base` 3.00→3.03;
 *    `V.xi1.base` 3.00→2.96;
 *  * `V.theta.sloan` 3.90→3.94; `V.xi1.sloan` 3.00→2.99;
 *  * `V.theta.kulkarni` 3.90→3.92; `V.xi1.kulkarni` 4.00→4.00;
 *  * `V.theta.iterKulkarni` 4.15→4.13; `V.xi1.iterKulkarni` 3.75→3.75;
 *  * `V.theta.nystrom` 3.90→3.84 (0.06);
 *    `V.theta.iterNystrom` 4.30→4.74 (0.44) — the WORST discrepancy. The cause is measured
 *    and not assumed: for this scheme the order oscillates noticeably from level to level
 *    (our measurement over three pairs: 4.27, 4.74, 4.26 — exactly as in the publication),
 *    and the published 4.74 is that very outlier on the pair 16->32. The scheme is reproduced
 *    exactly, the discrepancy is a property of picking one number per row and not a defect.
 *
 * ROWS WITHOUT EXTERNAL CONFIRMATION (32 of 56) — taken ONLY from an actual
 * run and fixing the current behaviour rather than confirming that it is right:
 *  * all 8 rows `combNystrom`/`iterCombNystrom` (these schemes are absent from the tables of the article);
 *  * all the schemes of the families `mu`/`lambda` except `base`: the publication gives for them only
 *    the base collocation (`table-families.tex`);
 *  * `F.mu.iterKulkarni` — the second former `Saturates` row: there is no direct key,
 *    the expectation 8.6 is taken from a measurement (8.23...8.84) and agrees with the confirmed 8.55 of `theta`.
 *
 * THE BOUNDS OF THE COVERAGE (deliberate, not by an oversight):
 *  * the problems — one per equation: `F2` (`K = 1/(1+t+s)`, `u* = 1/(t+1)`) and `V2`.
 *    Both are chosen because their solution lies in NO generating system, that
 *    is, the order is measurable for all three bases in one and the same way. The problem
 *    `V2win` (observed orders ~3/4/5/6) is not in the matrix: it is substantial
 *    in itself, but its kernel `K = t - s` is degenerate on the diagonal, and the orders on it
 *    are different — that is a separate table, not a row of this one;
 *  * `F2span`/`V2span` are excluded from the order measurement BY CONSTRUCTION: their solution
 *    `u* = t^2` lies in the span of the system B, and the error there is pure rounding noise
 *    (1e-14...1.6e-13), independent of the grid step. An "order" computed from it is
 *    the log2 of a ratio of two random quantities: it is NOT NaN but an arbitrary number of any sign
 *    (−2.1 was observed at E = 1.9e-14 and 8.2e-14) — and that is exactly why it is meaningless. NaN
 *    is returned by `orders` only at an exact zero. For these problems a separate check applies,
 *    [spanProblemsAreReproducedExactly] — "the error is below [SPAN_EXACTNESS_TOLERANCE]";
 *  * the family `xi1` ([DeBoorFixFunctionals]) does NOT combine with the four Nyström schemes:
 *    `nystromSupport()` starts with `require(!funcs.usesDerivative)`, and `xi1` is
 *    the only family of the matrix with `usesDerivative = true`. These 12 combinations per
 *    equation are excluded not by choice but by the contract of the code.
 */
class ConvergenceOrderTest {

    private companion object {

        /** The tolerance of the comparison of the observed order with the tabulated one. */
        const val ORDER_TOLERANCE = 0.4

        /**
         * The tolerance for NON-LAST grid pairs (the preasymptotic regime).
         *
         * ALL suitable pairs are checked, not only the last one, otherwise a degradation on coarse
         * grids passes silently. But on coarse grids the scheme has NOT yet reached the asymptotics,
         * and the same tolerance 0.4 would give false positives on healthy code.
         *
         * THE VALUE IS CHOSEN BY MEASUREMENT and not assigned. Over 303 early pairs of healthy
         * code the worst shortfall of the order is −0.578 (`V/H/mu/nystrom`, the pair 8->16: 3.17 against
         * the asymptotic 3.75); worse than −0.5 there are three cases, worse than −0.6 none. The threshold 0.8
         * gives a margin of 0.22 over the worst fact and is at the same time four times smaller than the shortfall (4...7)
         * produced by the target regression `kulkarniQuasi`, that is, it catches it with a large margin.
         */
        const val PREASYMPTOTIC_TOLERANCE = 0.8

        /**
         * The machine accuracy threshold: as soon as the error goes below it, the refinement
         * STOPS. Beyond that it is not the error of the method that decreases but the rounding noise, and the order
         * computed from such a pair characterizes the arithmetic and not the scheme.
         */
        const val MACHINE_PRECISION_FLOOR = 1e-13

        /**
         * The lower bound of TRUST in an error when computing the order.
         *
         * The threshold is above [MACHINE_PRECISION_FLOOR] on purpose. The rounding "plateau" on the problems
         * of the matrix lies in the range 9.5e-14...2.2e-13: having got into it, the error stops
         * decreasing under refinement AT ALL (rows of the form `1.5510e-13 1.5521e-13
         * 1.5521e-13` were observed). A value from this plateau is formally above the stopping threshold, but consists
         * of rounding noise, and an order computed AGAINST it is understated by units
         * (6.34 -> 0.82 was observed for one and the same scheme). Therefore a pair `(E_n, E_2n)`
         * is considered suitable only if BOTH errors are not below this threshold.
         *
         * WHY EXACTLY 2.7e-13 and not a round number. The threshold is chosen BY MEASUREMENT, in the
         * widest gap of the histogram: the largest value from the noise plateau is 2.2171e-13,
         * the smallest substantial value is 3.3329e-13, and 2.7e-13 is about 20 % away from
         * both. Any "round" 2e-13 or 3e-13 would press against one of the
         * bounds and make the classification fragile.
         */
        const val TRUSTED_ERROR_FLOOR = 2.7e-13

        /**
         * The upper bound of the error on the coarsest grid for the combinations marked
         * [Expected.Saturates]: for them already the second refinement level goes below
         * [TRUSTED_ERROR_FLOOR], and there is nothing to measure the order with. The observed maximum over such
         * combinations is 1.1e-11, the threshold gives a margin of an order of magnitude. What it is needed for: on a
         * degradation of a scheme (for example, a return of the piecewise linear reconstruction in
         * `kulkarniQuasi`) the error at n = 8 jumps by 6-8 orders, and the check
         * fires even where the order is undefined.
         */
        const val SATURATED_MAX_COARSE_ERROR = 1e-10

        /**
         * The threshold for the problems whose solution lies in the span of the generating system.
         *
         * The original requirement is "below 1e-10", but a literal 1e-10 here is AN ALMOST EMPTY CHECK:
         * the actual maximum over all 64 combinations equals 1.5521e-13
         * (`F/lambda/kulkarni/n=16`), that is, the margin was 645-fold. A degradation of the accuracy on span
         * by hundreds of times would pass silently.
         *
         * The threshold 5e-13 is three actual maxima (3.2x). The margin is needed not "just in case":
         * the quantity itself is the accumulated rounding error in the solution of the linear system, and it naturally
         * drifts between backends and JDK versions. At the same time 5e-13 is 200 times stricter than the former
         * threshold and fires long before the error becomes substantial.
         */
        const val SPAN_EXACTNESS_TOLERANCE = 5e-13

        /**
         * The limit on the length of the list of failures in the error message.
         *
         * The truncation itself is unavoidable: on a complete degradation the message grows to tens of
         * kilobytes and loses readability. What matters is something else — that the fact of the truncation be MARKED:
         * see [reportIfAny].
         */
        const val FAILURE_REPORT_LIMIT = 8000

        /** The order of the Gauss quadrature — the same as in the characterization baselines. */
        const val QUADRATURE_ORDER = 8

        /** The sequence of grids of the full matrix. */
        val GRID_SIZES = listOf(8, 16, 32, 64)

        /**
         * The sequence of grids of the fast subset: n = 64 costs about 6 s per
         * combination for Volterra and does not fit into the budget of `fastTest`.
         *
         * THE GRID n = 32 IS MANDATORY HERE, although it is twice as expensive as the pair 8/16. Measurement: on the pair
         * 8->16 the scheme `combNystrom` with the family `mu` gives the order 6.24 against the asymptotic
         * 6.65 — it has not yet reached the asymptotics, and even a one-sided comparison with the tolerance
         * 0.4 would fail on healthy code. On the pair 16->32 the same scheme gives 6.66.
         */
        val FAST_GRID_SIZES = listOf(8, 16, 32)

        /** The grids of the exactness check on span: two are enough, the error there does not depend on n. */
        val SPAN_GRID_SIZES = listOf(8, 16)

        const val FREDHOLM = "F"
        const val VOLTERRA = "V"

        val SYSTEMS = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val FAMILIES = listOf("theta", "xi1", "mu", "lambda")

        /** The schemes building an approximation INSIDE the spline space. */
        val SPLINE_SPACE_SCHEMES = listOf("base", "sloan", "kulkarni", "iterKulkarni")

        /**
         * The fast subset: the basis B and two families.
         *
         * Why not a single `theta`, as was proposed in the assignment. `theta` is a PROJECTOR, and
         * `kulkarni()` goes for it into the branch `kulkarniProjector` (a direct linear solve).
         * The branch `kulkarniQuasi` — the very one for the regression in which the acceptance
         * criterion was formulated — is called ONLY for quasi-interpolants. A subset of
         * a single `theta` would not react to it at all. Adding `mu` costs ~2 s and
         * closes this hole.
         */
        val FAST_FAMILIES = listOf("theta", "mu")
    }

    /** The expectation for one combination: either a numerical order, or a saturation at machine accuracy. */
    private sealed interface Expected {

        /**
         * The trust threshold FOR THIS ROW, and not a common one for the whole table.
         *
         * Why the threshold CANNOT be a single one. The schemes of the matrix differ in the absolute
         * level of the error by nine orders: `base` gives 1e-4, while `iterKulkarni`
         * on Fredholm already gives 3e-12 on the COARSEST grid. The common threshold 2.7e-13,
         * chosen by the noise plateau of most schemes, LAY ABOVE the whole convergence
         * range of the two most accurate rows — and their measurable order 8.6 was simply
         * discarded as "noise". A per-row threshold removes this blind spot.
         */
        val trustFloor: Double

        /** The expected empirical order; the comparison is with the tolerance [ORDER_TOLERANCE]. */
        data class Order(
            val p: Double,
            override val trustFloor: Double = TRUSTED_ERROR_FLOOR,
        ) : Expected

        /**
         * The combination saturates at machine accuracy before a suitable pair of errors
         * accumulates: the order is NOT MEASURABLE. The magnitude of the error on a coarse grid is checked.
         *
         * AFTER THE FIX OF THE PER-ROW THRESHOLD this variant is NOT USED by any row
         * of the table: both former `Saturates` rows turned out to be MEASURABLE (see the comment
         * on `F.theta.iterKulkarni`). The type is kept on purpose: a scheme that reaches machine
         * zero already on the first grid is physically possible, and then there will indeed be
         * nothing to check the order with.
         */
        data class Saturates(
            override val trustFloor: Double = TRUSTED_ERROR_FLOOR,
        ) : Expected
    }

    /**
     * THE EXPLICIT TABLE OF EXPECTED ORDERS. The key is `equation.family.scheme`.
     *
     * WHY THE KEY DOES NOT CONTAIN THE GENERATING SYSTEM. This is not a simplification but a MEASURED FACT:
     * over all 56 rows the spread of the observed order between the systems B, H and T does not
     * exceed 0.31 (typically 0.02...0.10), that is, the order is determined by the scheme and the
     * functional family, and not by the choice of `{1,t,t^2}` / `{1,sinh,cosh}` / `{1,sin,cos}`.
     * A single row for three systems makes this statement CHECKABLE: if some
     * edit makes one of the systems worse than the others, the row will fail.
     *
     * The values are the rounded middle of the spread observed over the three systems
     * (an actual run, the multik backend, the grids 8/16/32/64). The worst margin to the bound
     * of the tolerance is 0.31 for `V.theta.iterNystrom` (3.99...4.55 was observed: for this scheme on
     * Volterra the order oscillates noticeably from level to level).
     */
    private val expected: Map<String, Expected> = mapOf(
        // --- Fredholm, problem F2 -------------------------------------------------
        "F.theta.base" to Expected.Order(3.0),
        "F.theta.sloan" to Expected.Order(4.3),
        "F.theta.kulkarni" to Expected.Order(7.35),
        // A ROW WITH A LOWERED TRUST THRESHOLD. There used to be a `Saturates` here, and that was
        // a MISTAKE: the order here IS MEASURABLE (E_8 = 2.99e-12, E_16 = 7.77e-15 → p = 8.59), it is just that both
        // errors lay below the common threshold 2.7e-13 and were discarded as "noise".
        // The substitution by "E_8 < 1e-10" was 33 times weaker than the fact: a 30-fold degradation would pass silently.
        // The threshold 1e-15: above the absolute machine zero of the problem (~1e-16), but below the smallest
        // substantial error of the row (7.55e-15 over the three systems) with a 7-fold margin.
        // THE EXPECTATION 8.6 IS CONFIRMED BY THE PUBLICATION: `F.F2.B.theta.n8.iterKulkarni.ph` = 8.55
        // (`published-values.tsv`, `table-t2-fredholm.tex`); measured 8.44...8.79 over the three systems.
        "F.theta.iterKulkarni" to Expected.Order(8.6, trustFloor = 1e-15),
        "F.theta.nystrom" to Expected.Order(4.2),
        "F.theta.iterNystrom" to Expected.Order(4.2),
        "F.theta.combNystrom" to Expected.Order(7.0),
        "F.theta.iterCombNystrom" to Expected.Order(8.1),
        "F.xi1.base" to Expected.Order(3.0),
        "F.xi1.sloan" to Expected.Order(3.0),
        "F.xi1.kulkarni" to Expected.Order(5.9),
        "F.xi1.iterKulkarni" to Expected.Order(5.95),
        "F.mu.base" to Expected.Order(2.95),
        "F.mu.sloan" to Expected.Order(3.9),
        "F.mu.kulkarni" to Expected.Order(6.3),
        // The same as for `F.theta.iterKulkarni`: it was `Saturates`, in fact the order is measurable
        // (E_8 = 8.21e-12, E_16 = 1.99e-14 → p = 8.69; over the three systems 8.23...8.84). It is not fixed by the
        // publication DIRECTLY (the tables have no `mu` with `iterKulkarni`), but it agrees with the 8.55
        // of `theta`: a quasi-interpolant has the same superconvergence mechanism.
        "F.mu.iterKulkarni" to Expected.Order(8.6, trustFloor = 1e-15),
        "F.mu.nystrom" to Expected.Order(3.9),
        "F.mu.iterNystrom" to Expected.Order(3.9),
        "F.mu.combNystrom" to Expected.Order(6.65),
        "F.mu.iterCombNystrom" to Expected.Order(7.0),
        "F.lambda.base" to Expected.Order(3.1),
        "F.lambda.sloan" to Expected.Order(3.9),
        "F.lambda.kulkarni" to Expected.Order(6.6),
        "F.lambda.iterKulkarni" to Expected.Order(7.1),
        "F.lambda.nystrom" to Expected.Order(3.9),
        "F.lambda.iterNystrom" to Expected.Order(3.9),
        "F.lambda.combNystrom" to Expected.Order(6.85),
        "F.lambda.iterCombNystrom" to Expected.Order(7.25),
        // --- Volterra, problem V2 -------------------------------------------------
        "V.theta.base" to Expected.Order(3.0),
        "V.theta.sloan" to Expected.Order(3.9),
        "V.theta.kulkarni" to Expected.Order(3.9),
        "V.theta.iterKulkarni" to Expected.Order(4.15),
        "V.theta.nystrom" to Expected.Order(3.9),
        "V.theta.iterNystrom" to Expected.Order(4.3),
        "V.theta.combNystrom" to Expected.Order(3.9),
        "V.theta.iterCombNystrom" to Expected.Order(4.15),
        "V.xi1.base" to Expected.Order(3.0),
        "V.xi1.sloan" to Expected.Order(3.0),
        "V.xi1.kulkarni" to Expected.Order(4.0),
        "V.xi1.iterKulkarni" to Expected.Order(3.75),
        "V.mu.base" to Expected.Order(2.95),
        "V.mu.sloan" to Expected.Order(3.8),
        "V.mu.kulkarni" to Expected.Order(3.8),
        "V.mu.iterKulkarni" to Expected.Order(3.9),
        "V.mu.nystrom" to Expected.Order(3.75),
        "V.mu.iterNystrom" to Expected.Order(3.9),
        "V.mu.combNystrom" to Expected.Order(3.75),
        "V.mu.iterCombNystrom" to Expected.Order(3.9),
        "V.lambda.base" to Expected.Order(3.0),
        "V.lambda.sloan" to Expected.Order(3.9),
        "V.lambda.kulkarni" to Expected.Order(3.9),
        "V.lambda.iterKulkarni" to Expected.Order(3.9),
        "V.lambda.nystrom" to Expected.Order(3.9),
        "V.lambda.iterNystrom" to Expected.Order(3.9),
        "V.lambda.combNystrom" to Expected.Order(3.9),
        "V.lambda.iterCombNystrom" to Expected.Order(3.9),
    )

    // ------------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------------

    /**
     * THE FULL MATRIX: 2 equations x 3 systems x 4 families x (4 + 4) schemes minus
     * the forbidden combinations `xi1` x Nyström — 168 combinations, the grids 8/16/32/64.
     *
     * The comparison is TWO-SIDED: the order must not only not drop, but also not "improve"
     * inexplicably — a growth of the order by 0.5 means that the scheme computes something other than before,
     * and this is an equally weighty reason to investigate.
     *
     * The actual run time of THIS METHOD ITSELF is 79...99 s over two measurements
     * (the rest of the task time is two `fast` methods and the JVM start), hence the `slow` tag;
     * a separate task `./gradlew convergenceOrderTest` makes the suite runnable without
     * waiting for the whole `slowTest` to be repaired.
     */
    @Test
    @Tag("slow")
    fun convergenceOrdersMatchExpectedTable() {
        val failures = mutableListOf<String>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (system in SYSTEMS) {
                for (familyName in FAMILIES) {
                    val errorsByScheme = collectErrors(equation, system, familyName, GRID_SIZES)
                    for ((scheme, errs) in errorsByScheme) {
                        checked++
                        checkCombination(
                            key = "$equation.$familyName.$scheme",
                            label = "$equation/${system.name}/$familyName/$scheme",
                            gridSizes = GRID_SIZES,
                            errs = errs,
                            oneSided = false,
                            failures = failures,
                        )
                    }
                }
            }
        }
        assertTrue(
            checked == 168,
            "The matrix must contain 168 combinations (2 equations x 3 systems x " +
                "(4 families x 4 schemes + 3 families without a derivative x 4 Nyström schemes)), " +
                "checked $checked",
        )
        reportIfAny(failures, checked, "The convergence order does not match the table")
    }

    /**
     * THE FAST SUBSET: the basis B, the families `theta` and `mu`, the grids 8/16/32 — 32 combinations
     * (2 equations x 2 families x 8 schemes; both families are without a derivative, so all eight
     * schemes are available), measured 3.0-3.5 s. Run on every edit as part of `fastTest`.
     *
     * The comparison is ONE-SIDED — `p >= expected - tolerance`. This is not a relaxation but a
     * consequence of the fact that on grids up to n = 32 the scheme has not always reached the asymptotics yet:
     * for example, for `V/B/theta/iterNystrom` the order on the pair 16->32 equals 4.74 against
     * the asymptotic 4.3, and a two-sided comparison would fail on healthy code.
     * What has to be caught is a DEGRADATION, and any degradation only LOWERS the order. The exact
     * match with the table is checked by [convergenceOrdersMatchExpectedTable].
     */
    @Test
    @Tag("fast")
    fun convergenceOrdersDoNotDegradeOnFastSubset() {
        val failures = mutableListOf<String>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (familyName in FAST_FAMILIES) {
                val errorsByScheme =
                    collectErrors(equation, GeneratingSystem.B, familyName, FAST_GRID_SIZES)
                for ((scheme, errs) in errorsByScheme) {
                    checked++
                    checkCombination(
                        key = "$equation.$familyName.$scheme",
                        label = "$equation/B/$familyName/$scheme",
                        gridSizes = FAST_GRID_SIZES,
                        errs = errs,
                        oneSided = true,
                        failures = failures,
                    )
                }
            }
        }
        assertTrue(
            checked == 32,
            "The fast subset must contain 32 combinations (2 equations x 2 families x 8 schemes), " +
                "checked $checked",
        )
        reportIfAny(failures, checked, "The convergence order degraded relative to the table")
    }

    /**
     * The problems `F2span`/`V2span`: the solution `u* = t^2` lies in the span of the polynomial system B,
     * so the schemes building an approximation INSIDE the spline space must
     * reproduce it with an error below [SPAN_EXACTNESS_TOLERANCE].
     *
     * WHY THE ORDER IS NOT MEASURED. The error here is the rounding noise of the linear solve
     * (1e-14...1.6e-13), it does not depend on the grid step. `orders` will then return NOT NaN
     * but the log2 of a ratio of two noises — any number (−2.1, say); NaN would occur only at
     * an exact zero. Comparing such a number with the table of expectations is impossible, so the
     * MAGNITUDE of the error is checked.
     *
     * THE LIMITATIONS OF THE CHECK, both being properties of the method and not defects:
     *  * only the system B: `u* = t^2` does not lie in the span of `{1, sinh, cosh}` and `{1, sin, cos}`,
     *    and on the systems H/T the error naturally equals the usual ~4e-5;
     *  * only the schemes [SPLINE_SPACE_SCHEMES]: the Nyström approximation lies OUTSIDE the spline
     *    space (`u^N_h = f + L^N_h u`), it does not inherit the exactness on span — in fact
     *    ~5e-5 is observed, which agrees with its own order 4.
     */
    @Test
    @Tag("fast")
    fun spanProblemsAreReproducedExactly() {
        val failures = mutableListOf<String>()
        val spanErrors = LinkedHashMap<String, MutableList<Double>>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (familyName in FAMILIES) {
                for (n in SPAN_GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val schemes: List<Pair<String, () -> SolutionFunc>>
                    val exact: (Double) -> Double
                    if (equation == FREDHOLM) {
                        val problem = FredholmProblem.F2span
                        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                        val solver = problems.fredholm.secondKindSolver(problem, basis, funcs, op)
                        schemes = fredholmSchemes(solver, funcs)
                        exact = { t -> problem.exact(t) }
                    } else {
                        val problem = VolterraProblem.V2span
                        val op = VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                        val solver = problems.volterra.secondKindSolver(problem, basis, funcs, op)
                        schemes = volterraSchemes(solver, funcs)
                        exact = { t -> problem.exact(t) }
                    }
                    for ((scheme, build) in schemes) {
                        if (scheme !in SPLINE_SPACE_SCHEMES) continue
                        checked++
                        val error = errorEh(exact, build().eval, grid)
                        spanErrors.getOrPut("$equation/$familyName/$scheme") { mutableListOf() } += error
                    }
                }
                // The message is assembled AFTER walking all the grids of the family: the full
                // sequence of errors is needed in ALL failure branches,
                // and not only where the order is computed.
                for (scheme in SPLINE_SPACE_SCHEMES) {
                    val series = spanErrors["$equation/$familyName/$scheme"] ?: continue
                    val worst = series.max()
                    if (!(worst < SPAN_EXACTNESS_TOLERANCE)) {
                        val seq = SPAN_GRID_SIZES.indices.joinToString(", ") {
                            "n=${SPAN_GRID_SIZES[it]}: ${fmtError(series[it])}"
                        }
                        failures += "$equation/span/B/$familyName/$scheme: " +
                            "the largest E_h=${fmtError(worst)} must be below " +
                            "${fmtError(SPAN_EXACTNESS_TOLERANCE)} " +
                            "(u* = t^2 lies in the span of the generating system B). E_h [$seq]"
                    }
                }
            }
        }
        assertTrue(
            checked == 64,
            "The check on span must cover 64 combinations " +
                "(2 equations x 4 families x 2 grids x 4 schemes), checked $checked",
        )
        reportIfAny(failures, checked, "The exactness on the span of the generating system is not attained")
    }

    // ------------------------------------------------------------------------
    // Measurement
    // ------------------------------------------------------------------------

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "mu" -> AveragingFunctionals(basis, 0.5)
        "lambda" -> ThreePointFunctionals(basis, 0.5)
        else -> error("Unknown functional family: $name")
    }

    /**
     * The schemes of the Fredholm solver. The Nyström schemes are discarded for families with a derivative:
     * this is the contract of the code (`require(!funcs.usesDerivative)`), not a choice of the test.
     */
    private fun fredholmSchemes(
        solver: solvers.fredholm.FredholmSecondKindSolver,
        funcs: FunctionalFamily,
    ): List<Pair<String, () -> SolutionFunc>> {
        val core = listOf<Pair<String, () -> SolutionFunc>>(
            "base" to { solver.base() },
            "sloan" to { solver.sloan() },
            "kulkarni" to { solver.kulkarni() },
            "iterKulkarni" to { solver.iteratedKulkarni() },
        )
        if (funcs.usesDerivative) return core
        return core + listOf<Pair<String, () -> SolutionFunc>>(
            "nystrom" to { solver.nystrom() },
            "iterNystrom" to { solver.iteratedNystrom() },
            "combNystrom" to { solver.combinedNystrom() },
            "iterCombNystrom" to { solver.iteratedCombinedNystrom() },
        )
    }

    /** The schemes of the Volterra solver; the same restriction on families with a derivative. */
    private fun volterraSchemes(
        solver: solvers.volterra.VolterraSecondKindSolver,
        funcs: FunctionalFamily,
    ): List<Pair<String, () -> SolutionFunc>> {
        val core = listOf<Pair<String, () -> SolutionFunc>>(
            "base" to { solver.base() },
            "sloan" to { solver.sloan() },
            "kulkarni" to { solver.kulkarni() },
            "iterKulkarni" to { solver.iteratedKulkarni() },
        )
        if (funcs.usesDerivative) return core
        return core + listOf<Pair<String, () -> SolutionFunc>>(
            "nystrom" to { solver.nystrom() },
            "iterNystrom" to { solver.iteratedNystrom() },
            "combNystrom" to { solver.combinedNystrom() },
            "iterCombNystrom" to { solver.iteratedCombinedNystrom() },
        )
    }

    /**
     * The errors `E_h` of all the schemes on a sequence of grids.
     *
     * The solver is built ONCE per grid and reused by all the schemes: the assembly
     * of the matrices `M`/`M2` is the most expensive part, and repeating it per scheme would multiply
     * the run time by about eight.
     *
     * The refinement for a concrete scheme stops as soon as its error
     * has gone below [MACHINE_PRECISION_FLOOR]: beyond that the order is determined by
     * rounding noise and not by the method.
     */
    private fun collectErrors(
        equation: String,
        system: GeneratingSystem,
        familyName: String,
        gridSizes: List<Int>,
    ): Map<String, List<Double>> {
        val errors = LinkedHashMap<String, MutableList<Double>>()
        for (n in gridSizes) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val funcs = family(familyName, basis)
            val schemes: List<Pair<String, () -> SolutionFunc>>
            val exact: (Double) -> Double
            if (equation == FREDHOLM) {
                val problem = FredholmProblem.F2
                val op = FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                schemes = fredholmSchemes(
                    problems.fredholm.secondKindSolver(problem, basis, funcs, op),
                    funcs,
                )
                exact = { t -> problem.exact(t) }
            } else {
                val problem = VolterraProblem.V2
                val op = VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                schemes = volterraSchemes(
                    problems.volterra.secondKindSolver(problem, basis, funcs, op),
                    funcs,
                )
                exact = { t -> problem.exact(t) }
            }
            for ((scheme, build) in schemes) {
                val sequence = errors.getOrPut(scheme) { mutableListOf() }
                if (sequence.isNotEmpty() && sequence.last() < MACHINE_PRECISION_FLOOR) continue
                sequence += errorEh(exact, build().eval, grid)
            }
        }
        return errors
    }

    // ------------------------------------------------------------------------
    // Checking and diagnostics
    // ------------------------------------------------------------------------

    /**
     * ALL suitable pairs `(index, order)` in the order of refinement: those whose both
     * errors are not below [floor] and whose order is not `NaN`.
     *
     * ALL PAIRS ARE RETURNED, not the last one. Previously only the last pair was taken
     * (the closest to the asymptotics), and that left a hole: a preasymptotic degradation
     * — the one that spoils the accuracy on coarse grids but reaches the same slope by n = 64
     * — passed silently, although it is exactly on coarse grids that the library is used.
     */
    private fun trustedOrders(errs: List<Double>, floor: Double): List<Pair<Int, Double>> {
        val ps = orders(errs)
        return (0 until maxOf(errs.size - 1, 0)).mapNotNull { i ->
            if (errs[i] >= floor && errs[i + 1] >= floor && !ps[i].isNaN()) i to ps[i] else null
        }
    }

    /**
     * Cross-checks one combination against the table and, on a discrepancy, adds to [failures] a row
     * with the FULL sequence of errors and all the orders: without them it is impossible to understand from the
     * message whether the scheme broke or the measurement is degenerate.
     */
    private fun checkCombination(
        key: String,
        label: String,
        gridSizes: List<Int>,
        errs: List<Double>,
        oneSided: Boolean,
        failures: MutableList<String>,
    ) {
        val expectation = expected[key] ?: run {
            failures += "$label: the key '$key' is absent from the table of expected orders. " +
                diagnostics(gridSizes, errs, TRUSTED_ERROR_FLOOR)
            return
        }
        val floor = expectation.trustFloor
        val diagnostics = diagnostics(gridSizes, errs, floor)
        val trustedPairs = trustedOrders(errs, floor)
        val trusted = trustedPairs.lastOrNull()
        when (expectation) {
            is Expected.Saturates -> {
                // THE ORDER OF THE CHECKS MATTERS. First the magnitude of the error, then the presence
                // of a measurable order. On a degradation of a scheme BOTH conditions hold at once (the error
                // grew by orders AND a measurable order appeared), and what has to be reported is the
                // degradation, rather than proposing to update the table to fit broken code.
                val coarse = errs.first()
                if (!(coarse < SATURATED_MAX_COARSE_ERROR)) {
                    val orderPart = if (trusted == null) {
                        ""
                    } else {
                        " The error stopped running into machine accuracy: a measurable order " +
                            "${fmtOrder(trusted.second)} appeared — a sign of a DEGRADATION of the scheme."
                    }
                    failures += "$label: the table declares a saturation at machine accuracy, so " +
                        "the error on a coarse grid is checked: ${fmtError(coarse)} must be below " +
                        "${fmtError(SATURATED_MAX_COARSE_ERROR)}.$orderPart $diagnostics"
                    return
                }
                if (trusted != null) {
                    failures += "$label: the table declares a saturation at machine accuracy without a measurable " +
                        "order, but a suitable pair of errors was found (p=${fmtOrder(trusted.second)}) at " +
                        "the former error level. This is a change of behaviour — update the table " +
                        "deliberately. $diagnostics"
                }
            }

            is Expected.Order -> {
                if (trusted == null) {
                    failures += "$label: the table expects the order ${fmtOrder(expectation.p)}, but there is not a single " +
                        "pair of errors above the trust threshold ${fmtError(floor)} — " +
                        "there is nothing to measure the order with. $diagnostics"
                    return
                }
                // ALL SUITABLE PAIRS ARE CHECKED, not only the last one: otherwise a degradation
                // on coarse grids passes silently if the last pair is within the tolerance.
                val (lastIndex, _) = trusted
                for ((index, observed) in trustedPairs) {
                    val isLast = index == lastIndex
                    // THE EARLY PAIRS GO BY A RELAXED RULE, and this is not a relaxation for the sake of a green
                    // test but a property of the method: on the pair 8->16 the scheme has not reached the asymptotics yet.
                    // A measurement over 303 early pairs of HEALTHY code: the worst shortfall is −0.578
                    // (`V/H/mu/nystrom`, the pair 8->16), three cases in total worse than −0.5 and none worse than −0.6.
                    // Hence [PREASYMPTOTIC_TOLERANCE] = 0.8: twice the usual tolerance, with a margin of 0.22
                    // to the worst fact. The `kulkarniQuasi` mutation gives a shortfall of 4...7, that is, it is caught.
                    // A growth of the order on early pairs is NOT checked at all: for `V/*/xi1/iterKulkarni`
                    // it regularly reaches 4.94 against the asymptotic 3.75 — an outlier on a coarse grid,
                    // and not an improvement of the scheme.
                    val tolerance = if (isLast) ORDER_TOLERANCE else PREASYMPTOTIC_TOLERANCE
                    val degraded = observed < expectation.p - tolerance
                    val improved = isLast && observed > expectation.p + ORDER_TOLERANCE
                    if (!degraded && !(improved && !oneSided)) continue
                    val pair = "${gridSizes[index]}->${gridSizes[index + 1]}"
                    val direction = if (degraded) "BELOW" else "ABOVE"
                    val rule = when {
                        !isLast -> "for a preasymptotic pair p >= " +
                            "${fmtOrder(expectation.p - PREASYMPTOTIC_TOLERANCE)} is required"
                        oneSided -> "p >= ${fmtOrder(expectation.p - ORDER_TOLERANCE)} is required"
                        else -> "${fmtOrder(expectation.p)} +- $ORDER_TOLERANCE is required"
                    }
                    failures += "$label: the observed order ${fmtOrder(observed)} (pair n=$pair) is $direction " +
                        "the expected ${fmtOrder(expectation.p)}, $rule. $diagnostics"
                }
            }
        }
    }

    /**
     * The full diagnostics of a combination: the errors over the grids and all the orders between them.
     *
     * @param floor the trust threshold OF THIS row: the mark `[noise]` must coincide with what
     *        was actually discarded during the check, otherwise the diagnostics is misleading.
     */
    private fun diagnostics(gridSizes: List<Int>, errs: List<Double>, floor: Double): String {
        val ps = orders(errs)
        val errorPart = errs.indices.joinToString(", ") { "n=${gridSizes[it]}: ${fmtError(errs[it])}" }
        val orderPart = if (errs.size < 2) {
            "there are no orders (one level measured)"
        } else {
            (0 until errs.size - 1).joinToString(", ") { i ->
                val trusted = errs[i] >= floor && errs[i + 1] >= floor
                val mark = if (trusted) "" else " [noise]"
                "${gridSizes[i]}->${gridSizes[i + 1]}: ${fmtOrder(ps[i])}$mark"
            }
        }
        return "E_h [$errorPart]; orders [$orderPart]"
    }

    /**
     * A single report of the failures.
     *
     * @param subject what exactly did not match. A parameter and not a constant: the span check does NOT
     *        measure the order, and the heading "the order does not match the table" was simply
     *        wrong there — the reader would look for a row in the table `expected` that is not there.
     */
    private fun reportIfAny(failures: List<String>, checked: Int, subject: String) {
        val body = failures.joinToString("\n")
        // The truncation is MARKED EXPLICITLY: a silently truncated report looks like a full one,
        // and the last combinations may simply go unnoticed.
        val shown = if (body.length <= FAILURE_REPORT_LIMIT) {
            body
        } else {
            body.take(FAILURE_REPORT_LIMIT) +
                "\n[OUTPUT TRUNCATED: the first $FAILURE_REPORT_LIMIT of ${body.length} characters are shown; " +
                "the full list is in the XML report build/test-results]"
        }
        assertTrue(failures.isEmpty(), "$subject (${failures.size} of $checked combinations):\n$shown")
    }

    private fun fmtError(x: Double): String = String.format(Locale.ROOT, "%.4e", x)

    private fun fmtOrder(x: Double): String =
        if (x.isNaN()) "---" else String.format(Locale.ROOT, "%.2f", x)
}
