package characterization

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import solvers.core.SolutionFunc
import splines.functionals.AveragingFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.metrics.errorEh
import java.util.Locale

/**
 * THE SINGLE SOURCE of the additional characterization matrix (the snapshot `baseline-extra.tsv`).
 *
 * Why a separate matrix when there is already `baseline-eh.tsv`. The existing baseline covers
 * only the schemes `base`/`sloan`/`kulkarni`/`iteratedKulkarni`/`nystrom`/`iteratedNystrom`
 * on the grid [Grid.uniform] and the interval `[0,1]`. Outside the coverage remain exactly the places
 * that are most at risk when the common code of the Fredholm and Volterra solvers
 * is moved into a common base class:
 *
 *  1. [solvers.fredholm.FredholmSecondKindSolver.combinedNystrom] and
 *     [solvers.volterra.VolterraSecondKindSolver.combinedNystrom] — they have DIFFERENT stopping
 *     criteria of the simple iteration: Fredholm measures the discrepancy at the Gauss nodes
 *     `op.gNode` (`FredholmSolver.kt`), Volterra at `4n+1` uniform control
 *     points (`VolterraSolver.kt`). Merging the bodies into a common core will change the number of iterations,
 *     and hence the result;
 *  2. the non-uniform grids [Grid.quasiUniform], [Grid.geometric], [Grid.graded];
 *  3. intervals other than `[0,1]` — on them the scaling of the threshold
 *     [Grid.breakpointInclusionEps], introduced at stage 3, kicks in.
 *
 * Why the matrix is described ONCE and used both by the snapshot tool
 * ([ExtraBaselineSnapshotTool]) and by the check ([ExtraCharacterizationTest]): the pair
 * "BaselineSnapshotTool + EhCharacterizationTest" duplicates the enumeration of the combinations,
 * and any edit of one file without the other silently takes the baseline out from under the check.
 * Here such a desynchronization is impossible by construction.
 *
 * THE COMPOSITION (see [collect]):
 *  - the equations: Fredholm (`F2`) and Volterra (`V2`) — rational problems whose solution
 *    lies in no generating system (that is, the numbers do not degenerate
 *    into machine zero and are suitable for a relative comparison);
 *  - the schemes: `combinedNystrom`, `iteratedCombinedNystrom` (the targets) plus `base` and
 *    `kulkarni` (controls: they catch a corruption of the assembly of the matrices `M`/`M2`);
 *  - the generating systems: B, H, T;
 *  - the functional families: `theta`, `mu`, `lambda` — ONLY those without a derivative.
 *    The restriction is not a choice but is fixed by the code: `nystromSupport()` starts with
 *    `require(!funcs.usesDerivative)` (`FredholmSolver.kt:265`, symmetrically in
 *    `VolterraSolver.kt`), so the families `xi0`/`xi1`/`xi2` are invalid for the Nyström
 *    schemes and are not in the matrix;
 *  - the grids: `uniform`, `quasiUniform`, `geometric`, `graded`;
 *  - n: 8 and 16;
 *  - the intervals: `[0,1]` (the full matrix) and `[0,2]` (a reduced set, see [collect]).
 *
 * WHY `[0,2]` AND NOT `[-1,1]`: the kernel of the problems `F2`/`V2` equals `1/(1+t+s)` and on `[-1,1]`
 * has a pole at `t+s = -1`, that is, right inside the integration domain. The snapshot
 * would record not the behaviour of the solver but a division by zero. On `[0,2]` the denominator
 * lies in `[1,5]` — there is no singularity, while the scale of the interval is doubled, which is enough for
 * the relative threshold [Grid.breakpointInclusionEps] to kick in.
 *
 * THE DIAGNOSTIC MODE: the solvers are created with `throwOnDivergence = false`. This is DELIBERATE.
 * ON THE CURRENT MATRIX EVERYTHING CONVERGES: all 336 values of `*.iters` lie in the set
 * {13, 15, 47, 107}, no combination reaches the limit 200, including the interval `[0,2]`.
 * The mode is chosen not because divergence is observed, but because it MUST NOT
 * turn the snapshot into a useless one: with the default behaviour (an exception) the very first
 * non-converged combination would record the fact of an exception — ONE bit of information instead of
 * a number — and would bring down the whole shooting of the baseline. In the diagnostic mode both the
 * attained value and the number of iterations are recorded, so the net stays sensitive if
 * a future edit pushes some combination past the convergence limit.
 *
 * THE NUMBER OF ITERATIONS IS RECORDED EXPLICITLY (the keys `*.combNystrom.iters`) and this is a key element
 * of the net: if on merging the solvers the stopping criterion of one of them is substituted by the criterion
 * of the other, but the outcome accidentally coincides in the number of iterations, the value of E_h will not change —
 * it is exactly the iteration counter that makes such a substitution visible.
 */
object ExtraCharacterizationMatrix {

    /** The path to the resource with the baseline snapshot of the additional matrix. */
    const val RESOURCE_PATH = "/characterization/baseline-extra.tsv"

    /** The relative tolerance of the comparison with the baseline (the same as for the main gate). */
    const val RELATIVE_TOLERANCE = 1e-9

    /**
     * The absolute "floor" of the comparison: values below it are considered zero.
     * Needed for the same reason as in [EhCharacterizationTest]: for quantities lying
     * at the level of machine zero a relative comparison is meaningless.
     */
    const val ABSOLUTE_FLOOR = 1e-11

    /**
     * The rounding noise threshold for the E_h and `.iters` keys: a discrepancy |actual − expected|
     * not above it is not checked by the tolerances. `10³·ε·‖u‖∞` at `‖u‖∞ ≈ e` is the same
     * value and justification as for [EhCharacterizationTest.ABSOLUTE_FLOOR]. Measured
     * at the change of the LAPACK implementation (numerical-core 1.0.0): 248 of the 672 E_h keys of this
     * baseline shifted by at most `1.0e-15` abs. (`≈ 2·ε·‖u‖`), which at the tolerance
     * `1e-9` on `E_h ≈ 1e-13…1e-4` produced false failures.
     */
    const val NOISE_FLOOR = 6e-13

    /**
     * The rounding noise threshold SPECIFICALLY for the residual keys: `10·ε·‖u‖∞ ≈ 6e-15`.
     *
     * The common [NOISE_FLOOR] = 6e-13 is inapplicable here for the same reason as the common
     * [ABSOLUTE_FLOOR]: any converged residual is smaller than `1e-13`, and the threshold `6e-13`
     * would absorb ALL the residual keys — the net would become a hole again.
     *
     * Why the threshold is nevertheless needed. The residual is computed as a difference of quantities of order
     * one, so its rounding noise is of order `ε·‖u‖ ≈ 6e-16`, and NOT 1 ULP of the residual
     * value itself, as was assumed in the KDoc of [RESIDUAL_RELATIVE_TOLERANCE].
     * Measured at the change of the LAPACK implementation: 248 of the 336 residual keys shifted,
     * at most `7.8e-16` abs. (`≈ 3.5·ε`), that is, up to `1.3e-2` rel. at the tolerance `1e-3`.
     *
     * The price: for the smallest residuals (`~1.2e-14`) the sensitivity of the gate drops to
     * a `~50 %` shift instead of `0.1 %`. The mutation from the KDoc of [RESIDUAL_RELATIVE_TOLERANCE] was
     * not re-run under this threshold; the relaxation is recorded in
     * `docs/baseline-changes.md` (the entry of 2026-09-09).
     */
    const val RESIDUAL_NOISE_FLOOR = 6e-15

    /**
     * The relative tolerance for the E_h keys BOTH of whose values lie below [ABSOLUTE_FLOOR].
     *
     * Why a separate mode was needed. Previously such a pair was simply SKIPPED
     * (`continue`), and this was a hole of exactly the same kind as the one already closed for the residual
     * keys: under the floor `1e-11` fall 53 of the 672 E_h keys, and among them 3 `combNystrom`
     * keys and 21 `iterCombNystrom` ones, that is, the TARGET schemes of the stage. A change
     * `2.7e-12` → `9e-12` (threefold) passed silently.
     *
     * Why the tolerance is NOT `1e-9`, as for the ordinary keys. The values of this range
     * (the observed minimum is `3.3e-14` with the solutions themselves of order one) are a difference
     * of nearly coinciding numbers, that is, a catastrophic cancellation: a shift of the operands by one
     * ULP (`~2.2e-16`) gives here a relative change of up to a few per cent.
     * A tolerance of `1e-9` would be stricter than machine accuracy and would produce false failures.
     *
     * Why `1e-3`. The same threshold as for the residual keys, and for the same reason: it lies
     * orders of magnitude below any substantial effect of a transcription error (the mutation check
     * shifted the value THREEFOLD, that is, by 200%) and orders of magnitude above the storage noise.
     */
    const val SMALL_VALUE_RELATIVE_TOLERANCE = 1e-3

    /**
     * The absolute "floor" for the E_h keys that fell into the [SMALL_VALUE_RELATIVE_TOLERANCE] mode.
     *
     * It lies two orders below the observed minimum of E_h (`3.3e-14`) and serves only
     * as a protection against division by zero at an exact zero of the error.
     */
    const val SMALL_VALUE_ABSOLUTE_FLOOR = 1e-16

    /** The order of the Gauss quadrature — the same as in the main baseline. */
    private const val QUADRATURE_ORDER = 8

    /**
     * The suffix of the keys storing the ATTAINED RESIDUAL of the iteration (`SolutionFunc.residual`).
     *
     * What these keys exist for. The residual is exactly the quantity measured by the stopping
     * criterion: the uniform norm of the difference of neighbouring iterates ON THE CONTROL SET.
     * For the solvers this set is DIFFERENT (Fredholm — the Gauss nodes `op.gNode`, Volterra —
     * `4n+1` uniform points), and substituting one for the other when merging the solvers changes
     * exactly it.
     *
     * This is established BY MEASUREMENT and not by assumption. The mutation "the Volterra control points
     * are replaced by the Gauss nodes" was introduced and checked: E_h and the number of iterations
     * DID NOT CHANGE BY A SINGLE BIT (the iteration converges in the same number of steps and arrives at the same
     * point — both sets are dense enough, and the convergence is linear). Without the residual
     * keys the net would let this mutation through silently — that is, it would be useless exactly
     * in the place it was created for.
     */
    const val RESIDUAL_SUFFIX = ".residual"

    /**
     * The tolerance of the comparison of the residuals is 1e-3 (relative), and not the common 1e-9.
     *
     * Why NOT 1e-9. The residual is a difference of nearly coinciding iterates (of order 3.6e-14
     * with the values themselves of order 1), that is, a catastrophic cancellation. A perturbation
     * of the operands by just one ULP (~2.2e-16) gives in it a relative change of about
     * 2.2e-16/3.6e-14 ~ 0.6%, that is, a tolerance of 1e-9 here would be stricter than machine accuracy.
     *
     * Why EXACTLY 1e-3 — by measurement and not by taste. The mutation "the Volterra control points
     * `4n+1` → the Gauss nodes" changed 168 of the 336 residual keys (all the Volterra ones;
     * the Fredholm ones were untouched, as they should be), and ALL 168 by more than 1e-3
     * (the median of the changed ones ~1.5%, the maximum 4.0%). The threshold 1e-3 lies below the whole
     * effect of the mutation and 13 orders above a shift of the residual VALUE ITSELF by 1 ULP
     * (~1e-16), that is, it separates the signal from the storage noise.
     *
     * AN HONEST WARNING TO THE MAINTAINER. The `*.residual` keys are DELIBERATELY
     * HYPERSENSITIVE: because of the cancellation they amplify a difference in the last bits
     * by about 1e13 times. If a refactoring is mathematically neutral but NOT bitwise
     * identical (the addition order changed, say), these keys may fail WITH
     * CLEAN E_h keys. The right reaction in such a case is to make sure that (1) all the
     * E_h and `*.iters` keys matched and (2) the control sets of both solvers stayed
     * different, and after that to re-shoot the snapshot DELIBERATELY, and NOT to loosen this threshold.
     */
    const val RESIDUAL_RELATIVE_TOLERANCE = 1e-3

    /**
     * The absolute "floor" SPECIFICALLY for the residual keys: 1e-18.
     *
     * The common [ABSOLUTE_FLOOR] = 1e-11 is INAPPLICABLE here and would be a HIDDEN HOLE in the net.
     * The stopping criterion of the iteration equals 1e-13, so ANY converged residual is by
     * construction smaller than 1e-13, that is, always below 1e-11 — with the common floor ALL the residual
     * keys would be declared "zero" and would not be compared at all. This is not a guess:
     * the first edition of the test MISSED exactly this way the mutation (a) on a green run,
     * although the numbers in the snapshot already differed.
     *
     * 1e-18 lies four orders below the observed residuals (~3.6e-14) and serves
     * only as a protection against division by zero at an exact zero of the residual.
     */
    const val RESIDUAL_ABSOLUTE_FLOOR = 1e-18

    /** The description of the integration interval: a short tag for the key and the bounds themselves. */
    private data class Segment(val tag: String, val a: Double, val b: Double)

    /** The description of a grid factory: a short tag for the key and a builder. */
    private data class GridKind(val tag: String, val build: (Int, Double, Double) -> Grid)

    private val gridKinds = listOf(
        GridKind("uniform") { n, a, b -> Grid.uniform(n, a, b) },
        GridKind("quasi") { n, a, b -> Grid.quasiUniform(n, a, b) },
        GridKind("geom") { n, a, b -> Grid.geometric(n, a, b) },
        GridKind("graded") { n, a, b -> Grid.graded(n, a, b) },
    )

    private val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)

    /**
     * The functional families WITHOUT a derivative — the only ones applicable to the Nyström schemes.
     * The invariant is checked in [collect] by an explicit `usesDerivative` check.
     */
    private val familyNames = listOf("theta", "mu", "lambda")

    private val sizes = listOf(8, 16)

    private val unitSegment = Segment("s01", 0.0, 1.0)
    private val scaledSegment = Segment("s02", 0.0, 2.0)

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        "lambda" -> ThreePointFunctionals(basis)
        else -> error("ExtraCharacterizationMatrix: unknown family '$name'")
    }

    /**
     * The printed representation of a snapshot value.
     *
     * 17 significant digits — as in [BaselineSnapshotTool]: this is the minimal precision at
     * which the decimal record of a `double` is restored BITWISE. The former 12 digits do NOT
     * give a round-trip (`"%.12g".format(0.1 + 0.2)` = `0.300000000000` ≠ `0.1 + 0.2`), that is,
     * the baseline itself introduced a relative storage error of ~1e-12 — coarser than the real
     * discrepancy of the backends. The locale [Locale.ROOT]
     * is set explicitly: `"%.17g".format(x)` uses the default locale and on a machine with
     * a Russian locale writes a comma instead of a dot, after which the snapshot stops being readable.
     */
    fun formatValue(value: Double): String = when {
        value.isNaN() -> "NaN"
        value.isInfinite() -> if (value > 0) "Infinity" else "-Infinity"
        else -> String.format(Locale.ROOT, "%.17g", value)
    }

    /**
     * Computes one snapshot value, INTERCEPTING a failure.
     *
     * The snapshot is characterization: if a combination fails today, the fact of the
     * failure is recorded (`ERROR:<exception class>`), rather than the problem being "fixed". This keeps the
     * combination in the net: turning a failure into a number (or into another exception) after
     * a refactoring will be noticed just as reliably as a change of a number.
     */
    private fun evaluate(compute: () -> Double): String = try {
        formatValue(compute())
    } catch (e: Throwable) {
        "ERROR:" + (e::class.simpleName ?: e::class.java.name)
    }

    /**
     * Collects the whole matrix as a list of "key - value" pairs sorted by key.
     *
     * The sorting is mandatory: it makes both the snapshot file and its diff meaningful
     * regardless of the traversal order of the loops.
     */
    fun collect(): List<Pair<String, String>> {
        val rows = mutableListOf<Pair<String, String>>()

        // The full matrix on the unit interval.
        for (segment in listOf(unitSegment)) {
            for (kind in gridKinds) {
                for (system in systems) {
                    for (familyName in familyNames) {
                        for (n in sizes) {
                            collectFredholm(rows, segment, kind, system, familyName, n)
                            collectVolterra(rows, segment, kind, system, familyName, n)
                        }
                    }
                }
            }
        }

        // A reduced set on the interval [0,2]: the goal is to catch scale-dependent errors
        // (the breakpoint inclusion threshold, the step normalizations), and not to repeat the whole matrix.
        // Two grids are enough (a uniform one as a control and a geometric one as a
        // substantially non-uniform one) and one family.
        for (kind in gridKinds.filter { it.tag == "uniform" || it.tag == "geom" }) {
            for (system in systems) {
                for (n in sizes) {
                    collectFredholm(rows, scaledSegment, kind, system, "theta", n)
                    collectVolterra(rows, scaledSegment, kind, system, "theta", n)
                }
            }
        }

        return rows.sortedBy { it.first }
    }

    private fun keyPrefix(
        equation: String,
        problemName: String,
        system: GeneratingSystem,
        familyName: String,
        kind: GridKind,
        segment: Segment,
        n: Int,
    ): String = "$equation.$problemName.${system.name}.$familyName.${kind.tag}.${segment.tag}.n$n"

    /** The schemes of the Fredholm solver for one combination of parameters. */
    private fun collectFredholm(
        rows: MutableList<Pair<String, String>>,
        segment: Segment,
        kind: GridKind,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ) {
        val problem = problems.fredholm.FredholmProblem.F2
        val grid = kind.build(n, segment.a, segment.b)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        check(!funcs.usesDerivative) {
            "The family '${funcs.name}' uses a derivative and is incompatible with the Nyström schemes"
        }
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
        val solver = solvers.fredholm.FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            throwOnDivergence = false,
        )
        val exact = { t: Double -> problem.exact(t) }
        val prefix = keyPrefix("F", problem.name, system, familyName, kind, segment, n)
        emitSchemes(rows, prefix, grid, exact) { scheme ->
            when (scheme) {
                Scheme.BASE -> solver.base()
                Scheme.KULKARNI -> solver.kulkarni()
                Scheme.COMBINED_NYSTROM -> solver.combinedNystrom()
                Scheme.ITERATED_COMBINED_NYSTROM -> solver.iteratedCombinedNystrom()
            }
        }
    }

    /** The schemes of the Volterra solver for one combination of parameters. */
    private fun collectVolterra(
        rows: MutableList<Pair<String, String>>,
        segment: Segment,
        kind: GridKind,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ) {
        val problem = problems.volterra.VolterraProblem.V2
        val grid = kind.build(n, segment.a, segment.b)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        check(!funcs.usesDerivative) {
            "The family '${funcs.name}' uses a derivative and is incompatible with the Nyström schemes"
        }
        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
        val solver = solvers.volterra.VolterraSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            throwOnDivergence = false,
        )
        val exact = { t: Double -> problem.exact(t) }
        val prefix = keyPrefix("V", problem.name, system, familyName, kind, segment, n)
        emitSchemes(rows, prefix, grid, exact) { scheme ->
            when (scheme) {
                Scheme.BASE -> solver.base()
                Scheme.KULKARNI -> solver.kulkarni()
                Scheme.COMBINED_NYSTROM -> solver.combinedNystrom()
                Scheme.ITERATED_COMBINED_NYSTROM -> solver.iteratedCombinedNystrom()
            }
        }
    }

    /** The schemes shot for every combination of parameters. */
    private enum class Scheme(val tag: String) {
        BASE("base"),
        KULKARNI("kulkarni"),
        COMBINED_NYSTROM("combNystrom"),
        ITERATED_COMBINED_NYSTROM("iterCombNystrom"),
    }

    /**
     * Shoots the E_h of all the schemes and additionally the number of iterations of the iterative Nyström schemes.
     *
     * The solution is given by the function [solve] and not by a ready list. The solvers now have a
     * common base (`SecondKindSolverCore`), but the schemes called here, `combinedNystrom`
     * and `iteratedCombinedNystrom`, are DELIBERATELY NOT moved into it (different Nyström weights
     * and different stopping criteria), that is, they are declared in the subclasses independently.
     * Therefore a closure remains the only way to combine them in one matrix —
     * and this is RIGHT: the net must call exactly the implementations it guards.
     */
    private fun emitSchemes(
        rows: MutableList<Pair<String, String>>,
        prefix: String,
        grid: Grid,
        exact: (Double) -> Double,
        solve: (Scheme) -> SolutionFunc,
    ) {
        for (scheme in Scheme.entries) {
            var iterations: Int? = null
            var residual: Double? = null
            val value = evaluate {
                val solution = solve(scheme)
                iterations = solution.iterations
                residual = solution.residual
                errorEh(exact, solution.eval, grid)
            }
            rows += "$prefix.${scheme.tag}" to value
            // The iteration counter and the residual are meaningful only for the iterative schemes; for the direct
            // ones they are identically 0 and only add noise to the diff.
            //
            // A CAVEAT ON THE DEPENDENCE OF THE KEYS. For ITERATED_COMBINED_NYSTROM the pairs
            // `.iters`/`.residual` are BITWISE EQUAL to the corresponding COMBINED_NYSTROM keys:
            // `iteratedCombinedNystrom` inherits these fields from the original solution without
            // recomputation (one integration, no iterations of its own). That is, 168 pairs
            // of the 1344 values do NOT give independent coverage — there are 1176 independent values
            // in the net, and claiming 1344 as a measure of independent coverage is wrong.
            // The keys are nevertheless shot: they record THE VERY FACT of the inheritance. If an edit
            // makes `iteratedCombinedNystrom` recompute the stopping criterion anew
            // (for example, in an attempt to merge it with `combinedNystrom`), the equality will break
            // and the test will show it — while by E_h alone such an edit may pass silently.
            val iterationsMatter =
                scheme == Scheme.COMBINED_NYSTROM || scheme == Scheme.ITERATED_COMBINED_NYSTROM
            if (iterationsMatter) {
                val recordedIterations = iterations
                rows += "$prefix.${scheme.tag}.iters" to
                    (recordedIterations?.toString() ?: "ERROR:NoIterations")
                val recordedResidual = residual
                rows += "$prefix.${scheme.tag}$RESIDUAL_SUFFIX" to
                    (recordedResidual?.let { formatValue(it) } ?: "ERROR:NoResidual")
            }
        }
    }
}
