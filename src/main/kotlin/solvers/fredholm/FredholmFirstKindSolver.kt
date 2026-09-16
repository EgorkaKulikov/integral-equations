package solvers.fredholm

import numerics.*
import splines.*
import solvers.core.SolutionFunc
import splines.functionals.*
import splines.metrics.*
import solvers.core.RhsWithDerivatives

/**
 * Solver for the ill-posed FIRST-kind Fredholm equation `K u = f` by the method of
 * regularization.
 *
 * The mathematical idea: the equation is replaced by the perturbed one `(alpha I + K) u_alpha = f`,
 * which is algebraically equivalent to the SECOND-kind equation
 *
 *     u_alpha - K_eff u_alpha = f / alpha,   K_eff = -(1/alpha) K,
 *
 * after which the usual second-kind scheme with `c_L = -1/alpha` is applied.
 * The source of the method is given in `docs/REFERENCES.md` (section "First-kind equations").
 *
 * CONDITIONING LIMITATION: the entries of the matrix `M` grow as `alpha^{-1}`,
 * and those of `M2` as `alpha^{-2}`, so for small `alpha` the problem becomes ill-
 * conditioned. For this reason only the base scheme and the Sloan iteration are published:
 * the Kulkarni scheme, which uses `M2`, is deemed inapplicable to a first-kind equation.
 *
 * THE LIMIT OF APPLICABILITY IN `alpha` — MEASURED, NOT ASSUMED. The statement
 * above is qualitative; here are the numbers established by an independent numerical verification
 * (by a second, independently written code):
 *
 *  - AT `alpha = 1e-10` FEW SIGNIFICANT DIGITS REMAIN. The conditioning of the assembled
 *    system was `cond_inf ≈ 2.636149e+10`. Eight LEGITIMATE implementation
 *    variants (different evaluation orders of the kernel and the right-hand side, different dense
 *    linear solvers) produced FOUR distinct values of ONE and the same quantity:
 *    `3.822e-05 / 4.108e-05 / 4.176e-05 / 4.402e-05`, a spread of `15.19 %`.
 *  - THIS IS NOT A BUG BUT AN ACCURACY LIMIT. The a priori estimate
 *    `eps · cond / value ≈ 14.6 %` matches the observed spread, i.e. the
 *    effect is fully explained by the conditioning and not by an error in the code.
 *  - THE LIMIT IS ROUGHLY `alpha = 1e-8`. At `alpha = 1e-6` and `alpha = 1e-8` the same
 *    eight variants show NO discrepancy: `0.00 %` and `0.01 %` respectively.
 *    Below roughly `1e-8` few significant digits remain in the result, and `alpha`
 *    must be chosen deliberately.
 *  - HOW LITTLE IT TAKES TO SHIFT THE ANSWER. Writing the exact solution as the
 *    division `cos(wt)/(1+t)` versus the multiplication `cos(wt) · p`, where `p = 1/(1+t)`,
 *    differs by EXACTLY ONE rounding — and changes the result by `7.49 %`.
 *
 * HOW TO CHECK YOURSELF (a self-check that needs no external baseline). Compute
 * the quantity of interest in TWO algebraically EQUIVALENT ways
 * (for example, a division versus a multiplication by the reciprocal) and compare.
 * The discrepancy is NOT an error of one of the variants — it is a LOWER BOUND on the actually
 * available accuracy: the digits that differ between two legitimate ways of writing the formula
 * carry no information in either of them. From above the accuracy is bounded by
 * `cond_inf · ε`; both factors can be measured with [baseCondition] and
 * [numerics.LinearAlgebra.solveDiagnosed]. A detailed discussion is in `docs/ACCURACY.md`.
 *
 * @param basis minimal spline basis.
 * @param funcs family of approximation functionals.
 * @param op Fredholm operator with the original kernel.
 * @param rhs right-hand side `f(t)` of the original first-kind equation.
 * @param rhsDeriv first derivative `f'(t)`.
 * @param rhsDeriv2 second derivative `f''(t)`; needed by the family `xi^<0>`.
 * @param alpha regularization parameter; must be strictly positive.
 * @param throwOnDivergence policy for handling a failure to converge by the iterative
 *        schemes of the inner solver; see [FredholmSecondKindSolver.throwOnDivergence].
 * @throws IllegalArgumentException if `alpha <= 0`.
 */
public class FredholmFirstKindSolver(
    public val basis: MinimalSplineBasis,
    public val funcs: FunctionalFamily,
    public val op: FredholmOperator,
    rhs: (Double) -> Double,
    rhsDeriv: (Double) -> Double,
    rhsDeriv2: (Double) -> Double = { 0.0 },
    public val alpha: Double = DEFAULT_REGULARIZATION,
    public val throwOnDivergence: Boolean = true,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    public companion object {
        /**
         * Default value of the regularization parameter.
         *
         * This is an EXPERIMENTAL choice of the authors of the cited work, not a recommendation
         * from the theory of the regularization method: the optimal `alpha` depends on the noise level
         * in the data and in general should be tuned (for example, by the discrepancy principle).
         *
         * THE DEFAULT VALUE FALLS INTO THE PROBLEMATIC RANGE. It is exactly at
         * `alpha = 1e-10` that a spread of `15.19 %` was measured between eight algebraically
         * equivalent ways of writing one quantity, whereas at `1e-6`
         * and `1e-8` there is no discrepancy (`0.00 %` and `0.01 %`); the numbers and their provenance are
         * in the KDoc of [FredholmFirstKindSolver] and in `docs/ACCURACY.md`. The value is KEPT
         * (it must not be changed — it reproduces the publication), but it should not be accepted
         * silently: if the problem tolerates a larger `alpha`, the accuracy is higher.
         */
        public const val DEFAULT_REGULARIZATION: Double = 1e-10
    }

    init {
        require(alpha > 0.0) {
            "FredholmFirstKindSolver: the regularization parameter alpha must be positive, got alpha=$alpha"
        }
    }

    private val cL = -1.0 / alpha
    private val fEff = { t: Double -> rhs(t) / alpha }
    private val fEffDeriv = { t: Double -> rhsDeriv(t) / alpha }
    // The order in which the lambdas are built is preserved literally; they are merged into a single
    // object below, at the call boundary of the inner solver.
    // The second derivative of the right-hand side is passed to the inner solver WITHOUT FAIL:
    // without it the family xi^<0>, which reads f'', silently received zero instead of
    // the true value and built a wrong system without any diagnostics.
    private val fEffDeriv2 = { t: Double -> rhsDeriv2(t) / alpha }
    private val inner = FredholmSecondKindSolver(
        basis, funcs, op, cL,
        RhsWithDerivatives(fEff, fEffDeriv, fEffDeriv2),
        throwOnDivergence, ctx,
    )

    /** Base collocation scheme for the regularized equation. */
    public fun base(): SolutionFunc = inner.base()

    /**
     * Condition number `cond_inf` of the ASSEMBLED matrix `I - M` of the base scheme
     * — together with a reliability flag for the estimate itself.
     *
     * WHY. The limit of applicability in `alpha` (see the class KDoc) is stated in terms of
     * `cond_inf`, but without this method a user could not see it for THEIR OWN
     * problem and THEIR OWN `alpha`: the number in the KDoc refers to a specific run, not to the
     * method in general. The method was added PRECISELY for the reproducibility protocol.
     *
     * IT IS CALLED BY NO SCHEME and changes nothing in them: the O(n⁴) cost
     * (assembly of `M` plus an inversion) is paid only by those who explicitly ask.
     *
     * THE ESTIMATE MAY TURN OUT UNRELIABLE, and this is a regular outcome, not an error:
     * for small `alpha` the matrix is nearly singular, and an estimate via inversion in this
     * regime loses its meaning (measured: at `alpha <= 1e-12` the inversion residual
     * reached 0.08…760). That is exactly why a [ConditionEstimate] is returned
     * rather than a `Double`: it may be printed only after checking
     * [ConditionEstimate.isReliable] (or through [ConditionEstimate.valueOrNull]).
     */
    public fun baseCondition(
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = Conditioning.conditionInf(inner.baseMatrix(), tolerance)

    /** Sloan iteration applied to the regularized equation. */
    public fun sloan(): SolutionFunc = inner.sloan()

    /**
     * Kulkarni scheme for the regularized equation.
     *
     * NOT RECOMMENDED for use: the scheme uses the matrix `M2`, whose entries
     * grow as `alpha^{-2}`, which for the typical `alpha ~ 1e-10` makes the system
     * numerically unsolvable. The method is kept for API completeness and experiments.
     */
    public fun kulkarni(): SolutionFunc = inner.kulkarni()

    /** Iterated Kulkarni scheme; the same limitations as for [kulkarni]. */
    public fun iteratedKulkarni(): SolutionFunc = inner.iteratedKulkarni()
}
