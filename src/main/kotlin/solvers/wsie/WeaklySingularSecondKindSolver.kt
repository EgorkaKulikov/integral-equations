package solvers.wsie

import numerics.NumericsContext
import solvers.core.ImageTriple
import solvers.core.RhsWithDerivatives
import solvers.core.SecondKindSolverCore
import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily

/**
 * Second-kind solver for `u − cL·𝓛u = f` with a weakly singular operator 𝓛 ([WeaklySingularOperator]).
 *
 * All schemes come from [SecondKindSolverCore] unchanged: `base()` (collocation through the functionals chi),
 * `kulkarni()` (modified projection; the reduction `(I − M − M2 + M²)c = (I − M)g + d` for projector families,
 * plain iteration otherwise), `iteratedKulkarni()` and `sloan()` (kept as a reference scheme). This class only
 * supplies the images `𝓛ω_i`, `𝓛(𝓛ω_i)` and `𝓛g`.
 *
 * WHY ANY VALUE-ONLY FAMILY IS ACCEPTED. The schemes need nothing from chi beyond `chi_j(g)` and, for
 * Kulkarni, the projector property; both are properties of the family, not of the kernel. Families that use
 * derivatives (de Boor–Fix `xi`, `xi<0>`) are rejected: they would require `(𝓛u)'` and `(𝓛u)''`, which are
 * unbounded near the diagonal and at the ends for `|t − s|^(−alpha)` kernels, so there is nothing finite to
 * implement. The derivative slots of every [ImageTriple] and of the right-hand side therefore hold closures
 * that throw [UnsupportedOperationException]; value functionals never call them (`ValueFunctional.apply`
 * reads `f` only), and the core only builds — never calls — the derivative closures of `kulkarni()`.
 *
 * Status (AGENTS.md §8): the schemes are those of [SecondKindSolverCore] ([Kulkarni 2003] for projectors);
 * their application to weakly singular kernels with minimal-spline functionals is an adaptation without a
 * proven order here — convergence orders are numerical observations.
 *
 * @param op weakly singular operator; it must be built on the same [splines.Grid] object as [basis], because
 *   the supports `[x_j, x_{j+3}]` of the basis functions are passed to [WeaklySingularOperator.applyOnSupport]
 *   and must coincide with the breakpoints of the operator partition.
 */
public class WeaklySingularSecondKindSolver(
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    public val op: WeaklySingularOperator,
    cL: Double,
    rhs: (Double) -> Double,
    ctx: NumericsContext = NumericsContext.default(),
) : SecondKindSolverCore<(Double) -> Double>(
    basis, funcs, cL, RhsWithDerivatives(rhs, UNSUPPORTED_DERIVATIVE, UNSUPPORTED_DERIVATIVE), true, ctx,
) {
    init {
        require(!funcs.usesDerivative && !funcs.usesSecondDerivative) {
            "WeaklySingularSecondKindSolver: the functional family '${funcs.name}' uses derivatives; " +
                "derivatives of the image of a weakly singular operator are unbounded, " +
                "so only value-based families (e.g. theta, theta-tau, lambda) are supported"
        }
        require(op.grid === basis.grid) {
            "WeaklySingularSecondKindSolver: the operator and the basis must share the same Grid object"
        }
    }

    override val equationName: String
        get() = "WeaklySingular" + if (op is WeaklySingularVolterraOperator) "Volterra" else "Fredholm"

    override val kulkarniQuasiHint: String
        get() = "For quasi-interpolants the property P^2 = P does not hold, so the Kulkarni reduction is " +
            "inapplicable and a simple iteration is used; it requires a contraction"

    override val checkPoints: DoubleArray
        get() = grid.breakpoints

    override fun prepare(u: (Double) -> Double): (Double) -> Double = u

    override fun image(o: (Double) -> Double): (Double) -> Double = { t -> cL * op.apply(t, o) }

    /** Returns a throwing closure instead of throwing: `kulkarni()` builds this closure eagerly but never calls it. */
    override fun imageDeriv(o: (Double) -> Double): (Double) -> Double = UNSUPPORTED_DERIVATIVE

    /** See [imageDeriv]. */
    override fun imageDeriv2(o: (Double) -> Double, uD: (Double) -> Double): (Double) -> Double =
        UNSUPPORTED_DERIVATIVE

    override fun applyOperator(t: Double, u: (Double) -> Double): Double = op.apply(t, u)

    override fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double = unsupported()

    override fun applyOperatorDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double =
        unsupported()

    /** `cL·𝓛ω_i`, integrated over the support `[x_j, x_{j+3}] ∩ [a, b]` only (`j = i − 2`). */
    private fun omegaImage(i: Int): (Double) -> Double {
        val j = i - 2
        val lo = maxOf(grid.a, grid.x(j))
        val hi = minOf(grid.b, grid.x(j + 3))
        val omega = { s: Double -> basis.omega(j, s) }
        return { t -> cL * op.applyOnSupport(t, lo, hi, omega) }
    }

    override fun omegaImages(i: Int): ImageTriple =
        ImageTriple(omegaImage(i), UNSUPPORTED_DERIVATIVE, UNSUPPORTED_DERIVATIVE)

    override fun doubleOmegaImages(i: Int): ImageTriple {
        val inner = omegaImage(i)
        return ImageTriple({ t -> cL * op.apply(t, inner) }, UNSUPPORTED_DERIVATIVE, UNSUPPORTED_DERIVATIVE)
    }

    private companion object {
        private fun unsupported(): Nothing = throw UnsupportedOperationException(
            "Derivatives of a weakly singular operator image are unbounded and are not provided",
        )

        private val UNSUPPORTED_DERIVATIVE: (Double) -> Double = { unsupported() }
    }
}
