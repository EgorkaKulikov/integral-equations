package solvers.wsie

import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.NumericsContext
import numerics.ParallelAssembly
import solvers.core.ImageTriple
import solvers.core.RhsWithDerivatives
import solvers.core.SecondKindSolverCore
import solvers.core.SolutionFunc
import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily
import kotlin.math.abs

/**
 * Second-kind solver for `u − cL·𝓛u = f` with a weakly singular operator 𝓛 ([WeaklySingularOperator]).
 *
 * All schemes come from [SecondKindSolverCore] unchanged: `base()` (collocation through the functionals chi),
 * `kulkarni()` (modified projection; the reduction `(I − M − M2 + M²)c = (I − M)g + d` for projector families,
 * plain iteration otherwise), `iteratedKulkarni()` and `sloan()` (kept as a reference scheme). This class only
 * supplies the images `𝓛ω_i`, `𝓛(𝓛ω_i)` and `𝓛g`.
 *
 * The two-grid scheme [twoGridKulkarni] (M-DS) is defined here rather than in the core: it couples this solver
 * with a second one on a nested fine grid and uses only value images `𝓛ω_i` of both levels.
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
    /**
     * Two-grid modification of the Kulkarni method (M-DS), after [Grammont, Kulkarni, Vasconcelos 2023] (GKV),
     * transferred to minimal-spline projector families.
     *
     * The coarse level is this solver (grid X_n, family chi^(n), operator on X_n); [fine] supplies the nested
     * fine level X_m, m = n·p (basis, family chi^(m) and the operator built on X_m). The scheme solves
     * `u = f + U u` with `U = P_n 𝓛 P_m + 𝓛 P_m P_n − P_n 𝓛 P_m P_n`; under the nesting S(X_n) ⊂ S(X_m) and the
     * projector property of chi^(m) (`P_m P_n = P_n`) this is `P_n 𝓛 P_m + 𝓛 P_n − P_n 𝓛 P_n`. With
     * `c = chi^(n)(u)` it reduces to the system of size n + 2
     *   (I − M − M_nm M_mn + M²) c = (I − M) g_n + M_nm g_m,
     * M_nm = [chi^(n)_j(cL·𝓛ω^(m)_k)], M_mn = [chi^(m)_l(cL·𝓛ω^(n)_i)], M = M_nm R, R = [chi^(m)_l(ω^(n)_i)]
     * (refinement matrix), g_n = chi^(n)(f), g_m = chi^(m)(f), and the solution is recovered as
     *   u = f + Σ_j (c − g_n − M c)_j ω^(n)_j + cL·𝓛(Σ_i c_i ω^(n)_i).
     *
     * WHY THIS ASSEMBLY. Classical [kulkarni] needs M2 = chi(𝓛(𝓛ω_i)), a nested integral per entry; here
     * every entry is a single application of 𝓛 to one basis function on its support, i.e. O(n·m) operator
     * applications and no nested integrals. M is taken as M_nm R (the identity M_nm R = chi^(n)(𝓛ω^(n)) holds
     * under the nesting) rather than reassembled on the coarse operator, so that the system is exactly the
     * elimination of a = chi^(m)(u) from the two-level equations.
     *
     * p = 1 is accepted as a degenerate case: then U = 𝓛 P_n, the system becomes (I − M) c = g, and the scheme
     * coincides with [sloan] (iterated collocation), not with [kulkarni].
     *
     * Checked here: the fine grid has n·p cells and its nodes x^(m)_{pk} coincide with x^(n)_k within
     * [splines.Grid.breakpointInclusionEps]; both families are projectors; both levels use the same generating
     * system, the same cL and the same kind of operator with the same exponent. NOT checked: the space inclusion
     * S(X_n) ⊂ S(X_m) beyond the node nesting (it holds for the polynomial system B, whose minimal splines are
     * C¹ quadratic splines) and the equality of the smooth kernel factors k(t, s) (closures cannot be compared).
     * The right-hand side is that of this (coarse) solver; the right-hand side of [fine] is not used.
     *
     * Status (AGENTS.md §8): adaptation of GKV (piecewise polynomials, interpolatory projector, k ≡ 1) to
     * minimal-spline projector families; no convergence order is proven here — orders are numerical observations.
     *
     * @param fine solver on the nested fine grid with n·p cells; its operator must be built on its own grid.
     * @param p refinement factor m / n (>= 1).
     */
    public fun twoGridKulkarni(fine: WeaklySingularSecondKindSolver, p: Int): SolutionFunc {
        requireTwoGridPair(fine, p)
        val nc = dim
        val (mNm, mMn, r) = twoGridBlocks(fine)
        val m = LinearAlgebra.matMat(mNm, r, ctx.backend)
        val mm = LinearAlgebra.matMat(m, m, ctx.backend)
        val nmmn = LinearAlgebra.matMat(mNm, mMn, ctx.backend)
        val gN = vectorG()
        val gM = fine.funcs.projectorCoeffs(fEff)
        // A = I − M − M_nm M_mn + M²
        val a = DenseMatrix.zeros(nc, nc)
        for (row in 0 until nc) {
            for (col in 0 until nc) a[row, col] = -m[row, col] - nmmn[row, col] + mm[row, col]
            a[row, row] += 1.0
        }
        // rhs = (I − M) g_n + M_nm g_m
        val mg = LinearAlgebra.matVec(m, gN, ctx.backend)
        val nmg = LinearAlgebra.matVec(mNm, gM, ctx.backend)
        val rhs = DoubleArray(nc) { gN[it] - mg[it] + nmg[it] }
        val c = LinearAlgebra.solve(a, rhs, ctx.backend)
        // u = f + Σ_j e_j ω^(n)_j + cL·𝓛(P_n u), e = c − g_n − M c = chi^(n)(cL·𝓛(P_m − P_n)u).
        val mc = LinearAlgebra.matVec(m, c, ctx.backend)
        val e = DoubleArray(nc) { c[it] - gN[it] - mc[it] }
        val splineImage = image(prepare { s -> basis.evalSpline(c, s) })
        return SolutionFunc(eval = { t -> fEff(t) + basis.evalSpline(e, t) + splineImage(t) })
    }

    /**
     * The blocks (M_nm, M_mn, R) of [twoGridKulkarni]; `internal` so that the tests can check the nesting
     * (R reproduces ω^(n)_i in the fine basis) and the identity M_nm R = M without duplicating the assembly.
     */
    internal fun twoGridBlocks(fine: WeaklySingularSecondKindSolver): Triple<DenseMatrix, DenseMatrix, DenseMatrix> {
        // M_nm, (n+2) x (m+2): the fine operator integrates over the fine supports [x_k, x_{k+3}] of X_m.
        val mNm = chiColumns(funcs, fine.dim) { k -> fine.omegaImage(k) }
        // M_mn, (m+2) x (n+2): the coarse operator integrates over the coarse supports of X_n.
        val mMn = chiColumns(fine.funcs, dim) { i -> omegaImage(i) }
        // R, (m+2) x (n+2): coefficients of omega^(n)_i in the fine basis.
        val r = chiColumns(fine.funcs, dim) { i -> { s: Double -> basis.omega(i - 2, s) } }
        return Triple(mNm, mMn, r)
    }

    private fun requireTwoGridPair(fine: WeaklySingularSecondKindSolver, p: Int) {
        require(p >= 1) { "twoGridKulkarni: p must be >= 1, got p=$p" }
        require(fine.n == n * p) { "twoGridKulkarni: the fine grid must have n*p = ${n * p} cells, got ${fine.n}" }
        val eps = grid.breakpointInclusionEps
        for (k in 0..n) require(abs(fine.grid.x(p * k) - grid.x(k)) <= eps) {
            "twoGridKulkarni: the grids are not nested: fine x(${p * k}) = ${fine.grid.x(p * k)}, " +
                "coarse x($k) = ${grid.x(k)}"
        }
        require(funcs.isProjector && fine.funcs.isProjector) {
            "twoGridKulkarni: both families must be projectors; got '${funcs.name}' and '${fine.funcs.name}'"
        }
        require(fine.basis.sys == basis.sys) { "twoGridKulkarni: both levels must use the same generating system" }
        require(fine.cL == cL) { "twoGridKulkarni: both levels must use the same cL, got $cL and ${fine.cL}" }
        require(fine.op::class == op::class && fine.op.kernel.alpha == op.kernel.alpha) {
            "twoGridKulkarni: both levels must use the same kind of operator with the same exponent alpha"
        }
    }

    /**
     * Matrix `rowFuncs.chi(j − 2)(column(k))` of size (rowFuncs.n + 2) x [cols], assembled column by column as in
     * the core: the columns are independent and DenseMatrix is column-major, so no transposition is needed.
     */
    private fun chiColumns(rowFuncs: FunctionalFamily, cols: Int, column: (Int) -> (Double) -> Double): DenseMatrix {
        val rows = rowFuncs.n + 2
        val assembled = ParallelAssembly.assembleRows(cols, rows, ctx.parallel) { k ->
            val g = column(k)
            DoubleArray(rows) { j -> rowFuncs.chi(j - 2).apply(g, UNSUPPORTED_DERIVATIVE) }
        }
        val data = DoubleArray(rows * cols)
        for (k in 0 until cols) assembled[k].copyInto(data, k * rows)
        return DenseMatrix.fromColumnMajor(rows, cols, data)
    }

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
