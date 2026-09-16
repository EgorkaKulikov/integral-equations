package solvers.core

import kotlin.math.abs
import numerics.*
import splines.*
import solvers.core.SolutionFunc
import solvers.core.reportConvergence
import splines.functionals.*
import splines.metrics.*
import solvers.core.SecondKindDefaults.KULKARNI_QUASI_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.KULKARNI_QUASI_TOLERANCE

/**
 * The image `L u` together with its two derivatives — exactly what the functional `chi_j`
 * receives as input while the matrices `M` and `M2` are assembled.
 *
 * The triple is returned as ONE object rather than by three independent calls: for Volterra
 * all three closures are built from a single prepared operand, and splitting them would force
 * the operand (and with it the integrand cache) to be created anew,
 * changing the number of kernel evaluations.
 */
public class ImageTriple(
    public val value: (Double) -> Double,
    public val deriv: (Double) -> Double,
    public val deriv2: (Double) -> Double,
)

/**
 * SHARED CORE OF THE LINEAR SOLVERS FOR THE SECOND-KIND EQUATION `u - L u = f`.
 *
 * It collects everything that was LITERALLY identical in the Fredholm and Volterra solvers, or
 * differed only in how the operator `L` is applied: assembly of the matrices `M`, `M2`,
 * of the vectors `g`, `d`, the base collocation, the Sloan iteration and the whole Kulkarni family.
 *
 * Matrices of the discrete problem:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * ### What DELIBERATELY did not move here
 *
 * The whole Nyström family (`nystromSupport`, `nystromEval`, `nystrom`,
 * `iteratedNystrom`, `combinedNystrom`, `iteratedCombinedNystrom`) stayed
 * in the subclasses. The reason is structural rather than a matter of size: for Fredholm the
 * weights `b_r` do NOT depend on `t` and are computed once, while for Volterra they are recomputed
 * at every `t` because of the causal truncation of the support `[a, t]`. As a consequence
 * the result types differ (`Pair` versus a dedicated class), so do the signatures of
 * `nystromEval` and the split "matrix assembly / system solve"; for
 * `combinedNystrom` even the point set on which the stopping criterion is measured differs.
 * Merging them would change the order of operations, i.e. the numbers.
 *
 * ### The "prepared operand" abstraction ([Operand])
 *
 * The only substantial difference in the shared part is HOW the function to which `L` is
 * applied repeatedly is represented:
 *
 * * Fredholm: the integration limits are constant, hence a fixed set of global Gauss nodes
 *   exists and the operand is the array of function values at those nodes (`DoubleArray`);
 *   a repeated application of `L` is a convolution with the kernel over the already available
 *   values.
 * * Volterra: the upper limit `t` is variable, no fixed node set EXISTS, and the operand is the
 *   function itself (a closure); `L` is applied through a quadrature over the truncated interval.
 *
 * The difference is expressed by the type parameter [Operand] and the pair [prepare] / [image],
 * NOT by a boolean flag "this is Volterra": such a flag would have to be checked in every method,
 * and adding a third equation would require editing every branch.
 * A type parameter was chosen over a wrapper interface deliberately: `DoubleArray`
 * and `(Double) -> Double` are both reference types, so there is no boxing, whereas a wrapper
 * would add one object per prepared operand on the hot path of matrix assembly.
 *
 * ### Initialization order
 *
 * The initializers of the BASE class run BEFORE those of the derived class,
 * therefore there is not a single field here whose value is read through an
 * `abstract`/`open` member: [grid], [n] and [dim] are computed exclusively from the
 * primary constructor parameters. All extension points ([checkPoints],
 * [prepare], [image], …) are called ONLY from methods, i.e. after the subclass constructor
 * has finished. Violating this rule yields a silent defect: for instance `dim` would see
 * `n = 0` and become `2`, all matrices would become `2x2`, and the system would be
 * solvable and meaningless.
 *
 * @param basis minimal spline basis.
 * @param funcs family of functionals chi_j.
 * @param cL operator factor: `c_L = 1` for a second-kind equation;
 *        `c_L = -1/alpha` when a first-kind equation is regularized following Wazwaz.
 * @param rhs right-hand side `f` together with its two derivatives — supplied explicitly so that
 *        the solver can be reused by the first-kind solvers with a DIFFERENT (effective)
 *        right-hand side. As one object rather than three parameters: see [RhsWithDerivatives].
 * @param throwOnDivergence behaviour of the ITERATIVE schemes ([kulkarni] for
 *        quasi-interpolants, `combinedNystrom` in the subclasses) when convergence is
 *        not reached: `true` (default) — an exception, `false` — a result with
 *        `converged = false` and the attained residual in [SolutionFunc.residual].
 *        Direct schemes are unaffected. The parameter is set at the solver level rather than
 *        per method: it is an error-handling policy, not a property of a scheme.
 */
public abstract class SecondKindSolverCore<Operand>(
    public val basis: MinimalSplineBasis,
    public val funcs: FunctionalFamily,
    public val cL: Double,
    rhs: RhsWithDerivatives,
    public val throwOnDivergence: Boolean,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // The functional family solves linear systems while being built — with the same backend as the solver.
        NumericsContext.requireSame("SecondKindSolverCore", ctx, "funcs", funcs.ctx)
    }

    // The triple is UNPACKED into dedicated fields once, in the constructor.
    // This keeps every read on the hot path (assembly of M/M2, the Sloan and Kulkarni
    // iterations) exactly the same single field dereference as before the parameters
    // were merged: `rhs.value(t)` would add a second dereference on EVERY call.
    //
    // `rhs` itself is NOT a property but merely a constructor parameter: otherwise three functions
    // would turn into six public members (`rhs.value` and `fEff` are one and the same),
    // and two ways to obtain the same thing invite mistakes. As a parameter it also
    // occupies no field in the object.

    /** Right-hand side `f(t)`. */
    public val fEff: (Double) -> Double = rhs.value

    /** First derivative of the right-hand side `f'(t)`. */
    public val fEffDeriv: (Double) -> Double = rhs.deriv

    /** Second derivative of the right-hand side `f''(t)`. */
    public val fEffDeriv2: (Double) -> Double = rhs.deriv2

    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    public val dim: Int = n + 2

    // ==== Extension points ==================================================

    /** Equation name for diagnostic messages ("Fredholm" / "Volterra"). */
    protected abstract val equationName: String

    /**
     * Explanatory note for the divergence message of the Kulkarni scheme for quasi-interpolants.
     *
     * Made an extension point instead of being built from a common template because the
     * texts of the two solvers historically differ: the Fredholm one is longer
     * (it names the contraction condition explicitly). Diagnostics are part of the observable
     * behaviour, so merging the texts here would be a behavioural change.
     */
    protected abstract val kulkarniQuasiHint: String

    /**
     * Points at which the stopping criterion of the iterative schemes is measured.
     *
     * For Fredholm these are the global Gauss nodes of the operator: the iterate is already
     * computed there as a by-product of preparing the operand, so the criterion is free.
     * For Volterra there are no fixed nodes, and a separate uniform sample of
     * `4n+1` points is taken. The sets are DIFFERENT, and this matters: the same tolerance
     * `1e-13` applied to a different set yields a different iteration count.
     */
    protected abstract val checkPoints: DoubleArray

    /**
     * Prepares the function `u` for repeated application of the operator `L`.
     *
     * Fredholm: the values of `u` at the global Gauss nodes. Volterra: `u` itself
     * (preparation is impossible — the nodes depend on `t`).
     */
    protected abstract fun prepare(u: (Double) -> Double): Operand

    /** `(L u)(t)` from a prepared operand. */
    protected abstract fun image(o: Operand): (Double) -> Double

    /** `(L u)'(t)` from a prepared operand. */
    protected abstract fun imageDeriv(o: Operand): (Double) -> Double

    /**
     * `(L u)''(t)` from a prepared operand.
     *
     * @param uD derivative of the OPERAND itself. For Volterra it is mandatory: because of
     *        the variable upper limit the second derivative contains the Leibniz term
     *        `K(t,t) u'(t)`. For Fredholm the limits are constant, there is no such term,
     *        and the argument is not read — see the override in the Fredholm solver.
     */
    protected abstract fun imageDeriv2(o: Operand, uD: (Double) -> Double): (Double) -> Double

    /**
     * Values of the iterate at [checkPoints] — what the stopping criterion is measured on.
     *
     * An extension point rather than a plain formula: for Fredholm the check points
     * COINCIDE with the nodes used to prepare the operand, so the values are already computed
     * and it suffices to return the operand itself. A common formula would double the cost of
     * the stopping criterion there on EVERY iteration: `u` is not a constant there but the result
     * of applying the operator.
     */
    protected open fun checkValues(u: (Double) -> Double, o: Operand): DoubleArray =
        DoubleArray(checkPoints.size) { u(checkPoints[it]) }

    /** `(\mathcal K u)(t)` — application of the operator to an UNPREPARED function. */
    protected abstract fun applyOperator(t: Double, u: (Double) -> Double): Double

    /** `d/dt (\mathcal K u)(t)`. */
    protected abstract fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double

    /**
     * `d^2/dt^2 (\mathcal K u)(t)`.
     *
     * @param uD derivative of `u`; needed by Volterra only (the Leibniz term), ignored by
     *        Fredholm. The arity follows the "wider" variant: it cannot be narrowed,
     *        while the extra argument costs Fredholm nothing.
     */
    protected abstract fun applyOperatorDeriv2(
        t: Double,
        u: (Double) -> Double,
        uD: (Double) -> Double,
    ): Double

    /** `L omega_i` and its derivatives for column `i` of the matrix `M`. */
    protected abstract fun omegaImages(i: Int): ImageTriple

    /** `L(L omega_i)` and its derivatives for column `i` of the matrix `M2`. */
    protected abstract fun doubleOmegaImages(i: Int): ImageTriple

    // ==== Shared part =======================================================

    /** chi_j(g) from the values of g, g' and g'' (a wrapper). */
    private fun chiOf(
        g: (Double) -> Double,
        gD: (Double) -> Double,
        gDD: (Double) -> Double = { 0.0 },
    ): DoubleArray = DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD, gDD) }

    /**
     * Assembly of the matrix `chi_j(<image>_i)` column by column, followed by a transposition.
     *
     * The transposition was deliberately moved out of the parallel part: the columns
     * are independent in `i`, whereas writing rows into a shared matrix from several threads
     * would require synchronization. The traversal order of the copy is preserved literally.
     */
    private fun assembleChiMatrix(images: (Int) -> ImageTriple): DenseMatrix {
        // The columns of M are independent in i; cols[i] = column i.
        val cols = ParallelAssembly.assembleRows(dim, dim, ctx.parallel) { i ->
            val im = images(i)
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(im.value, im.deriv, im.deriv2) }
        }
        // DenseMatrix is stored COLUMN-MAJOR, so the assembled columns are simply laid out
        // one after another: the transposition required by row-major storage
        // disappears together with it. Not a single arithmetic operation is performed while copying.
        val data = DoubleArray(dim * dim)
        for (i in 0 until dim) cols[i].copyInto(data, i * dim)
        return DenseMatrix.fromColumnMajor(dim, dim, data)
    }

    /** Matrix M_{j,i} = chi_j(L omega_i). For xi it accounts for (L omega_i)', for xi^<0> also for (L omega_i)''. */
    public fun matrixM(): DenseMatrix = assembleChiMatrix { i -> omegaImages(i) }

    /** Matrix M2_{j,i} = chi_j(L(L omega_i)) (a double application of L). */
    public fun matrixM2(): DenseMatrix = assembleChiMatrix { i -> doubleOmegaImages(i) }

    /** g_j = chi_j(f). */
    public fun vectorG(): DoubleArray = chiOf(fEff, fEffDeriv, fEffDeriv2)

    /**
     * d_j = chi_j(L f).
     *
     * IMPORTANT: the image of the right-hand side is built through a DIRECT application of the
     * operator ([applyOperator]), not through [prepare] + [image]. For Volterra this means
     * that no integrand cache is created here — the right-hand side is applied
     * exactly once per functional, and a cache would only add allocations.
     * The fact is recorded in the KDoc of `VolterraOperator.IntegrandCache` and must not be changed.
     */
    public fun vectorD(): DoubleArray {
        val rhsImage = { t: Double -> cL * applyOperator(t) { s -> fEff(s) } }
        val rhsImageDeriv = { t: Double -> cL * applyOperatorDeriv(t) { s -> fEff(s) } }
        // The second derivative of the right-hand side image needs both f and f' (the Leibniz term for Volterra).
        val rhsImageDeriv2 = { t: Double -> cL * applyOperatorDeriv2(t, fEff, fEffDeriv) }
        return chiOf(rhsImage, rhsImageDeriv, rhsImageDeriv2)
    }

    /**
     * Matrix of the base scheme `I - M` — the very one the linear system in
     * [solveBaseCoeffs] is solved with.
     *
     * WHY IT IS PUBLIC. Without it there was no way to measure the conditioning of the
     * assembled system from the outside: [matrixM] gives only `M`, while the system is solved
     * with `I - M`, and rebuilding it on the caller side would duplicate the formula
     * of the scheme and risk diverging from it.
     *
     * The order of operations is preserved LITERALLY: extracting this method from
     * [solveBaseCoeffs] is a pure move of lines, it changes no numbers.
     */
    public fun baseMatrix(): DenseMatrix {
        val m = matrixM()
        val a = DenseMatrix.zeros(dim, dim)
        for (r in 0 until dim) { for (c in 0 until dim) a[r, c] = -m[r, c]; a[r, r] += 1.0 }
        return a
    }

    /** Base scheme: (I - M) c = g. */
    public fun solveBaseCoeffs(): DoubleArray =
        LinearAlgebra.solve(baseMatrix(), vectorG(), ctx.backend)

    public fun base(): SolutionFunc {
        val c = solveBaseCoeffs()
        return SolutionFunc(eval = { t -> basis.evalSpline(c, t) })
    }

    /** Sloan: ~u_h(t) = f(t) + (L u_h)(t). u_h is a spline, L is applied to a prepared operand. */
    public fun sloan(): SolutionFunc {
        val c = solveBaseCoeffs()
        val splineImage = image(prepare { s -> basis.evalSpline(c, s) })
        return SolutionFunc(eval = { t -> fEff(t) + splineImage(t) })
    }

    /**
     * Kulkarni for projectors (theta, xi): (I - M - M2 + M^2) c = (I - M) g + d;
     * u_h^K = y_h + (I - P_chi)[f + L y_h]. For quasi-interpolants (mu, lambda) there is no
     * reduction — a direct iteration with the finite-rank U^K_h [numerical observation].
     */
    public fun kulkarni(): SolutionFunc {
        return if (funcs.isProjector) kulkarniProjector() else kulkarniQuasi()
    }

    private fun kulkarniProjector(): SolutionFunc {
        val m = matrixM(); val m2 = matrixM2(); val g = vectorG(); val d = vectorD()
        val mm = LinearAlgebra.matMat(m, m, ctx.backend)
        // A = I - M - M2 + M^2
        val a = DenseMatrix.zeros(dim, dim)
        for (r in 0 until dim) {
            for (c in 0 until dim) a[r, c] = -m[r, c] - m2[r, c] + mm[r, c]
            a[r, r] += 1.0
        }
        // rhs = (I - M) g + d
        val mg = LinearAlgebra.matVec(m, g, ctx.backend)
        val rhs = DoubleArray(dim) { g[it] - mg[it] + d[it] }
        val c = LinearAlgebra.solve(a, rhs, ctx.backend) // coefficients of y_h
        // u_h^K = y_h + (I - P_chi)[f + L y_h]; (I - P_chi)w = w - P_chi w.
        val yh = { s: Double -> basis.evalSpline(c, s) }
        val yhD = { s: Double -> basis.evalSplineDeriv(c, s) }
        val yhOperand = prepare(yh)
        val yhImage = image(yhOperand)
        val yhImageDeriv = imageDeriv(yhOperand)
        val yhImageDeriv2 = imageDeriv2(yhOperand, yhD)
        val wFun = { t: Double -> fEff(t) + yhImage(t) }
        val wDFun = { t: Double -> fEffDeriv(t) + yhImageDeriv(t) }
        val wDDFun = { t: Double -> fEffDeriv2(t) + yhImageDeriv2(t) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun, wDDFun)
        return SolutionFunc(eval = { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) })
    }

    /**
     * Kulkarni for mu, lambda: the iteration u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)).
     * [numerical observation]: solvability/convergence are not guaranteed (P^2=P does not hold).
     *
     * The iterate is stored as a CONTINUOUS function u^{(m)}(t): a reconstruction of the same
     * order as the basis (through basis.evalSpline and the exact operator quadrature),
     * without an order-reducing piecewise-linear interpolation of nodal values. Previously the
     * iterate was stored as values on a uniform sample with piecewise-linear reconstruction,
     * which capped the accuracy at O(h_sample^2) REGARDLESS of the order of the basis.
     *
     * The check points ([checkPoints]) take part ONLY in the stopping criterion,
     * they are not used in the iteration itself.
     */
    private fun kulkarniQuasi(): SolutionFunc {
        var uFun: (Double) -> Double = { t -> fEff(t) }
        var uOperand = prepare(uFun)
        var uAtCheck = checkValues(uFun, uOperand)
        val stop = IterationStopCriterion(KULKARNI_QUASI_TOLERANCE)
        while (stop.performedIterations < KULKARNI_QUASI_MAX_ITERATIONS) {
            val curFun = uFun
            val curOperand = uOperand
            // P_chi u: the coefficients chi_j(u) from the continuous u^{(m)} (preserves the order).
            val pc = funcs.projectorCoeffs(curFun)
            val pcFun = { s: Double -> basis.evalSpline(pc, s) }
            val luFun = image(curOperand)                 // L u
            val pLu = funcs.projectorCoeffs(luFun)        // P_chi(L u)
            val lpu = image(prepare(pcFun))               // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(lpu)         // P_chi(L(P_chi u))
            // Continuous reconstruction of the next iterate u^{(m+1)}(t).
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(pLu, t) + lpu(t) - basis.evalSpline(pLPu, t)
            }
            val nextOperand = prepare(nextFun)
            val nextAtCheck = checkValues(nextFun, nextOperand)
            var diff = 0.0
            for (k in nextAtCheck.indices) diff = maxOf(diff, abs(nextAtCheck[k] - uAtCheck[k]))
            uFun = nextFun
            uOperand = nextOperand
            uAtCheck = nextAtCheck
            if (stop.accept(diff)) break
        }
        // Previously a non-converged iterate was returned SILENTLY: telling it apart from a
        // correct result was impossible. Now the single contract [reportConvergence] applies.
        reportConvergence(
            converged = stop.converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Kulkarni scheme for the quasi-interpolant '${funcs.name}' ($equationName)",
            iterations = stop.performedIterations,
            maxIterations = KULKARNI_QUASI_MAX_ITERATIONS,
            residual = stop.residual,
            tolerance = KULKARNI_QUASI_TOLERANCE,
            hint = kulkarniQuasiHint,
            diverged = stop.diverged,
        )
        val finalFun = uFun
        return SolutionFunc(
            eval = { t -> finalFun(t) },
            converged = stop.converged,
            iterations = stop.performedIterations,
            residual = stop.residual,
        )
    }

    /**
     * Iterated Kulkarni: ^u_h^K = f + L u_h^K.
     *
     * The convergence flag is INHERITED from [kulkarni]: the Sloan iteration itself is a single
     * integration without iterations, but its result is meaningful only when the
     * underlying approximation is meaningful.
     */
    public fun iteratedKulkarni(): SolutionFunc {
        val kulkarniSolution = kulkarni()
        val kulkarniImage = image(prepare { s -> kulkarniSolution.eval(s) })
        return SolutionFunc(
            eval = { t -> fEff(t) + kulkarniImage(t) },
            converged = kulkarniSolution.converged,
            iterations = kulkarniSolution.iterations,
            residual = kulkarniSolution.residual,
        )
    }
}
