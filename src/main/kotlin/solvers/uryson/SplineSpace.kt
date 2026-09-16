package solvers.uryson

import kotlin.math.abs
import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext

/**
 * Spline space: weights, Gram matrix and integral characteristics of the basis.
 *
 * Combines basis, grid and quadrature, providing the quantities shared by the schemes of
 * the second and the first kind: discrete-norm weights, Nyström weights and the matrix
 * of the Tikhonov stabilizer.
 *
 * @param basis basis of minimal splines.
 * @param quad quadrature for integrals over grid intervals.
 */
public class SplineSpace(
    public val basis: MinimalSplineBasis,
    public val quad: GaussLegendre,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    public val dim: Int = n + 2

    /**
     * Tolerance for discarding nodes that coincide with the ends of a subinterval.
     *
     * The value comes from the SINGLE SOURCE [Grid.breakpointInclusionEps] (which also carries
     * the rationale for making it relative and the caveat about short intervals) instead of being
     * recomputed here: the same formula used to be duplicated in `VolterraOperator`, and written
     * differently there (`coerceAtLeast` versus `maxOf`).
     *
     * DECLARED BEFORE [gramRInternal] ON PURPOSE: in Kotlin property initializers run in
     * declaration order, and `buildGram()` calls [subBreakpoints], which reads this threshold.
     * Declared AFTER, the threshold would still be `0.0`, and the Gram matrix would be built
     * with a DIFFERENT partition — a silent change of numbers with no error.
     */
    private val breakpointEps: Double = grid.breakpointInclusionEps

    /**
     * Discrete-norm weights `w_j = (x_{j+3} - x_j)/3`, `j = -2..n-1` (array index `j+2`).
     *
     * They are proportional to the support length of the spline `omega_j`; their sum equals the
     * length of the interval, which makes the discrete norm consistent with `L^2`.
     */
    private val weightsInternal: DoubleArray = DoubleArray(dim) { (grid.x(it - 2 + 3) - grid.x(it - 2)) / 3.0 }

    /**
     * Discrete-norm weights (A COPY: mutating the result does not affect the space).
     *
     * The field is COLD: all callers read it once and store it in a field of their own
     * (see `TikhonovSolver`), so the copy never enters a hot loop.
     */
    public val weights: DoubleArray get() = weightsInternal.copyOf()

    /** Nyström weights `W_j = \int_a^b omega_j(s) ds`. */
    private val wIntInternal: DoubleArray = DoubleArray(dim) { k ->
        val j = k - 2
        quad.integrate(grid.breakpoints) { t -> basis.omega(j, t) }
    }

    /**
     * Nyström weights (A COPY, see the rationale at [weights]).
     *
     * The only production reader is `UrysonSecondKindSolver.nystrom`, where the value is taken
     * into a local variable BEFORE the Newton loop.
     */
    public val wInt: DoubleArray get() = wIntInternal.copyOf()

    /**
     * Gram matrix of the stabilizer: `[R]_{i,j} = \int (omega_i omega_j + omega_i' omega_j') ds`.
     *
     * This is the inner-product matrix of the Sobolev space `W^{1,2}`, i.e. the Tikhonov
     * stabilizer penalizes both the function itself and its first derivative.
     * The matrix is symmetric, positive definite and banded: `|i-j| <= 2`, because the supports
     * of splines further apart do not overlap.
     */
    private val gramRInternal: DenseMatrix = buildGram()

    /**
     * Gram matrix of the stabilizer (A COPY, see the rationale at [weights]).
     *
     * In [DenseMatrix] the values live in ONE flat array, so the distinction between
     * "shallow/deep copy" disappears here together with the per-row arrays:
     * `copy()` copies the whole content, and write-through into the field is gone.
     */
    public val gramR: DenseMatrix get() = gramRInternal.copy()

    private fun buildGram(): DenseMatrix {
        val r = DenseMatrix.zeros(dim, dim)
        for (ki in 0 until dim) {
            val i = ki - 2
            for (kj in ki until dim) {
                val j = kj - 2
                if (abs(i - j) > 2) continue // banded structure: the supports do not overlap
                val lo = maxOf(grid.x(i), grid.x(j))
                val hi = minOf(grid.x(i + 3), grid.x(j + 3))
                if (hi <= lo) continue
                val sub = subBreakpoints(lo, hi)
                val value = quad.integrate(sub) { t ->
                    basis.omega(i, t) * basis.omega(j, t) +
                        basis.omegaDeriv(i, t) * basis.omegaDeriv(j, t)
                }
                r[ki, kj] = value
                r[kj, ki] = value
            }
        }
        return r
    }

    /** Grid nodes inside `[lo, hi]` plus the ends — the partition for the composite quadrature. */
    private fun subBreakpoints(lo: Double, hi: Double): DoubleArray {
        val pts = ArrayList<Double>()
        pts.add(lo)
        for (k in 0..n) {
            val xk = grid.x(k)
            if (xk > lo + breakpointEps && xk < hi - breakpointEps) pts.add(xk)
        }
        pts.add(hi)
        return pts.toDoubleArray()
    }

    /** Sum of the weights `sum_j w_j`; by construction it must equal the interval length `b - a`. */
    public fun weightsSum(): Double = weightsInternal.sum()

    /** Quadratic form of the stabilizer `Omega(x_h) = c^T R_h c`. */
    public fun omegaReg(c: DoubleArray): Double {
        // Inside the class the backing field is read directly: a copy would be pointless here.
        val rc = LinearAlgebra.matVec(gramRInternal, c, ctx.backend)
        var s = 0.0
        for (i in c.indices) s += c[i] * rc[i]
        return s
    }

}
