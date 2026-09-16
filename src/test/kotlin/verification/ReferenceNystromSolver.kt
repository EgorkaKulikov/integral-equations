package verification

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos

/**
 * A BASELINE Nyström method FOR CROSS-CHECKING. NOT PART OF THE LIBRARY.
 *
 * The purpose and why it exists separately
 * ------------------------------------------
 * All the regular checks of the project are written on its own code and therefore confirm only
 * internal consistency. This class is an independent implementation of the classical
 * Nyström method (a textbook scheme, [Atkinson 1997], ch. 4) for the Fredholm equation
 * of the second kind
 *
 *     u(t) - lambda * ∫_a^b K(t,s) u(s) ds = f(t),
 *
 * which serves as an EXTERNAL baseline inside the test run: the solutions of the project
 * schemes are cross-checked against it without going out to Python (the cross-check with SciPy is a separate loop,
 * `tools/verify_with_scipy.py`, the layers L6a/L6b).
 *
 * INDEPENDENCE IS THE MAIN REQUIREMENT, not a wish
 * ---------------------------------------------------
 * A baseline using the code under test closes on itself: an error lying in the
 * link common to both will cancel and become invisible. Therefore nothing from `src/main`
 * is DELIBERATELY used here:
 *
 *  - there is NO spline basis (`MinimalSplineBasis`) and no approximation functionals
 *    (`ProjFunctionals`, `AveragingFunctionals`) — the Nyström method does not require them
 *    in principle: the unknowns are the values of the solution at the quadrature nodes;
 *  - there is NO `numerics.GaussLegendre` — the nodes and weights are computed here ([legendreNodes]);
 *  - there is NO `LinearAlgebra`/`DenseOps`/project backends — the linear system is solved here
 *    ([solveDense]) by Gaussian elimination with column pivoting;
 *  - there is NO `Grid`: the grid of the project has nothing to do with the Nyström scheme.
 *
 * The only input from outside is `kernel`, `rhs`, `lambda`, `a`, `b` themselves: this is the STATEMENT
 * of the problem, not a means of solving it. Formally the independence is confirmed by the fact that
 * the import list of the file consists only of `kotlin.math`.
 *
 * The attainable accuracy
 * -------------------
 * On smooth kernels the Gauss-Legendre quadrature converges exponentially, so already
 * at [nodeCount] = 64 the solution is reproduced at the level of 1e-14…1e-15 — orders of magnitude
 * more accurate than the project schemes on the grids n = 8..32. This is what makes it a suitable baseline:
 * the quantity being measured (the error of a project scheme) is certainly larger than the error
 * of the baseline itself.
 *
 * @param kernel the kernel `K(t,s)`.
 * @param rhs the right-hand side `f(t)`.
 * @param lambda the parameter at the integral operator.
 * @param a the left bound of the interval.
 * @param b the right bound of the interval.
 * @param nodeCount the number of quadrature nodes (>= 2).
 */
class ReferenceNystromSolver(
    private val kernel: (Double, Double) -> Double,
    private val rhs: (Double) -> Double,
    private val lambda: Double = 1.0,
    private val a: Double = 0.0,
    private val b: Double = 1.0,
    private val nodeCount: Int = 64,
) {
    init {
        require(nodeCount >= 2) { "ReferenceNystromSolver: nodeCount >= 2 is required, got $nodeCount" }
        require(b > a) { "ReferenceNystromSolver: b > a is required, got a=$a, b=$b" }
    }

    /** The quadrature nodes and weights, mapped from `[-1,1]` onto `[a,b]`. */
    private val nodes: DoubleArray
    private val weights: DoubleArray

    /** The values of the solution at the quadrature nodes — the unknowns of the Nyström system. */
    private val nodeValues: DoubleArray

    init {
        val (raw, rawWeights) = legendreNodes(nodeCount)
        val half = 0.5 * (b - a)
        val mid = 0.5 * (a + b)
        nodes = DoubleArray(nodeCount) { mid + half * raw[it] }
        weights = DoubleArray(nodeCount) { half * rawWeights[it] }
        // The Nyström system: (I - lambda * w_j * K(t_i, t_j)) u = f.
        val matrix = Array(nodeCount) { i ->
            DoubleArray(nodeCount) { j ->
                (if (i == j) 1.0 else 0.0) - lambda * weights[j] * kernel(nodes[i], nodes[j])
            }
        }
        nodeValues = solveDense(matrix, DoubleArray(nodeCount) { rhs(nodes[it]) })
    }

    /**
     * The value of the baseline solution at an arbitrary point [t].
     *
     * The Nyström formula is used (interpolation by the equation itself rather than by a spline):
     * `u(t) = f(t) + lambda * sum_j w_j K(t, t_j) u_j`. It preserves the accuracy of the
     * quadrature outside the nodes too, so a separate interpolation is not needed.
     */
    fun eval(t: Double): Double {
        var sum = 0.0
        for (j in 0 until nodeCount) sum += weights[j] * kernel(t, nodes[j]) * nodeValues[j]
        return rhs(t) + lambda * sum
    }

    companion object {

        /**
         * The integral `∫_a^b f(s) ds` by an independent Gauss-Legendre quadrature.
         *
         * Extracted into a public method because the right-hand side of a model problem
         * `f = u - Ku` also contains an integral, and taking it with the project operator would mean
         * drawing the code under test into the baseline through a back door.
         *
         * On smooth integrands 96 nodes give machine accuracy.
         */
        fun integrate(
            a: Double,
            b: Double,
            nodeCount: Int = 96,
            f: (Double) -> Double,
        ): Double {
            require(nodeCount >= 2) { "integrate: nodeCount >= 2 is required, got $nodeCount" }
            require(b > a) { "integrate: b > a is required, got a=$a, b=$b" }
            val (raw, rawWeights) = legendreNodes(nodeCount)
            val half = 0.5 * (b - a)
            val mid = 0.5 * (a + b)
            var sum = 0.0
            for (i in 0 until nodeCount) sum += half * rawWeights[i] * f(mid + half * raw[i])
            return sum
        }

        /**
         * The nodes and weights of the Gauss-Legendre quadrature on `[-1,1]`.
         *
         * The algorithm: Newton's method on the zeros of the Legendre polynomial `P_m`, whose values and
         * derivative are taken from the three-term recurrence
         * `(k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}`, the weight `w = 2 / ((1-x^2) P'_m(x)^2)`.
         * The initial approximation is the Tricomi asymptotics `cos(pi (i - 1/4)/(m + 1/2))`.
         *
         * Implemented HERE and not taken from `numerics.GaussLegendre` exactly so that
         * the baseline does not depend on the code under test: the quadrature nodes enter the
         * project schemes too, so a common error in them would cancel in the comparison.
         */
        fun legendreNodes(m: Int): Pair<DoubleArray, DoubleArray> {
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until m) {
                var x = cos(PI * (i + 0.75) / (m + 0.5))
                var derivative = 0.0
                // Newton converges quadratically; 100 iterations is a certain margin,
                // in fact 4-5 are enough. The limit is mandatory: without it
                // an error in the formula would give an infinite loop instead of a clear failure.
                var converged = false
                for (iteration in 0 until 100) {
                    var previous = 1.0
                    var current = x
                    for (k in 1 until m) {
                        val next = ((2 * k + 1) * x * current - k * previous) / (k + 1)
                        previous = current
                        current = next
                    }
                    // current = P_m(x), previous = P_{m-1}(x).
                    derivative = m * (x * current - previous) / (x * x - 1.0)
                    val step = current / derivative
                    x -= step
                    if (abs(step) < 1e-15) {
                        converged = true
                        break
                    }
                }
                check(converged) { "legendreNodes: Newton's method did not converge for m=$m, node $i" }
                nodes[i] = x
                weights[i] = 2.0 / ((1.0 - x * x) * derivative * derivative)
            }
            return nodes to weights
        }

        /**
         * The solution of a dense system `A x = rhs` by Gaussian elimination with column
         * pivoting.
         *
         * Implemented HERE for the same reason as the quadrature: using the project's
         * `LinearAlgebra` would make the baseline dependent on the code under test.
         * The Nyström matrix on a smooth kernel is well conditioned (of order units),
         * so simple partial pivoting is enough — no refinements are needed.
         */
        fun solveDense(matrix: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
            val size = rhs.size
            val a = Array(size) { matrix[it].copyOf() }
            val x = rhs.copyOf()
            for (column in 0 until size) {
                var pivot = column
                for (row in column + 1 until size) {
                    if (abs(a[row][column]) > abs(a[pivot][column])) pivot = row
                }
                check(abs(a[pivot][column]) > 0.0) {
                    "solveDense: the matrix is singular, zero column $column"
                }
                if (pivot != column) {
                    val rowTmp = a[pivot]; a[pivot] = a[column]; a[column] = rowTmp
                    val valueTmp = x[pivot]; x[pivot] = x[column]; x[column] = valueTmp
                }
                for (row in column + 1 until size) {
                    val factor = a[row][column] / a[column][column]
                    if (factor == 0.0) continue
                    for (k in column until size) a[row][k] -= factor * a[column][k]
                    x[row] -= factor * x[column]
                }
            }
            for (row in size - 1 downTo 0) {
                var sum = x[row]
                for (k in row + 1 until size) sum -= a[row][k] * x[k]
                x[row] = sum / a[row][row]
            }
            return x
        }
    }
}
