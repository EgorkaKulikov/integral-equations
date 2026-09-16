package solvers.volterra

import java.util.concurrent.atomic.AtomicReferenceArray
import numerics.*
import splines.*
import splines.functionals.*
import splines.metrics.*

/**
 * The kernel `K(t,s)` of a linear Volterra equation together with its analytic partial
 * derivatives.
 *
 * @param k the kernel `K(t,s)` itself.
 * @param kT the derivative `K_t(t,s)`; required by the de Boor–Fix functional families
 *        `xi^<1>`, `xi^<2>`, and by the reduction of a first-kind equation.
 * @param kS the derivative `K_s(t,s)`; it enters the diagonal term of the second derivative
 *        of the image `(V u)''` and is needed by the family `xi^<0>`. Unlike the Fredholm
 *        equation, here it really is used: because of the variable upper
 *        limit the differentiation touches the diagonal `K(t,t)`.
 * @param kTT the second derivative `K_tt(t,s)`; required by the family `xi^<0>`.
 *
 * The default values are zero and are admissible ONLY when the corresponding
 * derivative is indeed identically zero, or when the chosen family of
 * functionals does not use it: otherwise the system is built incorrectly without any
 * diagnostics.
 */
public class KernelV(
    public val k: (Double, Double) -> Double,
    public val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kS: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
)

/**
 * Volterra operator: (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds, quadrature over [a,t].
 *
 * Because of the VARIABLE upper limit (unlike Fredholm) a fixed set of
 * global Gauss nodes is unusable: the integration domain depends on t.
 * The integral is therefore computed directly (by closures) over a composite partition of [a,t]
 * (the grid breakpoints falling into (a,t), plus the ends a and t).
 *
 * The derivative by the Leibniz rule (for the xi functionals and for the metric):
 *   d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds.
 * The BOUNDARY term K(t,t) u(t) is specific to Volterra (Fredholm has none).
 */
public class VolterraOperator(public val kernel: KernelV, public val grid: Grid, public val quad: GaussLegendre) {
    /** Left end of the interval — the lower integration limit in all formulas. */
    public val a: Double = grid.a

    /**
     * Breakpoint inclusion tolerance: a grid breakpoint counts as strictly inside (a, t) if `x < t - eps`.
     *
     * The value comes from the SINGLE SOURCE [Grid.breakpointInclusionEps] (which also carries
     * the rationale for its relative form and the caveat about small intervals) rather than being
     * recomputed here: previously the same formula was duplicated in `SplineSpace`.
     *
     * It is read by all three selection sites ([subBreakpoints], the cached [apply],
     * [integrateRange]) — this is not style but a CORRECTNESS REQUIREMENT: the cached and
     * uncached paths must select the SAME number of full cells, otherwise the
     * invariant of [IntegrandCache] breaks.
     */
    public val breakpointInclusionEps: Double = grid.breakpointInclusionEps

    /** Composite partition of [a, t]: the interior grid breakpoints < t, then t itself. */
    private fun subBreakpoints(t: Double): DoubleArray {
        val bp = grid.breakpoints
        val list = ArrayList<Double>()
        for (x in bp) { if (x < t - breakpointInclusionEps) list.add(x) else break }
        if (list.isEmpty()) list.add(a)
        list.add(t)
        return list.toDoubleArray()
    }

    /** (\mathcal V u)(t) = \int_a^t K(t,s) u(s) ds for an arbitrary u(s). */
    public fun apply(t: Double, u: (Double) -> Double): Double {
        // numerical-core 1.0.0 rejects non-numeric partition points with an exception; the operator
        // contract is to propagate a NaN argument, exactly as the cached path does.
        if (t.isNaN()) return Double.NaN
        if (t <= a) return 0.0
        return quad.integrate(subBreakpoints(t)) { s -> kernel.k(t, s) * u(s) }
    }

    /**
     * Gauss nodes of the FULL grid cells: `cellNodes[c][q]` is the q-th node of the composite
     * quadrature on the cell `[x_c, x_{c+1}]`.
     *
     * The key observation: when integrating over `[a,t]`, only the last, truncated cell
     * `[x_k, t]` depends on `t`; the nodes of all full cells are the same for every `t`.
     * The formula here repeats [GaussLegendre.integrate] literally (`half`, `mid`,
     * `mid + half * refNodes[q]`) on the same arguments, so the nodes coincide
     * BIT FOR BIT with the values the quadrature itself would compute.
     *
     * INVARIANT (important): this is a SNAPSHOT materialized once, whereas
     * the uncached path [apply] reads `grid.breakpoints` on every call.
     * The two paths are therefore equivalent PROVIDED the contents of
     * `grid.breakpoints` do not change after `cellNodes` is first touched.
     *
     * How that condition is protected: [Grid.breakpoints] is a HOT field, so it
     * DELIBERATELY does not return a copy (copying on each of tens of thousands of
     * accesses would defeat the very purpose of the cache). Instead an explicitly
     * documented "read-only" convention applies (see the KDoc of [Grid.breakpoints]), and the project
     * contains not a single write into that array. The cold arrays of the neighbouring APIs
     * (`GaussLegendre.refNodesWeights`, `SplineSpace.weights/wInt/gramR`) are handed out as copies.
     *
     * ON INITIALIZATION ORDER: the expression uses [refNodes], declared BELOW;
     * correctness is ensured PRECISELY by the laziness (by the time of the first access the
     * constructor has already finished). Replacing `by lazy` with eager initialization
     * without moving [refNodes] up would give a `NullPointerException`.
     */
    private val cellNodes: Array<DoubleArray> by lazy {
        val bp = grid.breakpoints
        Array(bp.size - 1) { c ->
            val lo = bp[c]
            val hi = bp[c + 1]
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            DoubleArray(refNodes.size) { q -> mid + half * refNodes[q] }
        }
    }

    /** Reference quadrature nodes/weights on [-1,1]: read in the hot loop without allocations. */
    private val refNodes: DoubleArray = quad.refNodesWeights().first
    private val refWeights: DoubleArray = quad.refNodesWeights().second

    /**
     * Cache of the integrand values `u(s)` at the nodes of the FULL grid cells.
     *
     * Its purpose is to remove the quadratic cost of applying the Volterra operator:
     * without the cache every call `apply(t, u)` at its own `t` recomputed `u`
     * at all `8 * (number of full cells)` nodes, even though the nodes themselves do not depend on `t`.
     *
     * The cache is TIGHTLY BOUND to a single function `u` (it is held in a field and cannot be another one),
     * so mixing up values between different integrands is impossible
     * by construction. Its size is bounded by the grid: `n * nodesPerSub` numbers (512 for n=64).
     *
     * The cache is ALSO BOUND TO THE OPERATOR that created it (see [IntegrandCache.owner] and
     * the check in [apply]). The check is needed precisely because the TYPE of the owner does not
     * distinguish them: in Kotlin an `inner class` has no type parameterized by the outer
     * INSTANCE, so the expression `op2.apply(t, op1.integrandCache(u))`
     * compiles without errors and, without the check, would silently give wrong numbers.
     *
     * Only a PURE `u` (deterministic, free of side effects) may be cached.
     * The actual consumers are ALL calls of [VolterraSecondKindSolver.applyL] (the basis `omega_i`
     * and their images in `matrixM`/`matrixM2`, the splines `evalSpline` in `sloan`/`kulkarni*`,
     * the iterates of `kulkarniQuasi`/`combinedNystrom`, the Nyström images), all of them pure.
     * NOTE: `vectorD` does NOT use the cache — it goes through the old overload
     * `apply(t, u)` and `applyDeriv`; `applyLDeriv`/`applyLDeriv2` are not cached either
     * (they have different kernels `kT`/`kTT`).
     *
     * THREAD SAFETY: the values of a cell are published as a whole array through
     * [AtomicReferenceArray], which gives correct publication (happens-before) without
     * locks. A race of two threads on the same cell is harmless: `u` is pure, hence both
     * compute bit-identical numbers, and the CAS winner decides whose array the others
     * will see. There is no serialization point in the hot loop — only a volatile read per cell.
     */
    public inner class IntegrandCache internal constructor(internal val u: (Double) -> Double) {
        /**
         * The operator instance that created this cache — the only one whose nodes match
         * the stored values.
         *
         * Why an explicit property rather than the implicit `inner class` reference: the implicit
         * reference to the outer instance is available only INSIDE the body of `IntegrandCache`
         * (`this@VolterraOperator`) and cannot be read from outside, i.e. from [apply], which is
         * where the check is needed. The reference is therefore captured in a field at creation time.
         * The cost is one reference per cache (not per call and not per node).
         */
        internal val owner: VolterraOperator = this@VolterraOperator

        private val cells = AtomicReferenceArray<DoubleArray>(cellNodes.size)

        /** Values of `u` at the nodes of the full cell `c`; computed on first access. */
        internal fun values(c: Int): DoubleArray {
            cells.get(c)?.let { return it }
            val nodes = cellNodes[c]
            val computed = DoubleArray(nodes.size) { q -> u(nodes[q]) }
            cells.compareAndSet(c, null, computed)
            return cells.get(c) ?: computed
        }
    }

    /** Creates a cache of nodal values for a SPECIFIC integrand. */
    public fun integrandCache(u: (Double) -> Double): IntegrandCache = IntegrandCache(u)

    /**
     * The same as [apply], but the values of `u` at the nodes of the full cells are taken from [cache].
     *
     * The arithmetic repeats [subBreakpoints] + [GaussLegendre.integrate] LITERALLY:
     * the same selection of partition points, the same traversal order (cells left to right, inside
     * a cell — nodes in increasing index order), the same `half`/`mid`, the same plain
     * accumulation `sum += half * w_q * (K(t,s_q) * u(s_q))` without factoring anything out.
     * There is exactly one difference: `u(s_q)` is not recomputed on the full cells.
     *
     * OWNERSHIP REQUIREMENT: [cache] must have been created by THIS very instance
     * of the operator. The check is performed ONCE on entry (a reference comparison, outside
     * any loop; its cost is negligible against `nodesPerSub * n` calls of `kernel.k`)
     * and computes nothing, so it does not affect the numbers. Without it a foreign cache would give
     * SILENTLY WRONG NUMBERS: the values of `u` would be taken at the nodes of ITS OWN grid, while the
     * kernel would be evaluated at the nodes of THIS operator's grid (or an
     * `IndexOutOfBoundsException` would occur if the owner's grid is coarser).
     *
     * @throws IllegalArgumentException if [cache] was created by another operator instance.
     */
    public fun apply(t: Double, cache: IntegrandCache): Double {
        require(cache.owner === this) {
            "the nodal value cache was passed to a VolterraOperator instance OTHER than the one " +
                "that created it. The cache stores the values u(s) at the Gauss nodes of ITS OWN " +
                "owner's grid, while the kernel here would be evaluated at the nodes of this operator's grid: " +
                "the grid nodes do NOT COINCIDE in general, and the result would be silently wrong " +
                "(or an IndexOutOfBoundsException would occur if the owner's grid is coarser). " +
                "Create the cache with the same operator you apply it on: " +
                "op.apply(t, op.integrandCache(u))."
        }
        if (t <= a) return 0.0
        val bp = grid.breakpoints
        // Number of leading grid breakpoints that fall into the partition (see subBreakpoints).
        var included = 0
        while (included < bp.size && bp[included] < t - breakpointInclusionEps) included++
        var sum = 0.0
        // Full cells [bp[c], bp[c+1]] — the nodes and the values of u are taken from the cache.
        for (c in 0 until included - 1) {
            val lo = bp[c]
            val hi = bp[c + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val nodes = cellNodes[c]
            val values = cache.values(c)
            for (q in refNodes.indices) {
                sum += half * refWeights[q] * (kernel.k(t, nodes[q]) * values[q])
            }
        }
        // Truncated cell [x_k, t] (or [a, t] if there are no grid breakpoints to the left of t).
        val lo = if (included == 0) a else bp[included - 1]
        // The condition is written as the NEGATION of `hi <= lo` from [GaussLegendre.integrate],
        // literally, rather than as `t > lo`: for finite `t` they are identical, but at `t = NaN`
        // both comparisons are false, so `t > lo` would give exactly 0.0, while the uncached
        // path (where `hi <= lo` is false as well and the computation continues) gives NaN. The variant with
        // `require(!t.isNaN())` was rejected: it would require editing the OLD `apply(t, u)`,
        // i.e. changing the behaviour of a public API out of scope (it is also called by the Uryson
        // solver, by `problems` and by the tests). For finite `t` the numbers do not change: the branch is the same.
        if (!(t <= lo)) {
            val half = 0.5 * (t - lo)
            val mid = 0.5 * (t + lo)
            for (q in refNodes.indices) {
                val s = mid + half * refNodes[q]
                sum += half * refWeights[q] * (kernel.k(t, s) * cache.u(s))
            }
        }
        return sum
    }

    /**
     * Integral int_lo^hi g(s) ds over the composite partition of [lo,hi] split by the grid breakpoints.
     * Used for the Nyström weights with the restriction to the compact support of omega_j
     * ([x_j,x_{j+3}]) -> at most three subintervals, which removes the O(n) cost per point.
     */
    public fun integrateRange(lo: Double, hi: Double, g: (Double) -> Double): Double {
        if (hi <= lo) return 0.0
        val list = ArrayList<Double>()
        list.add(lo)
        for (x in grid.breakpoints) if (x > lo + breakpointInclusionEps && x < hi - breakpointInclusionEps) list.add(x)
        list.add(hi)
        return quad.integrate(list.toDoubleArray(), g)
    }

    /**
     * d/dt (\mathcal V u)(t) = K(t,t) u(t) + \int_a^t dK/dt(t,s) u(s) ds (Leibniz).
     * IMPORTANT: the boundary term K(t,t)u(t) remains even at t=a (the integral over [a,a] is zero).
     * This term is critical for the V1->V2 reduction at the left end (g(a)=f'(a)/K(a,a)=u*(a)).
     */
    public fun applyDeriv(t: Double, u: (Double) -> Double): Double {
        if (t < a) return 0.0
        val boundary = kernel.k(t, t) * u(t)
        val integral = if (t.isNaN()) Double.NaN else if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kT(t, s) * u(s) }
        return boundary + integral
    }

    /**
     * Second derivative of the image:
     *
     *     (V u)''(t) = [2 K_t(t,t) + K_s(t,t)] u(t) + K(t,t) u'(t)
     *                  + \int_a^t K_tt(t,s) u(s) ds.
     *
     * The formula follows from applying the Leibniz rule once more to [applyDeriv]:
     * differentiating the boundary term `K(t,t) u(t)` gives `(K_t + K_s)(t,t) u(t)`
     * and `K(t,t) u'(t)`, while differentiating the integral gives one more term `K_t(t,t) u(t)`
     * plus the integral of `K_tt`.
     *
     * IMPORTANT: the term `K(t,t) u'(t)` is mandatory when `K(t,t) != 0`, which is why besides the
     * function itself its first derivative is passed as well. The diagonal term is kept
     * even at `t = a`, where the integral over `[a,a]` vanishes.
     *
     * @param u the function itself.
     * @param uD its first derivative.
     */
    public fun applyDeriv2(t: Double, u: (Double) -> Double, uD: (Double) -> Double): Double {
        if (t < a) return 0.0
        val kd = kernel.k(t, t)
        val diag = 2.0 * kernel.kT(t, t) + kernel.kS(t, t)
        val boundary = diag * u(t) + kd * uD(t)
        val integral = if (t.isNaN()) Double.NaN else if (t <= a) 0.0 else quad.integrate(subBreakpoints(t)) { s -> kernel.kTT(t, s) * u(s) }
        return boundary + integral
    }
}
