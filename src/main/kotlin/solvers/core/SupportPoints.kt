package solvers.core

import splines.functionals.ValueFunctional

/**
 * The set of DISTINCT support points of a family of value functionals together with an explicit
 * indexing by the pair "functional number, node number inside the functional".
 *
 * WHY. The Nyström schemes and the assembly of the collocation matrices work like this: the points of all
 * functionals `chi_j` are merged into a single set `{eta_r}`, and then for every node
 * `s_{j,q}` its index `r` in that set is needed. Previously the correspondence was looked up BY VALUE
 * through a `HashMap<Double, Int>` — i.e. it required a BIT-EXACT match of `Double`s.
 * It worked only because both filling the map and reading from it took the values from ONE
 * AND THE SAME array `ValueFunctional.nodes` of a cached functional: the match was
 * ensured by object identity, not by arithmetic. Any recomputation of the same
 * mathematical point by another (equivalent) route gives a discrepancy in the last bit,
 * and the lookup would fail with a `NoSuchElementException` — even though the problem would be well-posed.
 *
 * WHAT WAS DONE. The correspondence `(j, q) -> r` is computed ONCE while the set is assembled and
 * afterwards read by indices; there is no lookup by value at use time at all.
 * Coincidence of points of different functionals is decided by an EXPLICIT tolerance [mergeEps]
 * (see the factories) rather than by exact equality.
 *
 * THE ORDER OF THE POINTS is part of the contract, not an implementation detail: for Fredholm and Volterra
 * the order of summation of the aggregated weights depends on it, for Uryson so do the numbering of
 * the unknowns, the row order of the Newton matrix and the initial-guess vector. Hence there are
 * two factories, each reproducing exactly the order the replaced code had:
 * [byAscendingValue] (was `sortedSetOf`) and [byFirstOccurrence] (was `LinkedHashSet`).
 *
 * @property points the distinct support points. READ-ONLY BY CONVENTION: the contents must not be
 *   modified. A copy is deliberately not returned — this is HOT data: the array is read
 *   in the inner loops of the Nyström matrix assembly (about `P^2` accesses per system).
 */
public class SupportPoints private constructor(
    public val points: DoubleArray,
    private val indexByNode: Array<IntArray>,
) {
    /** Number of distinct support points. */
    public val size: Int get() = points.size

    /**
     * Index of the point `nodes[node]` of the functional [functional] in the set [points].
     *
     * @param functional ordinal number of the functional in the array passed to the factory
     *   (for families with the mathematical index `j = -2..n-1` it is `j + 2`).
     * @param node index of the node inside `ValueFunctional.nodes`.
     */
    public fun indexOf(functional: Int, node: Int): Int = indexByNode[functional][node]

    public companion object {
        /**
         * The set of points ordered BY ASCENDING VALUE (the Nyström schemes of Fredholm
         * and Volterra).
         *
         * Reproduces the former `sortedSetOf<Double>()`: the same set and the same order,
         * with a single difference — merging goes by the tolerance [mergeEps] rather than by exact
         * equality. The representative of a group of merged points is the SMALLEST of them
         * (on an exact match, as is the case now, the choice is immaterial and the result is bit-for-bit
         * the same as with `TreeSet`).
         *
         * @param mergeEps non-negative merging tolerance: the points `u <= v` count as one
         *   if `v - u <= mergeEps`. A meaningful value is `Grid.breakpointInclusionEps`.
         */
        public fun byAscendingValue(functionals: Array<ValueFunctional>, mergeEps: Double): SupportPoints {
            val slots = Slots(functionals)
            val clusters = slots.clusterByValue(mergeEps)
            val index = slots.emptyIndex()
            for (slot in slots.order) index[slots.funcOf[slot]][slots.nodeOf[slot]] = clusters.clusterOf[slot]
            return SupportPoints(clusters.representatives, index)
        }

        /**
         * The set of points in the ORDER OF FIRST OCCURRENCE during the traversal "functionals by
         * increasing index, inside them — nodes by increasing index" (the Uryson schemes).
         *
         * Reproduces the former `LinkedHashSet<Double>()`: the same set and the same order,
         * with the same single difference — merging by the tolerance [mergeEps]. The representative
         * of a group of merged points is the FIRST ENCOUNTERED one (this is also how `LinkedHashSet`
         * behaves, keeping the element added earlier).
         *
         * The insertion order is deliberately NOT replaced by sorting: for Uryson it fixes the
         * numbering of the unknowns and the row order of the Newton matrix.
         *
         * @param mergeEps non-negative merging tolerance (see [byAscendingValue]).
         */
        public fun byFirstOccurrence(functionals: Array<ValueFunctional>, mergeEps: Double): SupportPoints {
            val slots = Slots(functionals)
            val clusters = slots.clusterByValue(mergeEps)
            val index = slots.emptyIndex()
            // The clusters are numbered by increasing value; we renumber them in the order of
            // first occurrence during a LINEAR traversal of the slots (which is the insertion order).
            val renumbered = IntArray(clusters.representatives.size) { UNASSIGNED }
            val points = DoubleArray(clusters.representatives.size)
            var assigned = 0
            for (slot in 0 until slots.total) {
                val cluster = clusters.clusterOf[slot]
                if (renumbered[cluster] == UNASSIGNED) {
                    renumbered[cluster] = assigned
                    points[assigned] = slots.value[slot]
                    assigned++
                }
                index[slots.funcOf[slot]][slots.nodeOf[slot]] = renumbered[cluster]
            }
            return SupportPoints(points, index)
        }

        /** Marker "the cluster has not been assigned a number yet" in the renumbering of [byFirstOccurrence]. */
        private const val UNASSIGNED = -1

        /**
         * A flat expansion of all nodes of all functionals: a slot is a pair "functional, node".
         *
         * The slots are numbered in traversal order (functionals in increasing order, inside them — nodes
         * in increasing order), so a linear traversal of the slots is exactly the insertion order.
         */
        private class Slots(functionals: Array<ValueFunctional>) {
            val sizes: IntArray = IntArray(functionals.size) { functionals[it].nodes.size }
            val total: Int = sizes.sum()
            val value = DoubleArray(total)
            val funcOf = IntArray(total)
            val nodeOf = IntArray(total)

            init {
                var slot = 0
                for (j in functionals.indices) {
                    val nodes = functionals[j].nodes
                    for (q in nodes.indices) {
                        value[slot] = nodes[q]
                        funcOf[slot] = j
                        nodeOf[slot] = q
                        slot++
                    }
                }
            }

            /**
             * The slots ordered by increasing value.
             *
             * The sort is STABLE (`sortedBy` uses a stable object sort), so
             * on an exact match of values the order of the slots remains the insertion order —
             * this is what makes the representative of a group the same element `TreeSet` used to pick.
             */
            val order: List<Int> by lazy { (0 until total).sortedBy { value[it] } }

            fun emptyIndex(): Array<IntArray> = Array(sizes.size) { IntArray(sizes[it]) }

            /**
             * Splits the slots into groups of "identical" points in a single ascending pass.
             *
             * The comparison is made against the REPRESENTATIVE of the group rather than against the previous point:
             * otherwise a chain of pairwise close points could merge into a group much wider than
             * [mergeEps].
             */
            fun clusterByValue(mergeEps: Double): Clusters {
                require(mergeEps >= 0.0 && !mergeEps.isNaN()) {
                    "SupportPoints: the merging tolerance must be non-negative, got $mergeEps"
                }
                val clusterOf = IntArray(total)
                val representatives = DoubleArray(total)
                var count = 0
                for (slot in order) {
                    val v = value[slot]
                    if (count == 0 || v - representatives[count - 1] > mergeEps) {
                        representatives[count] = v
                        count++
                    }
                    clusterOf[slot] = count - 1
                }
                return Clusters(clusterOf, representatives.copyOf(count))
            }
        }

        /**
         * Result of the grouping: the group index for every slot and the representative of every
         * group (the groups are numbered by increasing value).
         */
        private class Clusters(val clusterOf: IntArray, val representatives: DoubleArray)
    }
}
