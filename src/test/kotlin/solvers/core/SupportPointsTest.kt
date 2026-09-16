package solvers.core

import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.functionals.ThreePointFunctionals
import splines.functionals.ValueFunctional
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals


/**
 * Tests of the indexing of the support points [SupportPoints] (item 8.2 of the plan).
 *
 * They answer two independent questions.
 *
 * 1. IS THE FRAGILITY REMOVED. Previously the index of a point was looked up BY VALUE in a
 *    `HashMap<Double, Int>`, that is, it required a BITWISE match of the `Double`s. A point
 *    recomputed by another — mathematically equivalent — path differs in the last
 *    bit, and the former scheme broke on it: [realFamiliesDisagreeBitwiseOnTheSameMidpoint].
 *    The new indexing by the pair "functional, node" plus merging by an explicit tolerance handles such
 *    a point: [samePointViaTwoPathsGetsSingleIndex].
 *
 * 2. DOES THE TOLERANCE WORK. The merging of points is governed by an explicit `mergeEps`: at a zero tolerance
 *    bitwise distinct points stay distinct ([zeroToleranceKeepsBitwiseDistinctPointsApart]),
 *    a chain of near points does not collapse beyond the tolerance
 *    ([chainOfNearPointsDoesNotCollapseBeyondTolerance]), a negative tolerance is rejected.
 */
@Tag("fast")
class SupportPointsTest {

    /**
     * The pair "one and the same point computed by two paths" is not invented but taken
     * from a real grid: the midpoint of the interval `[x_1, x_2]` of the grid `Grid.graded(8)`.
     *
     * `0.5*(p+q)` (this is how [ProjFunctionals.buildTheta] computes it) and `p + 0.5*(q-p)` (this is how
     * [AveragingFunctionals] and [ThreePointFunctionals] compute it at theta = 1/2) give DIFFERENT numbers,
     * differing by exactly 1 ulp. Both formulas are legitimate; the discrepancy is a property of IEEE-754.
     */
    private class UlpPair {
        val grid: Grid = Grid.graded(8)
        val p: Double = grid.x(1)
        val q: Double = grid.x(2)
        val viaAverage: Double = 0.5 * (p + q)
        val viaOffset: Double = p + 0.5 * (q - p)
    }

    @Test
    fun twoEquivalentMidpointFormulasDifferByOneUlp() {
        val pair = UlpPair()
        assertNotEquals(pair.viaAverage, pair.viaOffset, "the premise of the test: the paths differ bitwise")
        assertEquals(
            1.0,
            abs(pair.viaAverage - pair.viaOffset) / Math.ulp(pair.viaAverage),
            0.0,
            "the discrepancy of the paths must be exactly 1 ulp",
        )
    }

    /**
     * Both paths of computing the midpoint ARE USED IN THE PROJECT SIMULTANEOUSLY, and on a real
     * grid they give DIFFERENT numbers for one and the same interval.
     *
     * [ProjFunctionals] computes the midpoint as `0.5*(p+q)`, while [ThreePointFunctionals]
     * and [AveragingFunctionals] compute it as `p + theta*(q-p)`. Today this is safe only because
     * the schemes work with ONE family at a time: mixing two sources of points
     * would break the lookup by value. The test records that the problem is not invented.
     */
    @Test
    fun realFamiliesDisagreeBitwiseOnTheSameMidpoint() {
        val grid = Grid.graded(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        // theta_0: (x_0, mid(x_0,x_1), mid(x_1,x_2), mid(x_2,x_3), x_3) — the midpoint of [x_1,x_2] is no. 2.
        val theta0 = ProjFunctionals(basis).chi(0) as ValueFunctional
        // lambda_0: (x_1, x_1 + 1/2 (x_2 - x_1), x_2) — the same midpoint is no. 1.
        val lambda0 = ThreePointFunctionals(basis).chi(0) as ValueFunctional
        val fromTheta = theta0.nodes[2]
        val fromLambda = lambda0.nodes[1]
        assertEquals(1.0, abs(fromTheta - fromLambda) / Math.ulp(fromTheta), 0.0, "the discrepancy is exactly 1 ulp")
        assertFailsWith<NoSuchElementException>("the lookup by value does not find the same point") {
            HashMap<Double, Int>().apply { put(fromTheta, 0) }.getValue(fromLambda)
        }
        // The new indexing with the same two functionals works correctly.
        val mixed = SupportPoints.byFirstOccurrence(arrayOf(theta0, lambda0), grid.breakpointInclusionEps)
        assertEquals(mixed.indexOf(0, 2), mixed.indexOf(1, 1), "the midpoint of [x_1,x_2] is one point")
    }

    /**
     * AFTER THE FIX: the same pair of "recomputed" points merges into one, and both functionals
     * get ONE index — neither an exception nor a duplicate in the set.
     */
    @Test
    fun samePointViaTwoPathsGetsSingleIndex() {
        val pair = UlpPair()
        val eps = pair.grid.breakpointInclusionEps
        val byAverage = ValueFunctional(doubleArrayOf(pair.p, pair.viaAverage, pair.q), doubleArrayOf(1.0, 1.0, 1.0))
        val byOffset = ValueFunctional(doubleArrayOf(pair.viaOffset, pair.q), doubleArrayOf(1.0, 1.0))
        val functionals = arrayOf(byAverage, byOffset)

        for (support in listOf(
            SupportPoints.byAscendingValue(functionals, eps),
            SupportPoints.byFirstOccurrence(functionals, eps),
        )) {
            assertEquals(3, support.size, "points differing by 1 ulp must merge into one")
            assertEquals(
                support.indexOf(0, 1),
                support.indexOf(1, 0),
                "one and the same mathematical node must get one index",
            )
            assertEquals(support.indexOf(0, 2), support.indexOf(1, 1), "the node x_2 is shared by both functionals")
        }
    }

    /** Without a tolerance (`mergeEps = 0`) there is no merging: the tolerance really works and is not decorative. */
    @Test
    fun zeroToleranceKeepsBitwiseDistinctPointsApart() {
        val pair = UlpPair()
        val functionals = arrayOf(
            ValueFunctional(doubleArrayOf(pair.viaAverage), doubleArrayOf(1.0)),
            ValueFunctional(doubleArrayOf(pair.viaOffset), doubleArrayOf(1.0)),
        )
        assertEquals(2, SupportPoints.byAscendingValue(functionals, 0.0).size)
        assertEquals(1, SupportPoints.byAscendingValue(functionals, pair.grid.breakpointInclusionEps).size)
    }

    /** A negative tolerance is an error of the caller and not a silently ignored value. */
    @Test
    fun negativeToleranceRejected() {
        val functionals = arrayOf(ValueFunctional(doubleArrayOf(0.0), doubleArrayOf(1.0)))
        assertFailsWith<IllegalArgumentException> { SupportPoints.byAscendingValue(functionals, -1e-16) }
        assertFailsWith<IllegalArgumentException> { SupportPoints.byFirstOccurrence(functionals, Double.NaN) }
    }

    /**
     * A chain of pairwise near points does NOT collapse into one group of unbounded width:
     * the comparison goes against the REPRESENTATIVE of the group and not against the previous point.
     */
    @Test
    fun chainOfNearPointsDoesNotCollapseBeyondTolerance() {
        val eps = 1e-3
        val functionals = arrayOf(
            ValueFunctional(doubleArrayOf(0.0, 0.0009, 0.0018, 0.0027), DoubleArray(4) { 1.0 }),
        )
        val support = SupportPoints.byAscendingValue(functionals, eps)
        assertEquals(2, support.size, "the groups must be no wider than the tolerance")
        assertEquals(0, support.indexOf(0, 0))
        assertEquals(0, support.indexOf(0, 1))
        assertEquals(1, support.indexOf(0, 2))
        assertEquals(1, support.indexOf(0, 3))
    }
}
