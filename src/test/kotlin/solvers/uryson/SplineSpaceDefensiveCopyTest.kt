package solvers.uryson

import numerics.GaussLegendre
import org.junit.jupiter.api.Tag
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The encapsulation of the COLD arrays: a getter must return a COPY, so that a mutation by
 * the caller does not corrupt the source.
 *
 * Previously `GaussLegendre.refNodesWeights()` and the fields `SplineSpace.weights/wInt/gramR`
 * returned the internal arrays themselves. Any caller could silently change the quadrature
 * or the weights with one write AT ONCE FOR ALL the users of the object, and without
 * an exception — one simply got different numbers.
 *
 * THE BOUNDARY OF THE POLICY (important for reading the test): only the cold fields, read
 * once, are copied. The hot arrays (`Grid.breakpoints`, the `gNode`/`gW` of the operators,
 * `ValueFunctional.nodes/coeffs`) are read in hot loops tens of thousands of times per run
 * and stay read-only BY CONVENTION — copying on every access would roll back
 * the optimizations of the hot path. Therefore they are deliberately absent here.
 *
 * The same guarantee for `GaussLegendre.refNodesWeights()` is checked in the library
 * `numerical-core` (`QuadratureDefensiveCopyTest`).
 */
@Tag("fast")
class SplineSpaceDefensiveCopyTest {

    private fun space(): SplineSpace {
        val grid = Grid.uniform(8, 0.0, 1.0)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        return SplineSpace(basis, GaussLegendre(8))
    }

    /** A mutation of the nodes/weights obtained from the quadrature does not change the quadrature itself. */

    /**
     * The mutation must not reach the RESULT of the quadrature either: the check is not on the equality
     * of the arrays but on the computed number itself (a corruption of the nodes would show up exactly here).
     */

    /** `SplineSpace.weights` is a copy: a write into it changes neither the field nor `weightsSum()`. */
    @Test fun spaceWeightsReturnsCopy() {
        val space = space()
        val sumBefore = space.weightsSum()
        val w = space.weights
        val snapshot = w.copyOf()

        w[0] = 1e9

        assertTrue(snapshot.contentEquals(space.weights), "space.weights changed after a mutation of the copy")
        assertEquals(sumBefore, space.weightsSum(), 0.0, "weightsSum() changed after a mutation of the copy")
    }

    /** `SplineSpace.wInt` is a copy. */
    @Test fun spaceWIntReturnsCopy() {
        val space = space()
        val w = space.wInt
        val snapshot = w.copyOf()

        w[0] = 1e9

        assertTrue(snapshot.contentEquals(space.wInt), "space.wInt changed after a mutation of the copy")
    }

    /**
     * `SplineSpace.gramR` is a DEEP copy.
     *
     * After the move to [DenseMatrix] there are no row arrays: the values lie in one
     * flat buffer, and `copy()` copies them entirely. The check is kept verbatim —
     * a write into an element of the copy must be visible neither through the getter nor through omegaReg.
     */
    @Test fun spaceGramRReturnsDeepCopy() {
        val space = space()
        val g = space.gramR
        val original = g[0, 0]
        val regBefore = space.omegaReg(DoubleArray(space.dim) { 1.0 })

        g[0, 0] = original + 1e9

        assertEquals(original, space.gramR[0, 0], 0.0, "gramR changed: the copy turned out to be shallow")
        assertEquals(
            regBefore, space.omegaReg(DoubleArray(space.dim) { 1.0 }), 0.0,
            "omegaReg changed after a mutation of the copy of gramR",
        )
    }

    /** Different accesses to the getter return DIFFERENT objects (otherwise there is no copy). */
    @Test fun coldGettersReturnDistinctInstances() {
        val space = space()
        assertTrue(space.weights !== space.weights, "weights: one and the same array is returned")
        assertTrue(space.wInt !== space.wInt, "wInt: one and the same array is returned")
        assertTrue(space.gramR !== space.gramR, "gramR: one and the same matrix is returned")
        assertTrue(space.gramR.data !== space.gramR.data, "gramR: the value buffer is not copied")

        val quad = GaussLegendre(8)
        assertTrue(quad.refNodesWeights().first !== quad.refNodesWeights().first, "refNodes: the array is not copied")
        assertTrue(quad.refNodesWeights().second !== quad.refNodesWeights().second, "refWeights: the array is not copied")
    }

    /** The values of the copies coincide with the source — the copying did not distort the numbers. */
    @Test fun copiesCarryIdenticalValues() {
        val space = space()
        assertEquals(space.dim, space.weights.size)
        assertEquals(space.dim, space.wInt.size)
        assertEquals(space.dim, space.gramR.rows)
        // The sum of the weights = the length of the interval: a substantial invariant, not only the size.
        assertEquals(1.0, space.weights.sum(), 1e-12)
        assertEquals(space.weightsSum(), space.weights.sum(), 0.0)
        // The symmetry of the Gram matrix is preserved in the copy. The copy is taken once BEFORE the loops:
        // the getter copies deeply, and a call inside the loop would give dim^2 copies of the matrix.
        val gram = space.gramR
        for (i in 0 until space.dim) for (j in 0 until space.dim) {
            assertEquals(gram[i, j], gram[j, i], 1e-12)
        }
    }
}
