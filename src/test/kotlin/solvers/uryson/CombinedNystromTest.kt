package solvers.uryson

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.NoiseNorm
import problems.uryson.UrysonProblem
import problems.uryson.noisyRightHandSide
import problems.uryson.secondKindSolver
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Checks of the combined Nyström method for the Uryson equation and of the noise normalization.
 *
 *  1. For a kernel LINEAR in `u` (`K(t,s,u) = k(t,s) u`), the nonlinear combined
 *     Nyström must coincide with the linear [FredholmSecondKindSolver.combinedNystrom]
 *     on the same grid and basis: both implementations solve one and the same finite-dimensional
 *     equation `u = f + L_n u`, differing only in the way of solving (Newton against
 *     a simple iteration). The kernel is chosen contractive so that the simple iteration of the linear
 *     solver converges.
 *  2. On the problem B (`u* = e^t`) with a hyperbolic basis the solution lies in the spline
 *     space, and the combined Nyström reproduces it to rounding accuracy.
 *  3. The `SUP` noise normalization gives `max |xi| = delta` exactly.
 */
@Tag("fast")
class CombinedNystromTest {

    private val quad = GaussLegendre(8)

    @Test
    fun linearKernelMatchesFredholmCombinedNystrom() {
        // k(t,s) = 0.5 / (1 + t + s): ||L|| <= 0.5 * ln 3 < 1, the simple Fredholm iteration converges.
        val kLin = { t: Double, s: Double -> 0.5 / (1.0 + t + s) }
        val exact = { t: Double -> 1.0 / (t + 1.0) }
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = ProjFunctionals(basis)

            // The linear solver: u - L u = f, cL = 1.
            val opF = FredholmOperator(KernelF(k = kLin), grid, quad)
            val rhsF = { t: Double -> exact(t) - opF.apply(t) { s -> exact(s) } }
            val fredholm = FredholmSecondKindSolver(
                basis, funcs, opF, cL = 1.0,
                rhs = RhsWithDerivatives(value = rhsF, deriv = { 0.0 }),
            )
            val uF = fredholm.combinedNystrom()

            // The nonlinear solver with the same kernel written as K(t,s,u) = k(t,s) u, lambda = 1.
            val kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = kLin(t, s) * u
                override fun dkdu(t: Double, s: Double, u: Double) = kLin(t, s)
            }
            val problem = UrysonProblem("lin", kernel, lambda = 1.0, exact = exact, secondKind = true)
            val opU = UrysohnOperator(kernel, grid, quad)
            val space = SplineSpace(basis, quad)
            val uryson = secondKindSolver(problem, basis, funcs, space, opU)
            val uU = uryson.combinedNystrom()
            assertTrue(uU.converged, "The Newton of the combined Nyström did not converge, n=$n")

            var diff = 0.0
            val m = 100 * n
            for (i in 0..m) {
                val t = i.toDouble() / m
                diff = maxOf(diff, abs(uU.eval(t) - uF.eval(t)))
            }
            assertTrue(diff < 1e-10, "n=$n: the discrepancy with FredholmSolver.combinedNystrom = $diff")
        }
    }

    @Test
    fun problemBInHyperbolicSpanIsReproducedToRoundoff() {
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(UrysonProblem.B.kernel, grid, quad)
            val solver = secondKindSolver(UrysonProblem.B, basis, funcs, space, op)
            val exact = { t: Double -> UrysonProblem.B.exact(t) }
            val eComb = errorEh(exact, solver.combinedNystrom().eval, grid)
            val eIter = errorEh(exact, solver.iteratedCombinedNystrom().eval, grid)
            assertTrue(eComb < 1e-9, "n=$n: E_h(comb. Nyström) = $eComb")
            assertTrue(eIter < 1e-9, "n=$n: E_h(iter. Nyström) = $eIter")
        }
    }

    @Test
    fun iteratedKulkarniIsNotWorseThanKulkarniOnProblemA() {
        val grid = Grid.uniform(16)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val solver = secondKindSolver(UrysonProblem.A, basis, funcs, space, op)
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val eK = errorEh(exact, solver.kulkarni().eval, grid)
        val eIK = errorEh(exact, solver.iteratedKulkarni().eval, grid)
        assertTrue(eIK <= eK, "iter. Kulkarni $eIK is worse than Kulkarni $eK")
    }

    @Test
    fun supNoiseHasExactSupNorm() {
        val grid = Grid.uniform(16)
        val delta = 1e-3
        val zero = { _: Double -> 0.0 }
        val noisy = noisyRightHandSide(zero, grid, quad, delta, seed = 20240517L, norm = NoiseNorm.SUP)
        // The profile is piecewise linear with knots on the grid 4n: the control grid 100n contains them.
        val m = 100 * grid.n
        var maxAbs = 0.0
        for (i in 0..m) maxAbs = maxOf(maxAbs, abs(noisy(i.toDouble() / m)))
        assertTrue(abs(maxAbs - delta) < 1e-12, "max|xi| = $maxAbs, expected $delta")
        // The L2 variant at the same seed differs only by a factor: the profiles are proportional.
        val noisyL2 = noisyRightHandSide(zero, grid, quad, delta, seed = 20240517L, norm = NoiseNorm.L2)
        val t1 = 0.31; val t2 = 0.77
        val ratio1 = noisy(t1) / noisyL2(t1)
        val ratio2 = noisy(t2) / noisyL2(t2)
        assertTrue(abs(ratio1 - ratio2) < 1e-12 * abs(ratio1), "the SUP and L2 profiles are not proportional")
    }
}
