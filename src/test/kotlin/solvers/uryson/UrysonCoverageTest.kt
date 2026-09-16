package solvers.uryson

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import solvers.core.FirstKindSolution
import solvers.core.SolutionFunc
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyRightHandSide
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import kotlin.math.abs
import kotlin.test.Test

import kotlin.test.assertTrue

/**
 * Characterization tests of the solvers of the nonlinear Uryson equation and of the auxiliary types.
 *
 * IMPORTANT: the numerical thresholds here are not an analytic truth. The baseline values
 * were recorded by a single run of the current implementation and serve as a safety net
 * against regressions: they catch the appearance of a NaN, a divergence and a "blow-up" of the solution, but do not
 * confirm the theoretical convergence orders.
 */
@Tag("fast")
class UrysonCoverageTest {

    private val quad = GaussLegendre(8)

    private fun finite(x: Double) = !x.isNaN() && !x.isInfinite()

    private fun solverFor(
        problem: UrysonProblem,
        system: GeneratingSystem,
        n: Int,
    ): UrysonSecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, space, op)
    }

    /** The fields and the evaluator of [FirstKindSolution] — a simple data holder. */
    @Test
    fun firstKindSolutionFields() {
        val solution = FirstKindSolution(doubleArrayOf(1.0, 2.0), { t -> t * t }, 1e-3, 1e-4, 0.5)
        assertTrue(solution.coeffs.size == 2)
        assertTrue(solution.alpha == 1e-3 && solution.residual == 1e-4 && solution.omega == 0.5)
        assertTrue(abs(solution.eval(3.0) - 9.0) < 1e-12)
    }

    /**
     * The fields of [SolutionFunc]: the evaluator, the convergence flag, the iteration counter
     * and the attained residual.
     */
    @Test
    fun solutionFuncFields() {
        val solution = SolutionFunc(
            eval = { t -> t + 1 },
            converged = false,
            iterations = 7,
            residual = 1e-5,
        )
        assertTrue(abs(solution.eval(1.0) - 2.0) < 1e-12)
        assertTrue(!solution.converged && solution.iterations == 7 && solution.residual == 1e-5)
    }

    /**
     * The default values of [SolutionFunc] correspond to the DIRECT schemes: such schemes
     * solve a linear system and perform no iterations, so the convergence is trivial.
     */
    @Test
    fun solutionFuncDefaultsDescribeDirectSchemes() {
        val direct = SolutionFunc(eval = { t -> t })
        assertTrue(direct.converged, "A direct scheme is considered converged by construction")
        assertTrue(direct.iterations == 0 && direct.residual == 0.0)
    }

    /**
     * The properties of [SplineSpace]: the sum of the weights equals the length of the interval, the Gram matrix
     * is symmetric, the quadratic form of the stabilizer is non-negative.
     */
    @Test
    fun splineSpaceWeightsGramAndRegularizer() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val space = SplineSpace(basis, quad)
        assertTrue(abs(space.weightsSum() - 1.0) < 1e-12)
        assertTrue(space.weights.sum() > 0 && space.wInt.all { finite(it) })
        // The getter returns a DEEP copy — we take it once BEFORE the loops, otherwise every
        // iteration would copy the whole matrix (twice).
        val gram = space.gramR
        for (i in 0 until space.dim) {
            for (j in 0 until space.dim) {
                assertTrue(abs(gram[i, j] - gram[j, i]) < 1e-12)
            }
        }
        assertTrue(abs(space.omegaReg(DoubleArray(space.dim))) < 1e-15)
        assertTrue(space.omegaReg(DoubleArray(space.dim) { 1.0 }) > 0.0)
    }

    /**
     * The consistency of the interfaces [UrysohnOperator] and of the right-hand side for all four
     * model problems: the evaluation over the precomputed nodes coincides with the evaluation
     * through a closure.
     */
    @Test
    fun operatorApisAgreeForAllProblems() {
        val grid = Grid.uniform(8)
        val problems = listOf(
            UrysonProblem.A, UrysonProblem.B, UrysonProblem.C, UrysonProblem.D,
        )
        for (problem in problems) {
            val op = UrysohnOperator(problem.kernel, grid, quad)
            val t = 0.4
            val xNodes = DoubleArray(op.gNode.size) { problem.exact(op.gNode[it]) }
            val viaNodes = op.applyNodes(t, xNodes)
            val viaClosure = op.apply(t) { s -> problem.exact(s) }
            assertTrue(
                abs(viaNodes - viaClosure) < 1e-9,
                "${problem.name}: applyNodes and apply must coincide",
            )
            assertTrue(finite(op.frechet(t, { s -> problem.exact(s) }, { 1.0 })))
            assertTrue(finite(problem.rhsExact(t, op)))
            assertTrue(finite(problem.kernel.dkdu(t, 0.3, 1.0)))
        }
    }

    /** The vector `Xi` and the Jacobian `B` have the right sizes and are finite. */
    @Test
    fun collocationCoreProducesFiniteXiAndJacobian() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.A.exact(t) })
        val xi = core.xiVector(coefficients)
        val jacobian = core.bMatrix(coefficients)
        assertTrue(xi.size == grid.n + 2 && xi.all { finite(it) })
        assertTrue(
            jacobian.rows == grid.n + 2 && jacobian.cols == grid.n + 2 &&
                jacobian.data.all { finite(it) },
        )
        assertTrue(core.uAtSupport(coefficients).all { finite(it) })
    }

    /** All four second-kind schemes give a finite result on the contractive problem A. */
    @Test
    fun problemAllSchemesAreFinite() {
        val solver = solverFor(UrysonProblem.A, GeneratingSystem.B, 8)
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val solutions = listOf(solver.base(), solver.sloan(), solver.kulkarni(), solver.nystrom())
        for (solution in solutions) {
            val error = errorEh(exact, solution.eval, solver.grid)
            assertTrue(finite(error) && error < 1e-2, "Problem A: E_h = $error")
        }
    }

    /**
     * The problem B with a cubic kernel at `lambda = 1` is NON-contractive: the simple iteration
     * diverges, so the test checks exactly the Newton path of the solver.
     */
    @Test
    fun problemBNonContractiveSchemesConverge() {
        val solver = solverFor(UrysonProblem.B, GeneratingSystem.H, 8)
        val exact = { t: Double -> UrysonProblem.B.exact(t) }
        for (solution in listOf(solver.base(), solver.sloan(), solver.nystrom())) {
            val error = errorEh(exact, solution.eval, solver.grid)
            assertTrue(finite(error) && error < 1e-1, "Problem B: E_h = $error")
        }
    }

    /** The base scheme converges: the error decreases on the transition n = 8 -> 16. */
    @Test
    fun problemAConverges() {
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val errorN8 = errorEh(exact, solverFor(UrysonProblem.A, GeneratingSystem.B, 8).base().eval, Grid.uniform(8))
        val errorN16 = errorEh(exact, solverFor(UrysonProblem.A, GeneratingSystem.B, 16).base().eval, Grid.uniform(16))
        assertTrue(errorN16 < errorN8, "No convergence: E_8 = $errorN8, E_16 = $errorN16")
    }

    /**
     * The regularized solver on the problem C: the branches of the noise generator at a zero and
     * a non-zero level, the Gauss-Newton step, the residual computation and the Morozov path without noise.
     */
    @Test
    fun firstKindSolverOnProblemCWithoutNoise() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)

        val thetaExact = noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 0.0, 1L)
        assertTrue(thetaExact.size == grid.n + 2 && thetaExact.all { finite(it) })

        val thetaNoisy = noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 1e-3, 42L)
        assertTrue(thetaNoisy.all { finite(it) })

        val start = funcs.projectorCoeffs({ 1.0 })
        val coefficients = solver.solveFixedAlpha(thetaExact, 1e-4, start)
        assertTrue(coefficients.all { finite(it) })
        assertTrue(finite(solver.residual(coefficients, thetaExact)))

        val solution = solver.solveMorozov(thetaExact, 0.0)
        assertTrue(solution.alpha > 0 && finite(solution.residual) && finite(solution.eval(0.5)))
    }

    /**
     * The Morozov path with noise on the problem D with a cubic kernel: `dK/du(t,s,0) = 0`,
     * so the branch of a restart from a non-zero initial approximation is checked.
     */
    @Test
    fun firstKindSolverOnProblemDWithNoise() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.D.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta = noisyThetaCoefficients(UrysonProblem.D, solver, op, grid, quad, 1e-2, 7L)
        val solution = solver.solveMorozov(thetaFDelta, 1e-2)
        assertTrue(solution.alpha > 0 && finite(solution.residual) && finite(solution.omega))
        assertTrue(finite(solution.eval(0.3)))
    }

    /**
     * The noise generator is deterministic: with one seed the result is reproduced
     * bitwise, with different ones it differs.
     */
    @Test
    fun noiseGeneratorIsReproducible() {
        val grid = Grid.uniform(8)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val exactRhs = { t: Double -> UrysonProblem.C.rhsExact(t, op) }
        val first = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 123L)
        val second = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 123L)
        val other = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 456L)
        var differsFromOther = false
        for (i in 0..20) {
            val t = i / 20.0
            assertTrue(first(t) == second(t), "One seed must give identical noise at the point $t")
            if (abs(first(t) - other(t)) > 1e-15) differsFromOther = true
        }
        assertTrue(differsFromOther, "Different seeds must give different noise")
        // At a zero noise level the original right-hand side is returned.
        val noiseless = noisyRightHandSide(exactRhs, grid, quad, 0.0, 1L)
        assertTrue(noiseless(0.5) == exactRhs(0.5))
    }
}
