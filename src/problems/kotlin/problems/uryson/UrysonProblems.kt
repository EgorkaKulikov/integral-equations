package problems.uryson

import numerics.GaussLegendre
import splines.Grid
import splines.MinimalSplineBasis
import numerics.NumericsContext
import splines.functionals.ProjFunctionals
import solvers.uryson.Kernel
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import solvers.uryson.UrysonFirstKindSolver
import solvers.uryson.UrysonSecondKindSolver

/**
 * Model problem for the nonlinear Uryson equation: kernel, multiplier and exact solution.
 *
 * The right-hand side is NOT given explicitly but built from the exact solution numerically (by quadrature),
 * so the test data is always consistent with the operator and the quadrature rule.
 *
 * Relations between the exact solution `x*` and the right-hand side `f`:
 *  - equation of the second kind: `f(t) = x*(t) - lambda \int K(t,s,x*(s)) ds`;
 *  - equation of the first kind: `f(t) = \int K(t,s,x*(s)) ds`.
 *
 * The interval is defined not here but by the grid [Grid] passed to the operator.
 *
 * @param name short problem name for tables and test messages.
 * @param kernel kernel `K(t,s,u)` together with its derivative in `u`.
 * @param lambda multiplier in front of the integral operator (unused for problems of the first kind).
 * @param exact exact solution — the baseline for computing the error.
 * @param secondKind `true` — equation of the second kind, `false` — of the first.
 */
class UrysonProblem(
    val name: String,
    val kernel: Kernel,
    val lambda: Double,
    val exact: (Double) -> Double,
    val secondKind: Boolean,
) {
    /** Exact right-hand side `f(t)`, computed through the operator [op]. */
    fun rhsExact(t: Double, op: UrysohnOperator): Double {
        val integral = op.apply(t) { s -> exact(s) }
        return if (secondKind) exact(t) - lambda * integral else integral
    }

    companion object {
        /**
         * Problem A (second kind): `K = 1/(t+s+u)`, `lambda = -1`, `x* = 1/(t+1)`.
         *
         * The kernel is smooth and decreasing, the operator is contractive — the base convergence scenario.
         */
        val A = UrysonProblem(
            name = "A",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = -1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = true,
        )

        /**
         * Problem B (second kind): `K = e^{t-2s} u^3`, `lambda = 1`, `x* = e^t`.
         *
         * The cubic nonlinearity at `lambda = 1` makes the operator NON-contractive:
         * simple iteration diverges, so the problem exercises exactly the Newton path.
         * The solution `e^t` belongs to the hyperbolic generating system `phi^H`.
         */
        val B = UrysonProblem(
            name = "B",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = Math.exp(t - 2.0 * s) * u * u * u
                override fun dkdu(t: Double, s: Double, u: Double) = 3.0 * Math.exp(t - 2.0 * s) * u * u
            },
            lambda = 1.0,
            exact = { t -> Math.exp(t) },
            secondKind = true,
        )

        /** Problem C (first kind, ill-posed): `K = 1/(t+s+u)`, `x* = 1/(t+1)`. */
        val C = UrysonProblem(
            name = "C",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = 1.0 / (t + s + u)
                override fun dkdu(t: Double, s: Double, u: Double) = -1.0 / ((t + s + u) * (t + s + u))
            },
            lambda = 1.0,
            exact = { t -> 1.0 / (t + 1.0) },
            secondKind = false,
        )

        /**
         * Problem D (first kind, ill-posed): `K = e^{-(t-s)^2} u^3`, `x* = e^t`.
         *
         * Special feature: `dK/du(t,s,0) = 0`, so from a ZERO initial guess
         * the Jacobian is singular and the Gauss–Newton method never moves. That is exactly
         * why the solvers start from the projection of a constant function.
         */
        val D = UrysonProblem(
            name = "D",
            kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = Math.exp(-(t - s) * (t - s)) * u * u * u
                override fun dkdu(t: Double, s: Double, u: Double) = 3.0 * Math.exp(-(t - s) * (t - s)) * u * u
            },
            lambda = 1.0,
            exact = { t -> Math.exp(t) },
            secondKind = false,
        )
    }
}

/**
 * Creates a solver of the equation of the second kind for a model problem.
 *
 * @param ctx computation context; must coincide with the contexts of [funcs] and [space]
 *        (checked by the solver constructor).
 */
fun secondKindSolver(
    problem: UrysonProblem,
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
    ctx: NumericsContext = NumericsContext.default(),
): UrysonSecondKindSolver = UrysonSecondKindSolver(
    basis = basis,
    funcs = funcs,
    space = space,
    op = op,
    cL = problem.lambda,
    rhs = { t -> problem.rhsExact(t, op) },
    ctx = ctx,
)

/**
 * Creates a regularized solver of the equation of the first kind.
 *
 * @param ctx computation context; must coincide with the contexts of [funcs] and [space].
 */
fun firstKindSolver(
    basis: MinimalSplineBasis,
    funcs: ProjFunctionals,
    space: SplineSpace,
    op: UrysohnOperator,
    ctx: NumericsContext = NumericsContext.default(),
): UrysonFirstKindSolver = UrysonFirstKindSolver(basis, funcs, space, op, ctx = ctx)

/**
 * Number of control nodes of the noise profile per grid interval.
 *
 * The profile must be noticeably finer than the grid, otherwise the noise becomes "visible" to the basis
 * and is partly reproduced instead of acting as a perturbation of the data.
 */
private const val NOISE_NODES_PER_INTERVAL = 4

/**
 * Norm in which the noise level `delta` for [noisyRightHandSide] is specified.
 *
 *  * [L2] — `||xi||_{L^2(a,b)} = delta` (the former behaviour, used by the golden tests);
 *  * [SUP] — `||xi||_infty = max_t |xi(t)| = delta` — matches condition (VII)
 *    of the paper (the noise is given in `C[a,b]`). For a piecewise linear profile the maximum
 *    modulus is attained at a control node, so it is computed exactly, without quadrature.
 */
enum class NoiseNorm { L2, SUP }

/**
 * Builds a noisy right-hand side `f^delta = f + xi` with a prescribed noise norm
 * `||xi|| = delta` in the norm [norm] (`L^2` by default).
 *
 * The noise is modelled by a piecewise linear profile with random values at the control
 * nodes, scaled exactly to the required level `delta`.
 *
 * REPRODUCIBILITY: the generator is initialized by the EXPLICITLY passed [seed], so the
 * result is fully deterministic. There is no hidden source of randomness here.
 * For a given `seed` the profile (nodes and random values) is THE SAME for both norms —
 * only the scaling factor differs.
 *
 * @param exactRhs exact right-hand side `f`.
 * @param grid grid defining the interval.
 * @param quad quadrature for computing the `L^2` norm of the noise (unused for [NoiseNorm.SUP]).
 * @param delta required perturbation norm; at zero the original function is returned.
 * @param seed seed of the pseudo-random number generator.
 * @param norm norm in which `delta` is given.
 */
fun noisyRightHandSide(
    exactRhs: (Double) -> Double,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
    norm: NoiseNorm = NoiseNorm.L2,
): (Double) -> Double {
    if (delta == 0.0) return exactRhs
    val random = kotlin.random.Random(seed)
    val nodeCount = NOISE_NODES_PER_INTERVAL * grid.n
    val noiseNodes = DoubleArray(nodeCount + 1) { grid.a + (grid.b - grid.a) * it / nodeCount }
    val noiseValues = DoubleArray(nodeCount + 1) { random.nextDouble(-1.0, 1.0) }
    val noiseProfile = { t: Double ->
        var k = 0
        while (k < nodeCount - 1 && t >= noiseNodes[k + 1]) k++
        val left = noiseNodes[k]
        val right = noiseNodes[k + 1]
        val w = ((t - left) / (right - left)).coerceIn(0.0, 1.0)
        noiseValues[k] * (1 - w) + noiseValues[k + 1] * w
    }
    val profileNorm = when (norm) {
        NoiseNorm.L2 -> Math.sqrt(quad.integrate(noiseNodes) { t -> noiseProfile(t) * noiseProfile(t) })
        // A piecewise linear function attains its maximum modulus at a node.
        NoiseNorm.SUP -> noiseValues.maxOf { Math.abs(it) }
    }
    val scale = if (profileNorm > 0) delta / profileNorm else 0.0
    return { t -> exactRhs(t) + scale * noiseProfile(t) }
}

/**
 * Returns the vector `theta_j(f^delta)` of noisy data for a problem of the first kind —
 * the input of the method [UrysonFirstKindSolver.solveMorozov].
 *
 * @param delta noise level in the norm [norm] (`L^2` by default).
 * @param seed generator seed; fixed explicitly for reproducibility.
 * @param norm norm in which `delta` is given (see [NoiseNorm]).
 */
fun noisyThetaCoefficients(
    problem: UrysonProblem,
    solver: UrysonFirstKindSolver,
    op: UrysohnOperator,
    grid: Grid,
    quad: GaussLegendre,
    delta: Double,
    seed: Long,
    norm: NoiseNorm = NoiseNorm.L2,
): DoubleArray {
    val noisy = noisyRightHandSide({ t -> problem.rhsExact(t, op) }, grid, quad, delta, seed, norm)
    return solver.thetaOf(noisy)
}
