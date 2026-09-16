package solvers.uryson

import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.ParallelAssembly
import solvers.core.SolutionFunc
import solvers.core.SupportPoints
import splines.functionals.ValueFunctional
import solvers.core.reportConvergence

/**
 * COMBINED Nyström method for the nonlinear Uryson equation of the second kind
 * `u = f + cL L u`, where `(L u)(t) = \int_a^b K(t,s,u(s)) ds`.
 *
 * The approximation `u^N_h` is defined by the equation `u = f + cL L_n u` with the operator
 *
 *     L_n = P_theta L + (I - P_theta) L^N_h,
 *
 * i.e. the EXACT Uryson operator acts on the range of the projector, while on its complement
 * the quadrature `(L^N_h u)(t) = sum_j theta_j(K(t, ., u(.))) W_j`, `W_j = \int omega_j` acts.
 * This is the nonlinear counterpart of the construction `FredholmSecondKindSolver.combinedNystrom`
 * (the linear case) and of the combined operator of the new-01 paper (`eq:nystrom-comb`);
 * the plain Nyström method (`u = f + cL L^N_h u`) is implemented separately in
 * [UrysonSecondKindSolver.nystrom] and is kept unchanged.
 *
 * ## Finite-dimensional unknowns
 *
 * The right-hand side `G(u)(t) = f(t) + cL [ (L^N_h u)(t) + (P_theta(L u - L^N_h u))(t) ]`
 * depends on `u` ONLY through two finite sets of values:
 *  * `y_r = u(eta_r)` at the support points of the functionals (through the quadrature `L^N_h`);
 *  * `z_k = u(g_k)` at the nodes of the composite Gauss–Legendre quadrature (through the exact
 *    operator `L u`, computed as [UrysohnOperator.applyNodes]).
 *
 * Therefore the equation `u = G(u)` is equivalent to a finite-dimensional system in
 * `(y, z)`: `y_r = G(u)(eta_r)`, `z_k = G(u)(g_k)`. The `(y, z)` found recover
 * `u^N_h(t) = G(u)(t)` at any point `t`.
 *
 * ## Solution method
 *
 * Simple iteration `u <- G(u)` (as in the linear `FredholmSecondKindSolver.combinedNystrom`)
 * converges only when `||cL L_n|| < 1`; on problem A (`q = 4` over the solution range)
 * and problem B (cubic nonlinearity) it diverges. The system is therefore solved
 * by Newton's method with an ANALYTIC Jacobian assembled from `dK/du`:
 *
 *  * `d(L^N_h u)(t)/dy_r = b_r * dK/du(t, eta_r, y_r)`, where `b_r = sum_j W_j beta_{j,r}`
 *    and `beta_{j,r}` is the coefficient of the functional `theta_j` at the point `eta_r`;
 *  * `d(L u)(t)/dz_k = gW_k * dK/du(t, g_k, z_k)`;
 *  * `d(P_theta v)(t)/dxi = sum_j omega_j(t) * theta_j(dv/dxi)`, where
 *    `theta_j(w) = sum_q beta_{j,q} w(eta_q)` — the values of `dv/dxi` are needed only
 *    at the support points.
 *
 * The system size is `P + n_g`, where `P` is the number of support points (`2n+1`) and `n_g` the
 * number of quadrature nodes (`8n` with 8 nodes per cell). The initial guess is the projection
 * of the constant function (see the rationale in [UrysonSecondKindSolver.nystrom]).
 *
 * ## Iterated variant
 *
 * `\hat u^N_h = f + cL L u^N_h` — a single application of the exact operator to the
 * approximation found (the analogue of the Sloan iteration); it requires no new system.
 */
internal class CombinedNystromSolver(private val solver: UrysonSecondKindSolver) {
    private val basis = solver.basis
    private val funcs = solver.funcs
    private val op = solver.op
    private val cL = solver.cL
    private val rhs = solver.rhs
    private val n = solver.n
    private val dim = n + 2
    private val ctx = solver.ctx

    /** The functionals `theta_j` as linear combinations of values. */
    private val vfs: Array<ValueFunctional> = Array(dim) { funcs.valueFunctional(it - 2) }

    /**
     * Support points `eta_r` (order of first occurrence, as in [UrysonSecondKindSolver.nystrom])
     * and the indexing of pairs (functional, node) -> point number.
     */
    private val support: SupportPoints = SupportPoints.byFirstOccurrence(vfs, basis.grid.breakpointInclusionEps)
    private val pts: DoubleArray = support.points
    private val p: Int = pts.size

    /** Nyström weights `W_j = \int_a^b omega_j`. */
    private val wInt: DoubleArray = solver.space.wInt

    /**
     * Aggregated quadrature weights `b_r = sum_j W_j beta_{j,r}`:
     * `(L^N_h u)(t) = sum_r b_r K(t, eta_r, u(eta_r))`.
     */
    private val bAgg: DoubleArray = DoubleArray(p).also { b ->
        for (k in 0 until dim) {
            val vf = vfs[k]
            for (q in vf.nodes.indices) b[support.indexOf(k, q)] += vf.coeffs[q] * wInt[k]
        }
    }

    /** Nodes and weights of the composite quadrature of the exact operator. */
    private val gNode: DoubleArray = op.gNode
    private val gW: DoubleArray = op.gW
    private val ng: Int = gNode.size

    /** Values of the basis splines at the support points and quadrature nodes (for `P_theta`). */
    private val omegaAtPts: Array<DoubleArray> = Array(p) { r -> DoubleArray(dim) { k -> basis.omega(k - 2, pts[r]) } }
    private val omegaAtG: Array<DoubleArray> = Array(ng) { k -> DoubleArray(dim) { i -> basis.omega(i - 2, gNode[k]) } }

    init {
        // Invariants: the sum of the Nyström weights equals the interval length (partition of unity),
        // the projector reproduces constants at the support points.
        val lengthCheck = wInt.sum()
        val length = basis.grid.b - basis.grid.a
        check(kotlin.math.abs(lengthCheck - length) <= 1e-10 * (1.0 + length)) {
            "CombinedNystromSolver: sum W_j = $lengthCheck does not equal the interval length $length"
        }
        check(p > 0 && ng > 0) { "CombinedNystromSolver: empty set of support points or quadrature nodes" }
    }

    /** Quadrature operator `(L^N_h u)(t)` from the values `y = u(eta)`. */
    private fun quadratureAt(t: Double, y: DoubleArray): Double {
        var acc = 0.0
        for (r in 0 until p) acc += bAgg[r] * op.kernel.k(t, pts[r], y[r])
        return acc
    }

    /**
     * Coefficients of `P_theta v` from the values of `v` at the support points:
     * `c_j = theta_j(v) = sum_q beta_{j,q} v(eta_q)`.
     */
    private fun projectFromSupport(vAtPts: DoubleArray): DoubleArray = DoubleArray(dim) { k ->
        val vf = vfs[k]
        var s = 0.0
        for (q in vf.nodes.indices) s += vf.coeffs[q] * vAtPts[support.indexOf(k, q)]
        s
    }

    /** Value of the spline with coefficients `c` at a point with precomputed `omega`. */
    private fun splineDot(c: DoubleArray, omegaRow: DoubleArray): Double {
        var s = 0.0
        for (i in 0 until dim) s += c[i] * omegaRow[i]
        return s
    }

    /**
     * State of the right-hand side `G(u)` for given `(y, z)`: the values of `G` at the support
     * points and at the quadrature nodes, the projection coefficients of the difference `L u - L^N_h u`
     * (needed to recover the solution at an arbitrary point) and a copy of `y`.
     */
    private class State(
        val gAtPts: DoubleArray,
        val gAtG: DoubleArray,
        val diffCoeffs: DoubleArray,
        val yAll: DoubleArray,
    )

    private fun evaluate(y: DoubleArray, z: DoubleArray): State {
        val exactAtPts = DoubleArray(p) { r -> op.applyNodes(pts[r], z) }
        val quadAtPts = DoubleArray(p) { r -> quadratureAt(pts[r], y) }
        val diffCoeffs = projectFromSupport(DoubleArray(p) { exactAtPts[it] - quadAtPts[it] })
        val gAtPts = DoubleArray(p) { r ->
            rhs(pts[r]) + cL * (quadAtPts[r] + splineDot(diffCoeffs, omegaAtPts[r]))
        }
        val gAtG = DoubleArray(ng) { k ->
            rhs(gNode[k]) + cL * (quadratureAt(gNode[k], y) + splineDot(diffCoeffs, omegaAtG[k]))
        }
        return State(gAtPts, gAtG, diffCoeffs, y.copyOf())
    }

    /**
     * Jacobian `F'(y, z)` of the system `F(y,z) = (y - G(eta), z - G(g))`, of size `(p+ng)^2`.
     *
     * Derivatives of `G(t)` in `y_r` and `z_k`:
     *  * `dQ(t)/dy_r = b_r dK/du(t, eta_r, y_r)`  (quadrature);
     *  * `dE(t)/dz_k = gW_k dK/du(t, g_k, z_k)`   (exact operator);
     *  * `dG(t)/dy_r = cL [ dQ(t)/dy_r - sum_j omega_j(t) theta_j(dQ(.)/dy_r) ]`;
     *  * `dG(t)/dz_k = cL [ sum_j omega_j(t) theta_j(dE(.)/dz_k) ]`.
     */
    private fun jacobian(y: DoubleArray, z: DoubleArray): DenseMatrix {
        val total = p + ng
        val dQpts = Array(p) { rho -> DoubleArray(p) { r -> bAgg[r] * op.kernel.dkdu(pts[rho], pts[r], y[r]) } }
        val dEpts = Array(p) { rho -> DoubleArray(ng) { k -> gW[k] * op.kernel.dkdu(pts[rho], gNode[k], z[k]) } }
        // cQ[j][r] = theta_j(dQ(.)/dy_r), cE[j][k] = theta_j(dE(.)/dz_k).
        val cQ = Array(dim) { j ->
            val vf = vfs[j]
            DoubleArray(p) { r ->
                var s = 0.0
                for (q in vf.nodes.indices) s += vf.coeffs[q] * dQpts[support.indexOf(j, q)][r]
                s
            }
        }
        val cE = Array(dim) { j ->
            val vf = vfs[j]
            DoubleArray(ng) { k ->
                var s = 0.0
                for (q in vf.nodes.indices) s += vf.coeffs[q] * dEpts[support.indexOf(j, q)][k]
                s
            }
        }
        // Cell-by-cell assembly: the expression for each entry and the order of additions in `proj`
        // are verbatim the same as in the row-by-row assembly, so the numbers do not change.
        return ParallelAssembly.assembleDense(total, total, ctx.parallel) { row, col ->
            val tRow = if (row < p) pts[row] else gNode[row - p]
            val omegaRow = if (row < p) omegaAtPts[row] else omegaAtG[row - p]
            val value = if (col < p) {
                val dq = bAgg[col] * op.kernel.dkdu(tRow, pts[col], y[col])
                var proj = 0.0
                for (j in 0 until dim) proj += omegaRow[j] * cQ[j][col]
                -cL * (dq - proj)
            } else {
                var proj = 0.0
                for (j in 0 until dim) proj += omegaRow[j] * cE[j][col - p]
                -cL * proj
            }
            if (row == col) value + 1.0 else value
        }
    }

    /** Solves the system of the combined method; returns the state of `G` and the convergence information. */
    private fun solveSystem(): Pair<State, NewtonRun> {
        val constantProjection = funcs.projectorCoeffs({ 1.0 })
        val x = DoubleArray(p + ng) { i ->
            if (i < p) basis.evalSpline(constantProjection, pts[i]) else basis.evalSpline(constantProjection, gNode[i - p])
        }
        val newtonTol = maxOf(solver.tol, NEWTON_TOLERANCE_FLOOR)
        var lastState: State? = null
        val run = runNewtonIterations(
            x = x,
            maxSteps = solver.nystromMaxIter,
            tolerance = newtonTol,
            residualAt = { current ->
                val y = current.copyOfRange(0, p)
                val z = current.copyOfRange(p, p + ng)
                val st = evaluate(y, z)
                lastState = st
                DoubleArray(p + ng) { i -> if (i < p) y[i] - st.gAtPts[i] else z[i - p] - st.gAtG[i - p] }
            },
            stepAt = { current, residual ->
                val y = current.copyOfRange(0, p)
                val z = current.copyOfRange(p, p + ng)
                LinearAlgebra.solve(jacobian(y, z), DoubleArray(p + ng) { -residual[it] }, ctx.backend)
            },
        )
        reportConvergence(
            converged = run.converged,
            throwOnDivergence = solver.throwOnDivergence,
            methodName = "Newton (combined Uryson Nyström)",
            iterations = run.performedSteps,
            maxIterations = solver.nystromMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
        )
        // On convergence the last residual was measured at the returned point (see runNewtonIterations),
        // so lastState corresponds exactly to it; otherwise we recompute explicitly.
        val state = if (run.converged && lastState != null) {
            lastState!!
        } else {
            evaluate(x.copyOfRange(0, p), x.copyOfRange(p, p + ng))
        }
        return state to run
    }

    /** `u^N_h(t) = G(u)(t)` — recovery of the solution at an arbitrary point. */
    private fun evalFrom(state: State): (Double) -> Double = { t ->
        rhs(t) + cL * (quadratureAt(t, state.yAll) + basis.evalSpline(state.diffCoeffs, t))
    }

    fun combined(): SolutionFunc {
        val (state, run) = solveSystem()
        return SolutionFunc(
            eval = evalFrom(state),
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    fun iterated(): SolutionFunc {
        val (state, run) = solveSystem()
        // The exact operator at the u^N_h found: its values at the quadrature nodes are G(g).
        val uNodes = state.gAtG
        return SolutionFunc(
            eval = { t -> rhs(t) + cL * op.applyNodes(t, uNodes) },
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    private companion object {
        /** Lower bound of the stopping criterion (analytic Jacobian). */
        const val NEWTON_TOLERANCE_FLOOR = 1e-13
    }
}
