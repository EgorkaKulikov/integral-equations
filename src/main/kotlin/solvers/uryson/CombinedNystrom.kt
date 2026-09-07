package solvers.uryson

import numerics.LinearAlgebra
import numerics.ParallelAssembly
import solvers.core.SolutionFunc
import splines.functionals.SupportPoints
import splines.functionals.ValueFunctional
import solvers.core.reportConvergence

/**
 * КОМБИНИРОВАННЫЙ метод Nyström для нелинейного уравнения Урысона второго рода
 * `u = f + lambda L u`, где `(L u)(t) = \int_a^b K(t,s,u(s)) ds`.
 *
 * Приближение `u^N_h` определяется уравнением `u = f + lambda L_n u` с оператором
 *
 *     L_n = P_theta L + (I - P_theta) L^N_h,
 *
 * то есть на образе проектора действует ТОЧНЫЙ оператор Урысона, а на его дополнении —
 * квадратура `(L^N_h u)(t) = sum_j theta_j(K(t, ., u(.))) W_j`, `W_j = \int omega_j`.
 * Это нелинейный аналог конструкции `FredholmSecondKindSolver.combinedNystrom`
 * (линейный случай) и комбинированного оператора статьи new-01 (`eq:nystrom-comb`);
 * простой Nyström (`u = f + lambda L^N_h u`) реализован отдельно в
 * [UrysonSecondKindSolver.nystrom] и сохранён без изменений.
 *
 * ## Конечномерные неизвестные
 *
 * Правая часть `G(u)(t) = f(t) + lambda [ (L^N_h u)(t) + (P_theta(L u - L^N_h u))(t) ]`
 * зависит от `u` ТОЛЬКО через два конечных набора значений:
 *  * `y_r = u(eta_r)` в опорных точках функционалов (через квадратуру `L^N_h`);
 *  * `z_k = u(g_k)` в узлах составной квадратуры Гаусса–Лежандра (через точный оператор
 *    `L u`, вычисляемый как [UrysohnOperator.applyNodes]).
 *
 * Поэтому уравнение `u = G(u)` равносильно конечномерной системе относительно
 * `(y, z)`: `y_r = G(u)(eta_r)`, `z_k = G(u)(g_k)`. Найденные `(y, z)` восстанавливают
 * `u^N_h(t) = G(u)(t)` в любой точке `t`.
 *
 * ## Метод решения
 *
 * Простая итерация `u <- G(u)` (как в линейном `FredholmSecondKindSolver.combinedNystrom`)
 * сходится лишь при `||lambda L_n|| < 1`; на задаче A (`q = 4` на диапазоне решения)
 * и задаче B (кубическая нелинейность) она расходится. Поэтому система решается
 * методом Ньютона с АНАЛИТИЧЕСКИМ якобианом, собранным из `dK/du`:
 *
 *  * `d(L^N_h u)(t)/dy_r = b_r * dK/du(t, eta_r, y_r)`, где `b_r = sum_j W_j beta_{j,r}`,
 *    а `beta_{j,r}` — коэффициент функционала `theta_j` при точке `eta_r`;
 *  * `d(L u)(t)/dz_k = gW_k * dK/du(t, g_k, z_k)`;
 *  * `d(P_theta v)(t)/dxi = sum_j omega_j(t) * theta_j(dv/dxi)`, причём
 *    `theta_j(w) = sum_q beta_{j,q} w(eta_q)` — значения `dv/dxi` нужны лишь
 *    в опорных точках.
 *
 * Размер системы `P + n_g`, где `P` — число опорных точек (`2n+1`), `n_g` — число
 * узлов квадратуры (`8n` при 8 узлах на ячейку). Начальное приближение — проекция
 * постоянной функции (см. обоснование в [UrysonSecondKindSolver.nystrom]).
 *
 * ## Итерированный вариант
 *
 * `\hat u^N_h = f + lambda L u^N_h` — однократное применение точного оператора к
 * найденному приближению (аналог итерации Слоана), новой системы не требует.
 */
internal class CombinedNystromSolver(private val solver: UrysonSecondKindSolver) {
    private val basis = solver.basis
    private val funcs = solver.funcs
    private val op = solver.op
    private val lambda = solver.lambda
    private val rhs = solver.rhs
    private val n = solver.n
    private val dim = n + 2
    private val ctx = solver.ctx

    /** Функционалы `theta_j` как линейные комбинации значений. */
    private val vfs: Array<ValueFunctional> = Array(dim) { funcs.valueFunctional(it - 2) }

    /**
     * Опорные точки `eta_r` (порядок первого вхождения, как в [UrysonSecondKindSolver.nystrom])
     * и индексация пар (функционал, узел) -> номер точки.
     */
    private val support: SupportPoints = SupportPoints.byFirstOccurrence(vfs, basis.grid.breakpointInclusionEps)
    private val pts: DoubleArray = support.points
    private val p: Int = pts.size

    /** Веса Nyström `W_j = \int_a^b omega_j`. */
    private val wInt: DoubleArray = solver.space.wInt

    /**
     * Агрегированные веса квадратуры `b_r = sum_j W_j beta_{j,r}`:
     * `(L^N_h u)(t) = sum_r b_r K(t, eta_r, u(eta_r))`.
     */
    private val bAgg: DoubleArray = DoubleArray(p).also { b ->
        for (k in 0 until dim) {
            val vf = vfs[k]
            for (q in vf.nodes.indices) b[support.indexOf(k, q)] += vf.coeffs[q] * wInt[k]
        }
    }

    /** Узлы и веса составной квадратуры точного оператора. */
    private val gNode: DoubleArray = op.gNode
    private val gW: DoubleArray = op.gW
    private val ng: Int = gNode.size

    /** Значения базисных сплайнов в опорных точках и узлах квадратуры (для `P_theta`). */
    private val omegaAtPts: Array<DoubleArray> = Array(p) { r -> DoubleArray(dim) { k -> basis.omega(k - 2, pts[r]) } }
    private val omegaAtG: Array<DoubleArray> = Array(ng) { k -> DoubleArray(dim) { i -> basis.omega(i - 2, gNode[k]) } }

    init {
        // Инварианты: сумма весов Nyström равна длине отрезка (разбиение единицы),
        // проектор воспроизводит константы в опорных точках.
        val lengthCheck = wInt.sum()
        val length = basis.grid.b - basis.grid.a
        check(kotlin.math.abs(lengthCheck - length) <= 1e-10 * (1.0 + length)) {
            "CombinedNystromSolver: sum W_j = $lengthCheck не равна длине отрезка $length"
        }
        check(p > 0 && ng > 0) { "CombinedNystromSolver: пустой набор опорных точек или узлов квадратуры" }
    }

    /** Квадратурный оператор `(L^N_h u)(t)` по значениям `y = u(eta)`. */
    private fun quadratureAt(t: Double, y: DoubleArray): Double {
        var acc = 0.0
        for (r in 0 until p) acc += bAgg[r] * op.kernel.k(t, pts[r], y[r])
        return acc
    }

    /**
     * Коэффициенты `P_theta v` по значениям `v` в опорных точках:
     * `c_j = theta_j(v) = sum_q beta_{j,q} v(eta_q)`.
     */
    private fun projectFromSupport(vAtPts: DoubleArray): DoubleArray = DoubleArray(dim) { k ->
        val vf = vfs[k]
        var s = 0.0
        for (q in vf.nodes.indices) s += vf.coeffs[q] * vAtPts[support.indexOf(k, q)]
        s
    }

    /** Значение сплайна с коэффициентами `c` в точке с предвычисленными `omega`. */
    private fun splineDot(c: DoubleArray, omegaRow: DoubleArray): Double {
        var s = 0.0
        for (i in 0 until dim) s += c[i] * omegaRow[i]
        return s
    }

    /**
     * Состояние правой части `G(u)` для заданных `(y, z)`: значения `G` в опорных
     * точках и узлах квадратуры, коэффициенты проекции разности `L u - L^N_h u`
     * (нужны для восстановления решения в произвольной точке) и копия `y`.
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
            rhs(pts[r]) + lambda * (quadAtPts[r] + splineDot(diffCoeffs, omegaAtPts[r]))
        }
        val gAtG = DoubleArray(ng) { k ->
            rhs(gNode[k]) + lambda * (quadratureAt(gNode[k], y) + splineDot(diffCoeffs, omegaAtG[k]))
        }
        return State(gAtPts, gAtG, diffCoeffs, y.copyOf())
    }

    /**
     * Якобиан `F'(y, z)` системы `F(y,z) = (y - G(eta), z - G(g))`, размер `(p+ng)^2`.
     *
     * Производные `G(t)` по `y_r` и `z_k`:
     *  * `dQ(t)/dy_r = b_r dK/du(t, eta_r, y_r)`  (квадратура);
     *  * `dE(t)/dz_k = gW_k dK/du(t, g_k, z_k)`   (точный оператор);
     *  * `dG(t)/dy_r = lambda [ dQ(t)/dy_r - sum_j omega_j(t) theta_j(dQ(.)/dy_r) ]`;
     *  * `dG(t)/dz_k = lambda [ sum_j omega_j(t) theta_j(dE(.)/dz_k) ]`.
     */
    private fun jacobian(y: DoubleArray, z: DoubleArray): Array<DoubleArray> {
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
        return ParallelAssembly.assembleRows(total, total, ctx.parallel) { row ->
            val out = DoubleArray(total)
            val tRow = if (row < p) pts[row] else gNode[row - p]
            val omegaRow = if (row < p) omegaAtPts[row] else omegaAtG[row - p]
            for (r in 0 until p) {
                val dq = bAgg[r] * op.kernel.dkdu(tRow, pts[r], y[r])
                var proj = 0.0
                for (j in 0 until dim) proj += omegaRow[j] * cQ[j][r]
                out[r] = -lambda * (dq - proj)
            }
            for (k in 0 until ng) {
                var proj = 0.0
                for (j in 0 until dim) proj += omegaRow[j] * cE[j][k]
                out[p + k] = -lambda * proj
            }
            out[row] += 1.0
            out
        }
    }

    /** Решает систему комбинированного метода; возвращает состояние `G` и сведения о сходимости. */
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
            methodName = "Ньютон (комбинированный Nyström Урысона)",
            iterations = run.performedSteps,
            maxIterations = solver.nystromMaxIter,
            residual = run.residual,
            tolerance = newtonTol,
        )
        // При сходимости последняя невязка измерена в возвращаемой точке (см. runNewtonIterations),
        // так что lastState отвечает именно ей; иначе пересчитываем явно.
        val state = if (run.converged && lastState != null) {
            lastState!!
        } else {
            evaluate(x.copyOfRange(0, p), x.copyOfRange(p, p + ng))
        }
        return state to run
    }

    /** `u^N_h(t) = G(u)(t)` — восстановление решения в произвольной точке. */
    private fun evalFrom(state: State): (Double) -> Double = { t ->
        rhs(t) + lambda * (quadratureAt(t, state.yAll) + basis.evalSpline(state.diffCoeffs, t))
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
        // Точный оператор на найденном u^N_h: его значения в узлах квадратуры — это G(g).
        val uNodes = state.gAtG
        return SolutionFunc(
            eval = { t -> rhs(t) + lambda * op.applyNodes(t, uNodes) },
            converged = run.converged,
            iterations = run.performedSteps,
            residual = run.residual,
        )
    }

    private companion object {
        /** Нижняя граница критерия останова (аналитический якобиан). */
        const val NEWTON_TOLERANCE_FLOOR = 1e-13
    }
}
