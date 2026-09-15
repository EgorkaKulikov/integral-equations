package solvers.uryson

import numerics.GaussLegendre
import splines.Grid

/**
 * Ядро нелинейного уравнения Урысона `K(t,s,u)` и его частная производная по `u`.
 *
 * Производная `dK/du` нужна для производной Фреше оператора, а через неё — для
 * аналитического якобиана метода Ньютона.
 */
public interface Kernel {
    /** Значение ядра `K(t, s, u)`. */
    public fun k(t: Double, s: Double, u: Double): Double

    /** Частная производная `dK/du(t, s, u)`. */
    public fun dkdu(t: Double, s: Double, u: Double): Double
}

/**
 * Нелинейный интегральный оператор Урысона `(U x)(t) = \int_a^b K(t,s,x(s)) ds`.
 *
 * Интегралы вычисляются составной квадратурой Гаусса–Лежандра по сеточным интервалам.
 *
 * @param kernel ядро уравнения вместе с производной по `u`.
 * @param grid сетка, задающая отрезок интегрирования и разбиение для квадратуры.
 * @param quad квадратурная формула.
 */
public class UrysohnOperator(public val kernel: Kernel, public val grid: Grid, public val quad: GaussLegendre) {
    /** Значение `(U x)(t)` для произвольной функции `x(s)`. */
    public fun apply(t: Double, x: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s, x(s)) }

    /**
     * Производная Фреше `(U'(x) h)(t) = \int_a^b dK/du(t,s,x(s)) h(s) ds`.
     *
     * Задаёт линеаризацию оператора в точке `x`; именно по этой формуле выводится
     * якобиан [CollocationCore.bMatrix], который вычисляется эффективнее — за один
     * проход по узлам квадратуры сразу для всех базисных функций.
     */
    public fun frechet(t: Double, x: (Double) -> Double, h: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.dkdu(t, s, x(s)) * h(s) }

    /**
     * Глобальный набор узлов квадратуры по всей сетке: `\int h = sum_k gW[k] h(gNode[k])`.
     *
     * Предвычисление позволяет в схемах Кулкарни и Nyström не пересобирать разбиение
     * при каждом вычислении вложенных интегралов.
     *
     * READ-ONLY ПО СОГЛАШЕНИЮ: содержимое НЕЛЬЗЯ изменять. ГОРЯЧЕЕ поле — копия не
     * возвращается сознательно: массив читается в цикле [applyNodes] и на каждой
     * итерации квази-Ньютона. Записей в проекте нет.
     */
    public val gNode: DoubleArray

    /** Веса квадратуры, соответствующие узлам [gNode]. READ-ONLY по соглашению (горячее). */
    public val gW: DoubleArray

    init {
        val (referenceNodes, referenceWeights) = quad.refNodesWeights()
        val breakpoints = grid.breakpoints
        val nodes = ArrayList<Double>()
        val weights = ArrayList<Double>()
        for (m in 0 until breakpoints.size - 1) {
            val lo = breakpoints[m]
            val hi = breakpoints[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in referenceNodes.indices) {
                nodes.add(mid + half * referenceNodes[q])
                weights.add(half * referenceWeights[q])
            }
        }
        gNode = nodes.toDoubleArray()
        gW = weights.toDoubleArray()
    }

    /** Значение `(U x)(tau)` по предвычисленным значениям `x` в узлах [gNode]. */
    public fun applyNodes(tau: Double, xNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k], xNodes[k])
        return s
    }
}
