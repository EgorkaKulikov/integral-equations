package solvers.fredholm

import numerics.*
import splines.*
import splines.functionals.*
import splines.metrics.*

/**
 * Ядро K(t,s) линейного уравнения Фредгольма вместе с аналитическими частными
 * производными.
 *
 * @param k само ядро `K(t,s)`.
 * @param kT производная `K_t(t,s)`; требуется семействам функционалов
 *        де Бура–Фикса `xi^<1>`, `xi^<2>`.
 * @param kTT вторая производная `K_tt(t,s)`; требуется семейству `xi^<0>`.
 *
 * Значения по умолчанию равны нулю и допустимы ТОЛЬКО тогда, когда соответствующая
 * производная действительно тождественно нулевая, либо когда выбранное семейство
 * функционалов её не использует: иначе система будет построена неверно без какой-либо
 * диагностики.
 */
public class KernelF(
    public val k: (Double, Double) -> Double,
    public val kT: (Double, Double) -> Double = { _, _ -> 0.0 },
    public val kTT: (Double, Double) -> Double = { _, _ -> 0.0 },
)

/**
 * Оператор Фредгольма `(K u)(t) = \int_a^b K(t,s) u(s) ds` с постоянными пределами
 * интегрирования.
 *
 * При создании предвычисляются глобальные гауссовы узлы [gNode] и веса [gW]
 * составной квадратуры, так что `\int h = sum_k gW[k] * h(gNode[k])`. Это позволяет
 * многократно применять оператор к уже вычисленным в этих узлах значениям функции
 * (см. [applyNodes]), не пересчитывая её заново.
 *
 * @param kernel ядро уравнения.
 * @param grid сетка, задающая отрезок `[a,b]` и точки разбиения для составной квадратуры.
 * @param quad квадратурная формула Гаусса–Лежандра на ячейке.
 */
public class FredholmOperator(public val kernel: KernelF, public val grid: Grid, public val quad: GaussLegendre) {
    /**
     * Глобальные узлы составной квадратуры.
     *
     * READ-ONLY ПО СОГЛАШЕНИЮ: содержимое НЕЛЬЗЯ изменять. ГОРЯЧЕЕ поле — копия не
     * возвращается сознательно: массив читается во внутренних циклах [applyNodes],
     * [applyDerivNodes], [applyDeriv2Nodes] и при сборке матриц — копирование на каждом
     * обращении дало бы квадратичный рост аллокаций. Записей в проекте нет.
     */
    public val gNode: DoubleArray

    /** Веса квадратуры при узлах [gNode]. READ-ONLY по соглашению (горячее, см. [gNode]). */
    public val gW: DoubleArray

    init {
        val (rn, rw) = quad.refNodesWeights()
        val bp = grid.breakpoints
        val nodes = ArrayList<Double>()
        val ws = ArrayList<Double>()
        for (m in 0 until bp.size - 1) {
            val lo = bp[m]; val hi = bp[m + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo); val mid = 0.5 * (hi + lo)
            for (qi in rn.indices) { nodes.add(mid + half * rn[qi]); ws.add(half * rw[qi]) }
        }
        gNode = nodes.toDoubleArray()
        gW = ws.toDoubleArray()
    }

    /** (\mathcal K u)(t) для произвольной u(s). */
    public fun apply(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.k(t, s) * u(s) }

    /** d/dt (\mathcal K u)(t) = \int_a^b dK/dt(t,s) u(s) ds (для xi-функционалов). */
    public fun applyDeriv(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kT(t, s) * u(s) }

    /** d^2/dt^2 (\mathcal K u)(t) = \int_a^b d^2K/dt^2(t,s) u(s) ds (для xi^<0>). */
    public fun applyDeriv2(t: Double, u: (Double) -> Double): Double =
        quad.integrate(grid.breakpoints) { s -> kernel.kTT(t, s) * u(s) }

    /** (\mathcal K u)(tau) по предвычисленным значениям u в глобальных узлах. */
    public fun applyNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.k(tau, gNode[k]) * uNodes[k]
        return s
    }

    /** d/dt (\mathcal K u)(tau) по предвычисленным uNodes. */
    public fun applyDerivNodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kT(tau, gNode[k]) * uNodes[k]
        return s
    }

    /** d^2/dt^2 (\mathcal K u)(tau) по предвычисленным uNodes (для xi^<0>). */
    public fun applyDeriv2Nodes(tau: Double, uNodes: DoubleArray): Double {
        var s = 0.0
        for (k in gNode.indices) s += gW[k] * kernel.kTT(tau, gNode[k]) * uNodes[k]
        return s
    }
}
