package solvers.uryson

import kotlin.math.abs
import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.Grid
import numerics.LinearAlgebra
import splines.MinimalSplineBasis
import numerics.NumericsContext

/**
 * Сплайн-пространство: веса, матрица Грама и интегральные характеристики базиса.
 *
 * Объединяет базис, сетку и квадратуру, предоставляя величины, общие для схем
 * второго и первого рода: веса дискретной нормы, веса метода Nyström и матрицу
 * стабилизатора Тихонова.
 *
 * @param basis базис минимальных сплайнов.
 * @param quad квадратура для интегралов по сеточным интервалам.
 */
public class SplineSpace(
    public val basis: MinimalSplineBasis,
    public val quad: GaussLegendre,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    public val dim: Int = n + 2

    /**
     * Допуск отбрасывания узлов, совпавших с концами подынтервала.
     *
     * Значение берётся из ЕДИНОГО ИСТОЧНИКА [Grid.breakpointInclusionEps] (там же —
     * обоснование относительности и оговорка про мелкие отрезки), а не вычисляется
     * здесь повторно: раньше та же формула дублировалась в `VolterraOperator`, причём
     * была записана иначе (`coerceAtLeast` против `maxOf`).
     *
     * ОБЪЯВЛЕНО ДО [gramRInternal] НЕ СЛУЧАЙНО: инициализаторы свойств в Kotlin
     * выполняются в порядке объявления, а `buildGram()` вызывает [subBreakpoints],
     * читающий этот порог. При объявлении ПОСЛЕ порог был бы ещё `0.0`, и матрица
     * Грама считалась бы с ДРУГИМ разбиением — тихое изменение чисел без ошибки.
     */
    private val breakpointEps: Double = grid.breakpointInclusionEps

    /**
     * Веса дискретной нормы `w_j = (x_{j+3} - x_j)/3`, `j = -2..n-1` (индекс массива `j+2`).
     *
     * Пропорциональны длине носителя сплайна `omega_j`; их сумма равна длине отрезка,
     * что делает дискретную норму согласованной с `L^2`.
     */
    private val weightsInternal: DoubleArray = DoubleArray(dim) { (grid.x(it - 2 + 3) - grid.x(it - 2)) / 3.0 }

    /**
     * Веса дискретной нормы (КОПИЯ: мутация результата не затрагивает пространство).
     *
     * Поле ХОЛОДНОЕ: все вызывающие читают его один раз и сохраняют в своё поле
     * (см. `TikhonovSolver`), поэтому копия не попадает в горячий цикл.
     */
    public val weights: DoubleArray get() = weightsInternal.copyOf()

    /** Веса метода Nyström `W_j = \int_a^b omega_j(s) ds`. */
    private val wIntInternal: DoubleArray = DoubleArray(dim) { k ->
        val j = k - 2
        quad.integrate(grid.breakpoints) { t -> basis.omega(j, t) }
    }

    /**
     * Веса метода Nyström (КОПИЯ, см. обоснование у [weights]).
     *
     * Единственный боевой читатель — `UrysonSecondKindSolver.nystrom`, где значение берётся
     * в локальную переменную ДО цикла Ньютона.
     */
    public val wInt: DoubleArray get() = wIntInternal.copyOf()

    /**
     * Матрица Грама стабилизатора: `[R]_{i,j} = \int (omega_i omega_j + omega_i' omega_j') ds`.
     *
     * Это матрица скалярного произведения пространства Соболева `W^{1,2}`, то есть
     * стабилизатор Тихонова штрафует и саму функцию, и её первую производную.
     * Матрица симметрична, положительно определена и полосная: `|i-j| <= 2`, поскольку
     * носители сплайнов, отстоящих дальше, не пересекаются.
     */
    private val gramRInternal: DenseMatrix = buildGram()

    /**
     * Матрица Грама стабилизатора (КОПИЯ, см. обоснование у [weights]).
     *
     * У [DenseMatrix] значения лежат в ОДНОМ плоском массиве, поэтому различие
     * «поверхностная/глубокая копия» здесь исчезает вместе со строками-массивами:
     * `copy()` копирует всё содержимое, и сквозной записи в поле больше нет.
     */
    public val gramR: DenseMatrix get() = gramRInternal.copy()

    private fun buildGram(): DenseMatrix {
        val r = DenseMatrix.zeros(dim, dim)
        for (ki in 0 until dim) {
            val i = ki - 2
            for (kj in ki until dim) {
                val j = kj - 2
                if (abs(i - j) > 2) continue // полосная структура: носители не пересекаются
                val lo = maxOf(grid.x(i), grid.x(j))
                val hi = minOf(grid.x(i + 3), grid.x(j + 3))
                if (hi <= lo) continue
                val sub = subBreakpoints(lo, hi)
                val value = quad.integrate(sub) { t ->
                    basis.omega(i, t) * basis.omega(j, t) +
                        basis.omegaDeriv(i, t) * basis.omegaDeriv(j, t)
                }
                r[ki, kj] = value
                r[kj, ki] = value
            }
        }
        return r
    }

    /** Узлы сетки внутри `[lo, hi]` плюс концы — разбиение для составной квадратуры. */
    private fun subBreakpoints(lo: Double, hi: Double): DoubleArray {
        val pts = ArrayList<Double>()
        pts.add(lo)
        for (k in 0..n) {
            val xk = grid.x(k)
            if (xk > lo + breakpointEps && xk < hi - breakpointEps) pts.add(xk)
        }
        pts.add(hi)
        return pts.toDoubleArray()
    }

    /** Сумма весов `sum_j w_j`; по построению должна равняться длине отрезка `b - a`. */
    public fun weightsSum(): Double = weightsInternal.sum()

    /** Квадратичная форма стабилизатора `Omega(x_h) = c^T R_h c`. */
    public fun omegaReg(c: DoubleArray): Double {
        // Внутри класса читаем бэкинг-поле напрямую: копия здесь была бы лишней.
        val rc = LinearAlgebra.matVec(gramRInternal, c, ctx.backend)
        var s = 0.0
        for (i in c.indices) s += c[i] * rc[i]
        return s
    }

}
