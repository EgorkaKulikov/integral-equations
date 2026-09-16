package solvers.uryson

import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.Grid
import splines.MinimalSplineBasis
import numerics.NumericsContext
import numerics.ParallelAssembly
import splines.functionals.ProjFunctionals
import solvers.core.SupportPoints

/**
 * Ядро коллокационных вычислений: вектор `Xi(c) = Theta_h(U x_h)` и якобиан
 * `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
 *
 * Используется и схемами второго рода (метод Ньютона), и регуляризованной схемой
 * первого рода (метод Гаусса–Ньютона).
 */
public class CollocationCore(
    public val basis: MinimalSplineBasis,
    public val funcs: ProjFunctionals,
    public val op: UrysohnOperator,
    public val ctx: NumericsContext = NumericsContext.default(),
) {
    public val grid: Grid = basis.grid
    public val n: Int = grid.n
    private val quad = op.quad
    private val kernel = op.kernel

    /**
     * Эталонные узлы и веса квадратуры на [-1,1], полученные ОДИН РАЗ.
     *
     * [GaussLegendre.refNodesWeights] возвращает КОПИИ массивов, а [bMatrix] вызывается
     * НА КАЖДОЙ итерации Ньютона (`UrysonSecondKindSolver.newtonStep`) и Гаусса–Ньютона
     * (`TikhonovSolver.solveFixedAlpha`). Получение узлов внутри [bMatrix] давало бы две
     * аллокации на итерацию на ровном месте; здесь копия делается однократно.
     *
     * Значения и порядок арифметики те же: массивы неизменны и читаются только на чтение.
     */
    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    /**
     * Различные опорные точки всех функционалов `theta_j` (узлы и середины интервалов)
     * в ПОРЯДКЕ ПЕРВОГО ВХОЖДЕНИЯ при обходе `j = -2..n-1`.
     *
     * Порядок — часть контракта, а не деталь реализации: от него зависят нумерация
     * строк `G` в [bMatrix] и порядок суммирования в [xiVector]. READ-ONLY по соглашению.
     */
    public val supportPts: DoubleArray

    /**
     * Индексация опорных точек по паре (номер функционала `j+2`, номер узла).
     *
     * Раньше здесь была `HashMap<Double, Int>` с поиском по ЗНАЧЕНИЮ точки: она требовала
     * побитового совпадения Double и работала лишь потому, что и заполнение, и чтение
     * шли из одного и того же массива `nodes` кэшированного функционала.
     */
    private val support: SupportPoints

    init {
        val (rn, rw) = quad.refNodesWeights()
        refNodes = rn
        refWeights = rw
        val vfs = Array(n + 2) { funcs.valueFunctional(it - 2) }
        support = SupportPoints.byFirstOccurrence(vfs, grid.breakpointInclusionEps)
        supportPts = support.points
    }

    /** Значения `(U x_h)` в опорных точках (по одному интегралу на точку). */
    public fun uAtSupport(c: DoubleArray): DoubleArray =
        DoubleArray(supportPts.size) { p -> op.apply(supportPts[p]) { s -> basis.evalSpline(c, s) } }

    /** Вектор `Xi(c)_j = theta_j(U x_h)`, `j = -2..n-1` (индекс массива `j+2`). */
    public fun xiVector(c: DoubleArray): DoubleArray {
        val uVals = uAtSupport(c)
        return DoubleArray(n + 2) { k ->
            val th = funcs.valueFunctional(k - 2)
            var s = 0.0
            for (q in th.nodes.indices) s += th.coeffs[q] * uVals[support.indexOf(k, q)]
            s
        }
    }

    /**
     * Якобиан `B(c)_{j,i} = theta_j(U'(x_h) omega_i)`.
     *
     * Вычисляется в два прохода: сначала `G[p][i] = \int dK/du(tau_p, s, x_h(s)) omega_i(s) ds`
     * для всех опорных точек `tau_p` за один обход узлов квадратуры (на каждом интервале
     * ненулевыми являются лишь три базисных сплайна), затем строки `B` собираются как
     * линейные комбинации строк `G` с коэффициентами функционалов.
     */
    public fun bMatrix(c: DoubleArray): DenseMatrix {
        val np = supportPts.size
        val g = DenseMatrix.zeros(np, n + 2)
        // Узлы/веса получены однократно в init: [refNodesWeights] отдаёт копии, а этот
        // метод вызывается на каждой итерации Ньютона / Гаусса–Ньютона.
        val nodes = refNodes
        val weights = refWeights
        for (m in 0 until n) {
            val lo = grid.x(m)
            val hi = grid.x(m + 1)
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in nodes.indices) {
                val s = mid + half * nodes[q]
                val weight = half * weights[q]
                val splineValue = basis.evalSpline(c, s)
                val activeOmega = basis.activeOmega(m, s) // omega_{m-2}, omega_{m-1}, omega_m
                for (p in 0 until np) {
                    val dk = kernel.dkdu(supportPts[p], s, splineValue) * weight
                    if (dk == 0.0) continue
                    g[p, m] += dk * activeOmega[0]
                    g[p, m + 1] += dk * activeOmega[1]
                    g[p, m + 2] += dk * activeOmega[2]
                }
            }
        }
        // Элементы B независимы: каждая задача пишет в свою ячейку, g только читается.
        // Порядок сложений сохранён дословно (по q возрастающе), поэтому числа те же.
        return ParallelAssembly.assembleDense(n + 2, n + 2, ctx.parallel) { k, i ->
            val th = funcs.valueFunctional(k - 2)
            var acc = 0.0
            for (q in th.nodes.indices) acc += th.coeffs[q] * g[support.indexOf(k, q), i]
            acc
        }
    }
}
