package verification

import org.junit.jupiter.api.Tag
import problems.wsie.WeaklySingularProblem
import problems.wsie.WeaklySingularType
import splines.Grid
import verification.WsieOracles.bd
import java.io.File
import java.math.MathContext
import java.math.RoundingMode
import java.util.Locale
import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * VALIDATION ITEM 9, reduced (new-wsie r1, role 12, round V2): independent implementation of the BASE scheme
 * u_h = P_theta(f + cL L u_h) and recomputation of E_h for probe cells of `wsieProbe` (run1 / run3a).
 *
 * Independent of `src/main`: the basis is the own Cox-de Boor in g (g = id for B, g = t^(1 - alpha) for G, knots
 * g(x_i) with triple ends), the functionals theta / theta-tau come from the own BigDecimal biorthogonality solve
 * ([WsieOracles.ThetaOracle], item 6), the entries theta_j(L omega_i) from the own tanh-sinh rule
 * ([WsieOracles.TanhSinh]) on pieces split at the breakpoints and at t, graded geometrically towards t so that
 * the distance to the kernel singularity is never below the piece length, and the system is solved by the own
 * dense LU ([WsieOracles.solveDense]). Shared with the code under test: the grid ([Grid], validated in item 5),
 * the problem data rhs and exact solution (validated in items 3 and 8). E_h is the maximum of |u - u_h| over
 * the control set of `WsieProbeTool` (knots, midpoints, 8 Gauss-Legendre points per cell, 200 log points
 * t = 10^(-12 + 12k/199), mirrored for Fredholm), rebuilt here.
 *
 * Tolerance (plan, item 9): |E_h^lib - E_h^ref| <= 10 * condInf * 1e-13 * max(1, ||u||_inf), with condInf the
 * condInf of the probe row (plan, item 9); the exact infinity-norm condition number of the own I - M is reported. The oracle quadrature is self-checked by halving
 * the tanh-sinh step: the change of theta_j(L omega_i) must stay below 1e-14 * max(1, max|M|).
 */
@Tag("slow")
class WsieReferenceSchemeCrossCheckTest {

    private val mc = MathContext(40, RoundingMode.HALF_EVEN)

    /** A probe cell: the library value of E_h and condInf (run1/run3a probe.tsv, integral-equations 61af862). */
    private data class Cell(
        val problem: String, val space: String, val tau: Boolean, val grid: String, val n: Int,
        val ehLib: Double, val condLib: Double, val source: String,
    )

    private val cells = listOf(
        Cell("V-c", "B", false, "uniform", 64, 0.0063748617053098755, 16.697028694115890, "run1"),
        Cell("V-c", "G", false, "uniform", 64, 2.3466368302749174e-05, 19.447832277465398, "run1"),
        Cell("V-b", "B", false, "uniform", 64, 4.9616162178800494e-08, 120.13627063837701, "run1"),
        Cell("V-b", "G", false, "uniform", 64, 2.2190754744633168e-05, 162.66058837182345, "run1"),
        Cell("V-b", "B", false, "uniform", 8, 6.0318264197722904e-05, 73.880122109207480, "run1"),
        Cell("V-b", "B", false, "uniform", 16, 5.3957903527468430e-06, 96.323539694486060, "run1"),
        Cell("V-a", "B", false, "uniform", 8, 0.21493772898902820, 73.880122109207480, "run1"),
        Cell("V-a", "B", false, "power:3", 64, 0.0018648562498242427, 24.136081988183125, "run1"),
        Cell("V-a", "B", false, "power:4", 64, 0.0046113656458004470, 21.799638134343720, "run1"),
        Cell("V-a", "G", false, "power:4", 64, 0.0071130847099638570, 21.766474146425605, "run1"),
        Cell("V-a", "G", true, "power:4", 64, 0.0068685918832827040, 21.770677218268535, "run1"),
        Cell("V-c", "B", false, "power:4", 64, 6.8434572849440660e-05, 5.2077398848708570, "run1"),
        Cell("V-c", "G", false, "power:3", 64, 4.5116762810160080e-05, 5.5181654114027180, "run1"),
        Cell("F-a", "B", false, "power:3", 32, 0.0015630521938225783, 6.7761376142778300, "run1"),
        Cell("V-c", "B", false, "uniform", 64, 0.0063748617053100975, 16.697028694115890, "run3a refine=20"),
        Cell("V-b", "B", false, "uniform", 64, 4.9616162178800494e-08, 120.13627063837698, "run3a refine=20"),
    )

    private class Ref(val eh: Double, val cond: Double, val uMax: Double, val quadChange: Double, val mMax: Double)

    /** Own base scheme on the breakpoints [x] for problem [p]; [beta] = null is the space B. */
    private class Scheme(val p: WeaklySingularProblem, val x: DoubleArray, val beta: Double?) {
        val n = x.size - 1
        private val y = DoubleArray(n + 5) { i -> g(x[(i - 2).coerceIn(0, n)]) }

        fun g(t: Double): Double = if (beta == null) t else t.pow(beta)

        fun cell(t: Double): Int {
            var lo = 0; var hi = n - 1
            while (lo < hi) { val m = (lo + hi + 1) / 2; if (x[m] <= t) lo = m else hi = m - 1 }
            return lo
        }

        fun spline(c: DoubleArray, t: Double): Double {
            val k = cell(t)
            val v = WsieOracles.quadraticBSplinesD(y, k, g(t))
            return c[k] * v[0] + c[k + 1] * v[1] + c[k + 2] * v[2]
        }

        /** (L omega_i)(t), i = -2..n-1 at index i + 2; k = 1, the kernel is |t - s|^(-alpha). */
        fun images(t: Double, ts: WsieOracles.TanhSinh): DoubleArray {
            val out = DoubleArray(n + 2)
            val end = if (p.type == WeaklySingularType.VOLTERRA) t else x[n]
            for (k in 0 until n) {
                val lo = x[k]
                if (lo >= end) break
                val hi = minOf(x[k + 1], end)
                if (t > lo && t < hi) { graded(k, lo, t, t, out, ts); graded(k, t, hi, t, out, ts) } else graded(k, lo, hi, t, out, ts)
            }
            return out
        }

        /** Splits [lo, hi] geometrically towards t (t <= lo or t >= hi) so that dist(t, piece) >= |piece|. */
        private fun graded(k: Int, lo: Double, hi: Double, t: Double, out: DoubleArray, ts: WsieOracles.TanhSinh) {
            if (t == lo || t == hi) { piece(k, lo, hi, t, out, ts); return }
            if (t > hi) {
                var b = hi
                while (b > lo) { val len = t - b; val a = if (len >= b - lo) lo else b - len; piece(k, a, b, t, out, ts); b = a }
            } else {
                var a = lo
                while (a < hi) { val len = a - t; val b = if (len >= hi - a) hi else a + len; piece(k, a, b, t, out, ts); a = b }
            }
        }

        private fun piece(k: Int, a: Double, b: Double, t: Double, out: DoubleArray, ts: WsieOracles.TanhSinh) {
            ts.forEach(a, b) { s, dLo, dHi, w ->
                val dist = when { t == b -> dHi; t == a -> dLo; t > b -> t - s; else -> s - t }
                val kw = w * dist.pow(-p.alpha)
                val v = WsieOracles.quadraticBSplinesD(y, k, g(s))
                out[k] += kw * v[0]; out[k + 1] += kw * v[1]; out[k + 2] += kw * v[2]
            }
        }
    }

    private fun gridOf(c: Cell, p: WeaklySingularProblem): Grid = when {
        c.grid == "uniform" -> Grid.uniform(c.n)
        p.type == WeaklySingularType.VOLTERRA -> Grid.power(c.n, r = c.grid.removePrefix("power:").toDouble())
        else -> Grid.symmetricPower(c.n, r = c.grid.removePrefix("power:").toDouble())
    }

    /** Control set of WsieProbeTool, rebuilt: knots, midpoints, 8 GL points per cell, 200 log points (mirrored for F). */
    private fun controlSet(x: DoubleArray, fredholm: Boolean): DoubleArray {
        val gl = WsieOracles.gaussLegendreNodes(8)
        val pts = ArrayList<Double>()
        for (i in 0 until x.size - 1) {
            val lo = x[i]; val hi = x[i + 1]
            pts += lo; pts += 0.5 * (lo + hi)
            for (z in gl) pts += 0.5 * (lo + hi) + 0.5 * (hi - lo) * z
        }
        pts += x.last()
        val a = x.first(); val b = x.last()
        for (k in 0 until 200) {
            val t = 10.0.pow(-12.0 + 12.0 * k / 199.0)
            pts += a + t * (b - a)
            if (fredholm) pts += b - t * (b - a)
        }
        return pts.filter { it >= a && it <= b }.distinct().sorted().toDoubleArray()
    }

    private fun compute(c: Cell): Ref {
        val p = WeaklySingularProblem.ALL.first { it.name == c.problem }
        val grid = gridOf(c, p)
        val x = DoubleArray(c.n + 1) { grid.x(it) }
        val beta = if (c.space == "G") 1.0 - p.alpha else null
        val scheme = Scheme(p, x, beta)
        val oracle = WsieOracles.ThetaOracle(x, beta?.let { bd(it) }, mc)
        val dim = c.n + 2
        val funcs = (-2..c.n - 1).map { j ->
            val (pts, w) = oracle.theta(j, c.tau)
            DoubleArray(pts.size) { pts[it].toDouble() } to DoubleArray(w.size) { w[it].toDouble() }
        }
        val coarse = WsieOracles.TanhSinh(1.0 / 16); val fine = WsieOracles.TanhSinh(1.0 / 32)
        val imgC = HashMap<Double, DoubleArray>(); val imgF = HashMap<Double, DoubleArray>()
        for ((pts, _) in funcs) for (t in pts) if (t !in imgF) { imgC[t] = scheme.images(t, coarse); imgF[t] = scheme.images(t, fine) }
        val a = Array(dim) { DoubleArray(dim) }
        val rhs = DoubleArray(dim)
        var quadChange = 0.0; var mMax = 0.0
        for ((row, f) in funcs.withIndex()) {
            val (pts, w) = f
            a[row][row] = 1.0
            for (i in 0 until dim) {
                var mf = 0.0; var mcz = 0.0
                for (q in pts.indices) { mf += w[q] * imgF.getValue(pts[q])[i]; mcz += w[q] * imgC.getValue(pts[q])[i] }
                a[row][i] -= p.cL * mf
                quadChange = maxOf(quadChange, abs(p.cL * (mf - mcz))); mMax = maxOf(mMax, abs(p.cL * mf))
            }
            for (q in pts.indices) rhs[row] += w[q] * p.rhs(pts[q])
        }
        val coef = WsieOracles.solveDense(a, rhs)
        val normA = a.maxOf { r -> r.sumOf { abs(it) } }
        val inv = Array(dim) { e -> WsieOracles.solveDense(a, DoubleArray(dim) { if (it == e) 1.0 else 0.0 }) }
        val normInv = (0 until dim).maxOf { r -> (0 until dim).sumOf { e -> abs(inv[e][r]) } }
        var eh = 0.0; var uMax = 0.0
        for (t in controlSet(x, p.type == WeaklySingularType.FREDHOLM)) {
            val u = p.exact(t)
            uMax = maxOf(uMax, abs(u)); eh = maxOf(eh, abs(u - scheme.spline(coef, t)))
        }
        return Ref(eh, normA * normInv, uMax, quadChange, mMax)
    }

    @Test
    fun `base scheme E_h of probe cells agrees with an independent implementation (own basis, theta, quadrature, LU)`() {
        val rows = ArrayList<String>()
        val failures = ArrayList<String>()
        for (c in cells) {
            val r = compute(c)
            val tol = 10.0 * c.condLib * 1e-13 * maxOf(1.0, r.uMax)
            val dev = abs(r.eh - c.ehLib)
            val quadOk = r.quadChange <= 1e-14 * maxOf(1.0, r.mMax)
            if (!quadOk) failures += "ORACLE $c: tanh-sinh h=1/16 vs 1/32 change ${r.quadChange}"
            if (dev > tol) failures += "$c: E_h ref ${r.eh} vs lib ${c.ehLib}, |dev| $dev > tol $tol"
            rows += String.format(
                Locale.ROOT, "%s\t%s\t%s\t%s\t%d\t%s\t%.17g\t%.17g\t%.3e\t%.3e\t%.3e\t%.6g\t%.6g\t%.3e",
                c.problem, c.space, if (c.tau) "theta-tau" else "theta", c.grid, c.n, c.source.replace(' ', '_'),
                c.ehLib, r.eh, dev, dev / c.ehLib, tol, c.condLib, r.cond, r.quadChange,
            )
            println("item9 ${rows.last()}")
        }
        val out = File("build/wsie-validation").apply { mkdirs() }
        File(out, "item9-base.tsv").writeText(
            "problem\tspace\tfamily\tgrid\tn\tsource\tEhLib\tEhRef\tabsDev\trelDev\ttol\tcondInfLib\tcondInfRef\tquadChange\n" +
                rows.joinToString("\n") + "\n",
        )
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }
}
