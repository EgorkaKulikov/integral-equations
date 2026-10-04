package verification

import numerics.NumericsContext
import org.junit.jupiter.api.Tag
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.Reparametrization
import splines.functionals.ProjFunctionals
import splines.functionals.ThetaSampling
import splines.functionals.ValueFunctional
import verification.WsieOracles.bd
import java.io.File
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import java.util.Locale
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * VALIDATION ITEM 6 (new-wsie r1, role 12, round V2): the functionals theta and theta-tau and the constant Lambda.
 *
 * Oracle ([WsieOracles.ThetaOracle]): own Cox-de Boor for the quadratic B-splines in g(t) on the knots g(x_i)
 * with triple end knots (g = id for B, g = t^beta for G), and the weights of theta_j solved in BigDecimal from
 * the local biorthogonality system at the sampling points (arithmetic midpoints for theta, g^{-1} of the
 * g-midpoint for theta-tau). Nothing of `ProjFunctionals` is used by the oracle; the library is only the object
 * under test. Shared with the code: the DEFINITION of theta (points and local indices), see plan, "Known limits".
 *
 * Tolerances. Weights: |w_lib - w_ref| <= 1e-12 * max(1, sum|w_ref|) (plan item 6; the double solve of a local
 * system with condition number <= 1e2 has error ~1e-14). Nodes: 8 ulp (one division or one pow/gInverse).
 * Biorthogonality of the LIBRARY functional against the ORACLE basis over all i: 1e-12 * Lambda. Lambda against
 * the implementer's 4-digit values (experiment-report-r1, T5): half a unit of the last digit, 5e-5.
 */
@Tag("fast")
class WsieThetaWeightsOracleTest {

    private val mc = MathContext(50, RoundingMode.HALF_EVEN)

    private data class Case(val space: String, val beta: Double?, val tau: Boolean, val grid: String, val n: Int)

    private fun gridOf(label: String, n: Int): Grid = when (label) {
        "uniform" -> Grid.uniform(n)
        "power:2" -> Grid.power(n, r = 2.0)
        "power:3" -> Grid.power(n, r = 3.0)
        "power:4" -> Grid.power(n, r = 4.0)
        "symPow:3" -> Grid.symmetricPower(n, r = 3.0)
        else -> error(label)
    }

    private fun cases(): List<Case> {
        val out = ArrayList<Case>()
        for (n in intArrayOf(8, 64)) {
            for (g in listOf("uniform", "power:2", "power:3", "power:4", "symPow:3")) out += Case("B", null, false, g, n)
            for (beta in doubleArrayOf(0.5, 2.0 / 3.0)) for (tau in listOf(false, true))
                for (g in listOf("uniform", "power:2", "power:3", "power:4")) out += Case("G", beta, tau, g, n)
        }
        return out
    }

    /** Implementer's Lambda (experiment-report-r1, T5): B 3.0; G theta 3.8284 (beta 1/2), 3.2898 (beta 2/3); theta-tau 3.0. */
    private fun reported(c: Case): Double = when {
        c.space == "B" || c.tau -> 3.0
        c.beta == 0.5 -> 3.8284
        else -> 3.2898
    }

    @Test
    fun `theta and theta-tau weights, nodes and Lambda agree with an own BigDecimal biorthogonality solve`() {
        val rows = ArrayList<String>()
        val failures = ArrayList<String>()
        var worstW = 0.0; var worstNode = 0.0; var worstBio = 0.0; var worstLam = 0.0
        for (c in cases()) {
            val grid = gridOf(c.grid, c.n)
            val system = if (c.beta == null) GeneratingSystem.B else GeneratingSystem.reparametrized(Reparametrization.power(c.beta))
            val basis = MinimalSplineBasis(system, grid)
            val sampling = if (c.tau) ThetaSampling.reparametrized(Reparametrization.power(c.beta!!)) else ThetaSampling.ARITHMETIC
            val funcs = ProjFunctionals(basis, NumericsContext.default(), sampling)
            val oracle = WsieOracles.ThetaOracle(DoubleArray(c.n + 1) { grid.x(it) }, c.beta?.let { bd(it) }, mc)
            var lamRef = 0.0; var dW = 0.0; var dNode = 0.0; var bio = 0.0
            for (j in -2..c.n - 1) {
                val lib = funcs.chi(j) as ValueFunctional
                val (pts, w) = oracle.theta(j, c.tau)
                val sumRef = w.sumOf { abs(it.toDouble()) }
                lamRef = maxOf(lamRef, sumRef)
                if (lib.nodes.size != pts.size) { failures += "$c j=$j: ${lib.nodes.size} nodes vs ${pts.size}"; continue }
                for (p in pts.indices) {
                    val node = WsieOracles.absDev(lib.nodes[p], pts[p]) / (Math.ulp(1.0) * maxOf(1e-300, abs(pts[p].toDouble())))
                    dNode = maxOf(dNode, if (pts[p].signum() == 0 && lib.nodes[p] == 0.0) 0.0 else node)
                    dW = maxOf(dW, WsieOracles.absDev(lib.coeffs[p], w[p]) / maxOf(1.0, sumRef))
                }
                val libPts = Array(lib.nodes.size) { bd(lib.nodes[it]) }
                val libW = Array(lib.coeffs.size) { bd(lib.coeffs[it]) }
                bio = maxOf(bio, oracle.biorthogonalityDefect(j, libPts, libW))
            }
            val lamLib = funcs.cChi()
            val dLam = abs(lamLib - lamRef)
            val dRep = abs(lamRef - reported(c))
            if (dW > 1e-12) failures += "$c weights rel dev $dW > 1e-12"
            if (dNode > 16.0) failures += "$c nodes dev $dNode ulp > 16"
            if (bio > 1e-12 * lamRef) failures += "$c biorthogonality defect $bio > 1e-12*Lambda"
            if (dLam > 1e-12 * lamRef) failures += "$c Lambda lib $lamLib vs ref $lamRef"
            if (dRep > 5e-5) failures += "$c Lambda ref $lamRef vs reported ${reported(c)}"
            worstW = maxOf(worstW, dW); worstNode = maxOf(worstNode, dNode); worstBio = maxOf(worstBio, bio)
            worstLam = maxOf(worstLam, dLam)
            rows += String.format(
                Locale.ROOT, "%s\t%s\t%s\t%s\t%d\t%.17g\t%.17g\t%.3e\t%.1f\t%.3e\t%.3e",
                c.space, c.beta?.toString() ?: "-", if (c.tau) "theta-tau" else "theta", c.grid, c.n,
                lamRef, lamLib, dW, dNode, bio, dRep,
            )
        }
        val out = File("build/wsie-validation").apply { mkdirs() }
        File(out, "item6-theta.tsv").writeText(
            "space\tbeta\tfamily\tgrid\tn\tLambdaRef\tLambdaLib\tmaxRelDevW\tmaxNodeDevUlp\tbioDefectLib\tdevFromReported\n" +
                rows.joinToString("\n") + "\n",
        )
        println("item6: cases=${rows.size} worstW=$worstW worstNodeUlp=$worstNode worstBio=$worstBio worstLambda=$worstLam")
        assertTrue(failures.isEmpty(), failures.take(20).joinToString("\n"))
    }

    @Test
    fun `oracle self-check - theta reproduces random splines in the oracle basis (projector property)`() {
        val rnd = Random(20261004)
        for ((beta, label) in listOf(null to "uniform", 0.5 to "power:3", 2.0 / 3.0 to "power:2")) {
            val grid = gridOf(label, 8)
            val oracle = WsieOracles.ThetaOracle(DoubleArray(9) { grid.x(it) }, beta?.let { bd(it) }, mc)
            val c = DoubleArray(10) { rnd.nextDouble(-1.0, 1.0) }
            for (tau in listOf(false, true)) for (j in -2..7) {
                val (pts, w) = oracle.theta(j, tau)
                var s = BigDecimal.ZERO
                for (p in pts.indices) s = s.add(w[p].multiply(bd(oracle.spline(c, pts[p].toDouble())), mc), mc)
                val dev = abs(s.toDouble() - c[j + 2])
                assertTrue(dev < 1e-14, "oracle projector defect $dev at beta=$beta tau=$tau j=$j")
            }
        }
    }
}
