package verification

import java.io.File
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.test.Test
import numerics.GaussLegendre
import numerics.LinearAlgebra
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF
import solvers.volterra.KernelV
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * THE PART OF THE ARTICLE TABLES THAT [Sec4VerificationTool] DOES NOT COVER.
 *
 * 1. `tab:sol-freq` — a fixed kernel (M2, w0 = 40; M3, w0 = 30), a fixed generating system
 *    T_{w0} chosen by the kernel, and a varying exact solution u1..u7; the ratio
 *    G = E_h[theta(B)] / E_h[theta(T_{w0})] for the four schemes on the uniform grid n = 64.
 *    The right-hand side is recomputed for every solution by the method of manufactured solutions.
 *    With the envelope p(t) = 1/(1+t):
 *      u1 = p;  u2, u3, u4 = p cos(c w0 t), c = 1/2, 1, 3/2;
 *      u5 = p (cos w0 t + 0.3 cos 2 w0 t);
 *      u6 = p (cos w0 t + cos 0.6 w0 t)            (two incommensurable harmonics);
 *      u7 = p cos(w0 t + 0.25 w0 t^2)               (chirp, frequency from w0 to 1.5 w0).
 *    The derivatives of the solutions are analytic.
 * 2. `rem:m4-split` — the regularization component ||u* - u_alpha||_inf of Problem M4
 *    (omega = 20) for alpha = 1e-6, 1e-8, 1e-10, 1e-12, computed through the exact identity
 *    u* - u_alpha = alpha (alpha I + K)^{-1} u* by an independent Nystrom method on composite
 *    Gauss--Legendre quadrature (Nsub cells x 8 nodes); splines take no part in it.
 *
 * A GENERATOR OF NUMBERS rather than a check: run by the task `paperTables` together with
 * [Sec4VerificationTool]; the results are written into build/sec4/ next to the other tables.
 * System properties: `paper.solset` = all (u1..u7, default) | published (u1..u5) | new (u6, u7);
 * `paper.solfreq.n` (default 64); `paper.m4split.nsub` (default 64,128,256);
 * `sec4.quad`, `sec4.refine` as in [Sec4VerificationTool].
 */
class PaperTablesTool {

    private val QUAD: Int = (System.getProperty("sec4.quad") ?: "8").toInt()
    private val REFINE: Int = (System.getProperty("sec4.refine") ?: "100").toInt()

    // ==================== generating systems (the same as in Sec4VerificationTool) ====================

    private fun sysT(w: Double) = GeneratingSystem(
        name = "T$w",
        rho = { t -> sin(w * t) }, sigma = { t -> cos(w * t) },
        rhoD = { t -> w * cos(w * t) }, sigmaD = { t -> -w * sin(w * t) },
        rhoDD = { t -> -w * w * sin(w * t) }, sigmaDD = { t -> -w * w * cos(w * t) },
    )

    // ==================== exact solutions ====================

    private class Sol(
        val name: String,
        val u: (Double) -> Double,
        val uD: (Double) -> Double,
        val uDD: (Double) -> Double,
    )

    /** Envelope p of the exact solution with p' and p''. */
    private class Env(val p: (Double) -> Double, val pD: (Double) -> Double, val pDD: (Double) -> Double)

    /** Envelope of the solutions u1..u7 of tab:sol-freq: p(t) = 1/(1+t), p' = -p^2, p'' = 2p^3. */
    private val envSol = Env(
        { t -> 1.0 / (1.0 + t) }, { t -> -1.0 / ((1.0 + t) * (1.0 + t)) },
        { t -> 2.0 / ((1.0 + t) * (1.0 + t) * (1.0 + t)) },
    )

    /** u = p(t) sum_k c_k cos(l_k t) (Leibniz rule for the derivatives). */
    private fun cosSum(name: String, env: Env, terms: List<Pair<Double, Double>>): Sol = Sol(
        name,
        u = { t -> var s = 0.0; for ((c, l) in terms) s += c * cos(l * t); env.p(t) * s },
        uD = { t ->
            var s = 0.0
            for ((c, l) in terms) s += c * (env.pD(t) * cos(l * t) - l * env.p(t) * sin(l * t))
            s
        },
        uDD = { t ->
            var s = 0.0
            for ((c, l) in terms) {
                s += c * ((env.pDD(t) - l * l * env.p(t)) * cos(l * t) - 2.0 * l * env.pD(t) * sin(l * t))
            }
            s
        },
    )

    /** u = p(t) cos(w0 t + beta t^2); the phase derivative is w0 + 2 beta t. */
    private fun chirp(name: String, env: Env, w0: Double, beta: Double): Sol = Sol(
        name,
        u = { t -> env.p(t) * cos(w0 * t + beta * t * t) },
        uD = { t ->
            val ph = w0 * t + beta * t * t; val dph = w0 + 2.0 * beta * t
            env.pD(t) * cos(ph) - dph * env.p(t) * sin(ph)
        },
        uDD = { t ->
            val ph = w0 * t + beta * t * t; val dph = w0 + 2.0 * beta * t
            env.pDD(t) * cos(ph) - 2.0 * dph * env.pD(t) * sin(ph) +
                env.p(t) * (-dph * dph * cos(ph) - 2.0 * beta * sin(ph))
        },
    )

    private fun solutions(w0: Double, env: Env): List<Sol> {
        val all = listOf(
            cosSum("u1", env, listOf(1.0 to 0.0)),
            cosSum("u2", env, listOf(1.0 to 0.5 * w0)),
            cosSum("u3", env, listOf(1.0 to w0)),
            cosSum("u4", env, listOf(1.0 to 1.5 * w0)),
            cosSum("u5", env, listOf(1.0 to w0, 0.3 to 2.0 * w0)),
            cosSum("u6", env, listOf(1.0 to w0, 1.0 to 0.6 * w0)),
            chirp("u7", env, w0, 0.25 * w0),
        )
        return when (val set = System.getProperty("paper.solset") ?: "all") {
            "all" -> all
            "published" -> all.take(5)
            "new" -> all.drop(5)
            else -> error("paper.solset must be all|published|new, got $set")
        }
    }

    // ==================== kernels ====================

    /** M2: K = cos(w d) e^{-d^2}, d = t - s; derivatives in t. */
    private fun kernelM2(w: Double): KernelF {
        val k = { d: Double -> cos(w * d) * exp(-d * d) }
        val kd = { d: Double -> (-w * sin(w * d) - 2.0 * d * cos(w * d)) * exp(-d * d) }
        val kdd = { d: Double ->
            (-(w * w + 2.0) * cos(w * d) + 4.0 * d * w * sin(w * d) + 4.0 * d * d * cos(w * d)) * exp(-d * d)
        }
        return KernelF(k = { t, s -> k(t - s) }, kT = { t, s -> kd(t - s) }, kTT = { t, s -> kdd(t - s) })
    }

    /** M3: K = cos(w (t - s)), K(t,t) = 1. */
    private fun kernelM3(w: Double) = KernelV(
        k = { t, s -> cos(w * (t - s)) },
        kT = { t, s -> -w * sin(w * (t - s)) },
        kS = { t, s -> w * sin(w * (t - s)) },
        kTT = { t, s -> -w * w * cos(w * (t - s)) },
    )

    /** The right-hand side is a pure function of t: the cache returns bit-for-bit the same value. */
    private fun memo(f: (Double) -> Double): (Double) -> Double {
        val cache = ConcurrentHashMap<Double, Double>(8192)
        return { t -> cache[t] ?: f(t).also { cache[t] = it } }
    }

    private val schemes = listOf("base", "sloan", "kulkarni", "iterKulkarni")

    private fun fredErrors(kernel: KernelF, s: Sol, sys: GeneratingSystem, n: Int): Map<String, Double> {
        val g = Grid.uniform(n)
        val basis = MinimalSplineBasis(sys, g)
        val op = FredholmOperator(kernel, g, GaussLegendre(QUAD))
        val f = memo { t -> s.u(t) - op.apply(t) { x -> s.u(x) } }
        val fD = memo { t -> s.uD(t) - op.applyDeriv(t) { x -> s.u(x) } }
        val fDD = memo { t -> s.uDD(t) - op.applyDeriv2(t) { x -> s.u(x) } }
        val solver = FredholmSecondKindSolver(basis, ProjFunctionals(basis), op, 1.0, RhsWithDerivatives(f, fD, fDD))
        return schemes.associateWith { sc ->
            errorEh(s.u, when (sc) {
                "base" -> solver.base().eval
                "sloan" -> solver.sloan().eval
                "kulkarni" -> solver.kulkarni().eval
                else -> solver.iteratedKulkarni().eval
            }, g, REFINE)
        }
    }

    private fun voltErrors(kernel: KernelV, s: Sol, sys: GeneratingSystem, n: Int): Map<String, Double> {
        val g = Grid.uniform(n)
        val basis = MinimalSplineBasis(sys, g)
        val op = VolterraOperator(kernel, g, GaussLegendre(QUAD))
        val f = memo { t -> s.u(t) - op.apply(t) { x -> s.u(x) } }
        val fD = memo { t -> s.uD(t) - op.applyDeriv(t) { x -> s.u(x) } }
        val fDD = memo { t -> s.uDD(t) - op.applyDeriv2(t, { x -> s.u(x) }, s.uD) }
        val solver = VolterraSecondKindSolver(basis, ProjFunctionals(basis), op, 1.0, RhsWithDerivatives(f, fD, fDD))
        return schemes.associateWith { sc ->
            errorEh(s.u, when (sc) {
                "base" -> solver.base().eval
                "sloan" -> solver.sloan().eval
                "kulkarni" -> solver.kulkarni().eval
                else -> solver.iteratedKulkarni().eval
            }, g, REFINE)
        }
    }

    // ==================== output ====================

    private val log = StringBuilder()
    private val results = LinkedHashMap<String, Double>()
    private val t0 = System.currentTimeMillis()

    private fun fmt(v: Double): String = if (v.isNaN()) "NaN" else String.format("%.6e", v)

    private fun emit(key: String, value: Double) {
        results[key] = value
        log.append(key).append('\t').append(fmt(value)).append('\n')
    }

    private fun header(tag: String) {
        log.append("# PaperTablesTool — ").append(tag).append('\n')
        log.append("# backend: ").append(System.getProperty("numerics.backend", "default")).append('\n')
        log.append("# quad nodes/cell: ").append(QUAD).append("; control grid: ").append(REFINE).append("n+1\n")
        log.append("# started: ").append(java.time.Instant.now()).append('\n')
    }

    private fun dump(tag: String) {
        log.append("# elapsed: ").append((System.currentTimeMillis() - t0) / 1000.0).append(" s\n")
        print(log)
        val dir = File(System.getProperty("user.dir")).resolve("build/sec4")
        dir.mkdirs()
        File(dir, "$tag.txt").writeText(log.toString())
        File(dir, "$tag.json").writeText(buildString {
            append("{\n")
            append(results.entries.joinToString(",\n") { (k, v) ->
                " \"" + k + "\": " + (if (v.isNaN() || v.isInfinite()) "null" else v.toString())
            })
            append("\n}\n")
        })
        println("written: " + File(dir, "$tag.json").absolutePath)
    }

    // ==================== tab:sol-freq ====================

    @Test
    fun solFreqTable() {
        header("tab:sol-freq, solutions u1..u7 (paper.solset)")
        val ns = (System.getProperty("paper.solfreq.n") ?: "64").split(",").map { it.trim().toInt() }
        // M2 (Fredholm II, w0 = 40) and M3 (Volterra II, w0 = 30): the generating system is chosen by the kernel.
        for ((prob, w0) in listOf("M2" to 40.0, "M3" to 30.0)) {
            val sysB = GeneratingSystem.B
            val sysM = sysT(w0)
            for (s in solutions(w0, envSol)) {
                for (n in ns) {
                    val eB: Map<String, Double>
                    val eT: Map<String, Double>
                    if (prob == "M2") {
                        val k = kernelM2(w0)
                        eB = fredErrors(k, s, sysB, n); eT = fredErrors(k, s, sysM, n)
                    } else {
                        val k = kernelM3(w0)
                        eB = voltErrors(k, s, sysB, n); eT = voltErrors(k, s, sysM, n)
                    }
                    for (sc in schemes) {
                        emit("tab:sol-freq|$prob|${s.name}|$n|$sc|ehB", eB.getValue(sc))
                        emit("tab:sol-freq|$prob|${s.name}|$n|$sc|ehT", eT.getValue(sc))
                        emit("tab:sol-freq|$prob|${s.name}|$n|$sc|G", eB.getValue(sc) / eT.getValue(sc))
                    }
                }
            }
        }
        dump("solfreq")
    }

    // ==================== rem:m4-split ====================

    /**
     * The regularization component of the M4 error, ||u* - u_alpha||_inf, by an independent Nystrom
     * method. From (alpha I + K) v = u* it follows that u* - u_alpha = alpha v and also
     * u* - u_alpha = u* - (K v); the latter (the natural Nystrom extension) is defined at every t,
     * including the endpoints, where the maximum is attained. The supremum is taken over 12801
     * equally spaced points of [0, 1]; for comparison the maximum over the quadrature nodes alone
     * (all strictly inside (0, 1)) is printed as well.
     */
    @Test
    fun m4RegularizationSplit() {
        header("rem:m4-split")
        val w = 20.0
        val u = { t: Double -> (9.0 - 5.0 * t) * exp(t * t) * cos(w * t) }  // the exact solution of M4
        val kern = { t: Double, s: Double -> val d = t - s; cos(w * d) * exp(-d * d) }
        val nsubs = (System.getProperty("paper.m4split.nsub") ?: "64,128,256").split(",").map { it.trim().toInt() }
        val (rn, rw) = GaussLegendre(8).refNodesWeights()
        for (nSub in nsubs) {
            val nodes = DoubleArray(nSub * rn.size)
            val weights = DoubleArray(nSub * rn.size)
            for (m in 0 until nSub) {
                val lo = m.toDouble() / nSub
                val hi = (m + 1).toDouble() / nSub
                val half = 0.5 * (hi - lo)
                val mid = 0.5 * (hi + lo)
                for (i in rn.indices) {
                    nodes[m * rn.size + i] = mid + half * rn[i]
                    weights[m * rn.size + i] = half * rw[i]
                }
            }
            val p = nodes.size
            for (alpha in listOf(1e-6, 1e-8, 1e-10, 1e-12)) {
                val a = Array(p) { i -> DoubleArray(p) { j -> weights[j] * kern(nodes[i], nodes[j]) } }
                for (i in 0 until p) a[i][i] += alpha
                val v = LinearAlgebra.solve(a, DoubleArray(p) { u(nodes[it]) })
                var onNodes = 0.0
                for (x in v) onNodes = maxOf(onNodes, abs(alpha * x))
                var sup = 0.0
                val m = 12800
                for (k in 0..m) {
                    val t = k.toDouble() / m
                    var acc = 0.0
                    for (j in 0 until p) acc += weights[j] * kern(t, nodes[j]) * v[j]
                    sup = maxOf(sup, abs(u(t) - acc))
                }
                val tag = String.format("%.0e", alpha)
                emit("rem:m4-split|$tag|$nSub|sup", sup)
                emit("rem:m4-split|$tag|$nSub|nodes", onNodes)
            }
        }
        dump("m4split")
    }
}
