package verification

import java.io.File
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.cosh
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sinh
import kotlin.test.Test
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.DeBoorFixFunctionals
import splines.functionals.DiscreteDeBoorFixFunctionals
import splines.functionals.FunctionalFamily
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import numerics.orders
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmFirstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF
import solvers.volterra.KernelV
import solvers.volterra.VolterraFirstKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * AN INDEPENDENT VERIFICATION DRIVER OF THE §4 TABLES (role 12).
 *
 * It is not part of the library and changes nothing in it: it uses ONLY
 * the public API of the `main` branch. The model problems M1--M5 and the frequency-tuned
 * generating systems are defined here anew (they are absent on `main`), so that the run
 * is independent of the code the tables were produced with.
 *
 * The result: a TSV in stdout + a file given by the system property `sec4.out`.
 */
class Sec4VerificationTool {

    // ==================== generating systems ====================

    private fun sysT(w: Double) = GeneratingSystem(
        name = "T$w",
        rho = { t -> sin(w * t) }, sigma = { t -> cos(w * t) },
        rhoD = { t -> w * cos(w * t) }, sigmaD = { t -> -w * sin(w * t) },
        rhoDD = { t -> -w * w * sin(w * t) }, sigmaDD = { t -> -w * w * cos(w * t) },
    )

    private fun sysH(w: Double) = GeneratingSystem(
        name = "H$w",
        rho = { t -> sinh(w * t) }, sigma = { t -> cosh(w * t) },
        rhoD = { t -> w * cosh(w * t) }, sigmaD = { t -> w * sinh(w * t) },
        rhoDD = { t -> w * w * sinh(w * t) }, sigmaDD = { t -> w * w * cosh(w * t) },
    )

    private fun system(tag: String): GeneratingSystem = when {
        tag == "B" -> GeneratingSystem.B
        tag == "H" -> GeneratingSystem.H
        tag == "T" -> GeneratingSystem.T
        tag.startsWith("H") -> sysH(tag.removePrefix("H").toDouble())
        tag.startsWith("T") -> sysT(tag.removePrefix("T").toDouble())
        else -> error("unknown system $tag")
    }

    // ==================== model problems ====================

    /** u*(t) = cos(w t)/(1+t) and two derivatives; p = 1/(1+t), p' = -p^2, p'' = 2p^3. */
    private fun oscSolution(w: Double): Triple<(Double) -> Double, (Double) -> Double, (Double) -> Double> {
        val u = { t: Double -> cos(w * t) / (1.0 + t) }
        val uD = { t: Double ->
            val p = 1.0 / (1.0 + t)
            -w * sin(w * t) * p - cos(w * t) * p * p
        }
        val uDD = { t: Double ->
            val p = 1.0 / (1.0 + t)
            -w * w * cos(w * t) * p + 2.0 * w * sin(w * t) * p * p + 2.0 * cos(w * t) * p * p * p
        }
        return Triple(u, uD, uDD)
    }

    /** The M2/M4 kernel: cos(w d) e^{-d^2}, d = t-s; derivatives in t. */
    private fun gaussOsc(w: Double): Triple<(Double) -> Double, (Double) -> Double, (Double) -> Double> {
        val k = { d: Double -> cos(w * d) * exp(-d * d) }
        val kd = { d: Double -> (-w * sin(w * d) - 2.0 * d * cos(w * d)) * exp(-d * d) }
        val kdd = { d: Double ->
            (-(w * w + 2.0) * cos(w * d) + 4.0 * d * w * sin(w * d) + 4.0 * d * d * cos(w * d)) * exp(-d * d)
        }
        return Triple(k, kd, kdd)
    }

    private class FredProblem(
        val name: String,
        val kernel: KernelF,
        val exact: (Double) -> Double,
        val exactD: (Double) -> Double,
        val exactDD: (Double) -> Double,
        val secondKind: Boolean,
    )

    private class VoltProblem(
        val name: String,
        val kernel: KernelV,
        val exact: (Double) -> Double,
        val exactD: (Double) -> Double,
        val exactDD: (Double) -> Double,
        val secondKind: Boolean,
    )

    /** M1: Fredholm II, K = 1/(1+t+s), u* = 1/(1+t). */
    private val m1 = FredProblem(
        "M1",
        KernelF(
            k = { t, s -> 1.0 / (1.0 + t + s) },
            kT = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
            kTT = { t, s -> 2.0 / Math.pow(1.0 + t + s, 3.0) },
        ),
        exact = { t -> 1.0 / (1.0 + t) },
        exactD = { t -> -1.0 / ((1.0 + t) * (1.0 + t)) },
        exactDD = { t -> 2.0 / Math.pow(1.0 + t, 3.0) },
        secondKind = true,
    )

    /** M2: Fredholm II, K = cos(40 d) e^{-d^2}, u* = cos(40t)/(1+t). */
    private val m2 = run {
        val (k, kd, kdd) = gaussOsc(40.0)
        val (u, uD, uDD) = oscSolution(40.0)
        FredProblem(
            "M2",
            KernelF(k = { t, s -> k(t - s) }, kT = { t, s -> kd(t - s) }, kTT = { t, s -> kdd(t - s) }),
            u, uD, uDD, secondKind = true,
        )
    }

    /** M4: Fredholm I, the kernel at w=20, u* = cos(20t)/(1+t). */
    private val m4 = run {
        val (k, kd, kdd) = gaussOsc(20.0)
        val (u, uD, uDD) = oscSolution(20.0)
        FredProblem(
            "M4",
            KernelF(k = { t, s -> k(t - s) }, kT = { t, s -> kd(t - s) }, kTT = { t, s -> kdd(t - s) }),
            u, uD, uDD, secondKind = false,
        )
    }

    /** M3: Volterra II, K = cos(30 d), u* = cos(30t)/(1+t); K(t,t)=1. */
    private val m3 = run {
        val w = 30.0
        val (u, uD, uDD) = oscSolution(w)
        VoltProblem(
            "M3",
            KernelV(
                k = { t, s -> cos(w * (t - s)) },
                kT = { t, s -> -w * sin(w * (t - s)) },
                kS = { t, s -> w * sin(w * (t - s)) },
                kTT = { t, s -> -w * w * cos(w * (t - s)) },
            ),
            u, uD, uDD, secondKind = true,
        )
    }

    /** M5: Volterra I, K = 1 + d - sin(30 d)/30, u* = cos(30t)/(1+t); K(t,t)=1. */
    private val m5 = run {
        val w = 30.0
        val (u, uD, uDD) = oscSolution(w)
        VoltProblem(
            "M5",
            KernelV(
                k = { t, s -> 1.0 + (t - s) - sin(w * (t - s)) / w },
                kT = { t, s -> 1.0 - cos(w * (t - s)) },
                kS = { t, s -> cos(w * (t - s)) - 1.0 },
                kTT = { t, s -> w * sin(w * (t - s)) },
            ),
            u, uD, uDD, secondKind = false,
        )
    }

    // ==================== functional families ====================

    private fun family(tag: String, basis: MinimalSplineBasis): FunctionalFamily = when (tag) {
        "theta" -> ProjFunctionals(basis)
        "xi0" -> DeBoorFixFunctionals(basis, 0)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "xi2" -> DeBoorFixFunctionals(basis, 2)
        "xit1" -> DiscreteDeBoorFixFunctionals(basis, 1)
        "xit2" -> DiscreteDeBoorFixFunctionals(basis, 2)
        else -> error("unknown family $tag")
    }

    private fun grid(kind: String, n: Int): Grid =
        if (kind == "uniform") Grid.uniform(n) else Grid.graded(n, ratio = 2.0)

    /**
     * MEMOIZATION OF THE RIGHT-HAND SIDE. `f` is built by the method of manufactured solutions, that is,
     * EVERY call of it is a quadrature over ng nodes. When assembling the vector `d_j = chi_j(L f)`
     * it is called FROM a quadrature, which gives O(ng^2) kernel evaluations per node (for
     * n=128 that is ~2e8 and practically unfinishable).
     *
     * The cache does not change the numbers: `f` is a pure function of `t`, the set of requested points
     * is finite (the Gauss nodes and the support points of the functionals) and repeats for all j.
     * Exactly the same value is returned as by a direct call; this is checked by a health check
     * `memo consistency`.
     */
    private fun memo(f: (Double) -> Double): (Double) -> Double {
        val cache = HashMap<Double, Double>(4096)
        return { t -> cache.getOrPut(t) { f(t) } }
    }

    // ==================== collecting the results ====================

    private val log = StringBuilder()
    private val results = LinkedHashMap<String, Double>()

    private fun emit(key: String, value: Double) {
        results[key] = value
        log.append(key).append('\t').append(fmt(value)).append('\n')
    }

    private fun fmt(v: Double): String = if (v.isNaN()) "NaN" else String.format("%.6e", v)

    private fun runFred(
        p: FredProblem, gridKind: String, n: Int, sysTag: String, famTag: String,
        schemes: List<String>, alpha: Double = 1e-10, quadNodes: Int = 8,
    ): Map<String, Double> {
        val g = grid(gridKind, n)
        val basis = MinimalSplineBasis(system(sysTag), g)
        val funcs = family(famTag, basis)
        val op = FredholmOperator(p.kernel, g, GaussLegendre(quadNodes))
        val rhs = memo { t: Double ->
            val i = op.apply(t) { s -> p.exact(s) }
            if (p.secondKind) p.exact(t) - i else i
        }
        val rhsD = memo { t: Double ->
            val i = op.applyDeriv(t) { s -> p.exact(s) }
            if (p.secondKind) p.exactD(t) - i else i
        }
        val rhsDD = memo { t: Double ->
            val i = op.applyDeriv2(t) { s -> p.exact(s) }
            if (p.secondKind) p.exactDD(t) - i else i
        }
        val out = LinkedHashMap<String, Double>()
        if (p.secondKind) {
            val solver = FredholmSecondKindSolver(
                basis, funcs, op, 1.0, RhsWithDerivatives(rhs, rhsD, rhsDD),
            )
            for (s in schemes) {
                out[s] = errorEh(p.exact, when (s) {
                    "base" -> solver.base().eval
                    "sloan" -> solver.sloan().eval
                    "kulkarni" -> solver.kulkarni().eval
                    "iterKulkarni" -> solver.iteratedKulkarni().eval
                    "nystrom" -> solver.nystrom().eval
                    "iterNystrom" -> solver.iteratedNystrom().eval
                    "combinedNystrom" -> solver.combinedNystrom().eval
                    "iterCombinedNystrom" -> solver.iteratedCombinedNystrom().eval
                    else -> error("scheme $s")
                }, g)
            }
        } else {
            val solver = FredholmFirstKindSolver(basis, funcs, op, rhs, rhsD, rhsDD, alpha = alpha)
            for (s in schemes) {
                out[s] = errorEh(p.exact, when (s) {
                    "base" -> solver.base().eval
                    "sloan" -> solver.sloan().eval
                    "kulkarni" -> solver.kulkarni().eval
                    "iterKulkarni" -> solver.iteratedKulkarni().eval
                    else -> error("scheme $s for F1")
                }, g)
            }
        }
        return out
    }

    private fun runVolt(
        p: VoltProblem, gridKind: String, n: Int, sysTag: String, famTag: String,
        schemes: List<String>, quadNodes: Int = 8,
    ): Map<String, Double> {
        val g = grid(gridKind, n)
        val basis = MinimalSplineBasis(system(sysTag), g)
        val funcs = family(famTag, basis)
        val op = VolterraOperator(p.kernel, g, GaussLegendre(quadNodes))
        val out = LinkedHashMap<String, Double>()
        if (p.secondKind) {
            val rhs = memo { t: Double -> p.exact(t) - op.apply(t) { s -> p.exact(s) } }
            val rhsD = memo { t: Double -> p.exactD(t) - op.applyDeriv(t) { s -> p.exact(s) } }
            val rhsDD = memo { t: Double ->
                p.exactDD(t) - op.applyDeriv2(t, { s -> p.exact(s) }, p.exactD)
            }
            val solver = VolterraSecondKindSolver(
                basis, funcs, op, 1.0, RhsWithDerivatives(rhs, rhsD, rhsDD),
            )
            for (s in schemes) {
                out[s] = errorEh(p.exact, when (s) {
                    "base" -> solver.base().eval
                    "sloan" -> solver.sloan().eval
                    "kulkarni" -> solver.kulkarni().eval
                    "iterKulkarni" -> solver.iteratedKulkarni().eval
                    "nystrom" -> solver.nystrom().eval
                    "iterNystrom" -> solver.iteratedNystrom().eval
                    "combinedNystrom" -> solver.combinedNystrom().eval
                    "iterCombinedNystrom" -> solver.iteratedCombinedNystrom().eval
                    else -> error("scheme $s")
                }, g)
            }
        } else {
            val rhsD = memo { t: Double -> op.applyDeriv(t) { s -> p.exact(s) } }
            val solver = VolterraFirstKindSolver(basis, funcs, p.kernel, rhsD, p.exact, p.exactD)
            for (s in schemes) {
                out[s] = errorEh(p.exact, when (s) {
                    "base" -> solver.base().eval
                    "sloan" -> solver.sloan().eval
                    "kulkarni" -> solver.kulkarni().eval
                    "iterKulkarni" -> solver.iteratedKulkarni().eval
                    else -> error("scheme $s for V1")
                }, g)
            }
        }
        return out
    }

    // ==================== HEALTH-CHECKS ====================

    private val hc = StringBuilder()
    private var hcFailures = 0

    private fun hcLine(name: String, value: Double, tol: Double, ok: Boolean) {
        if (!ok) hcFailures++
        hc.append(String.format("%-58s %-14s tol=%-12s %s%n", name, fmt(value), fmt(tol),
            if (ok) "PASS" else "FAIL"))
    }

    private fun quadratureSmoke() {
        val q = GaussLegendre(8)
        val bp = Grid.uniform(4).breakpoints
        val cases = listOf<Triple<String, (Double) -> Double, Double>>(
            Triple("quad f=1 -> 1", { _ -> 1.0 }, 1.0),
            Triple("quad f=x -> 1/2", { x -> x }, 0.5),
            Triple("quad f=x^2 -> 1/3", { x -> x * x }, 1.0 / 3.0),
            Triple("quad f=x^3 -> 1/4", { x -> x * x * x }, 0.25),
        )
        for ((name, f, exactVal) in cases) {
            val err = abs(q.integrate(bp, f) - exactVal)
            hcLine(name, err, 1e-12, err <= 1e-12)
        }
    }

    /** max_{i,j} |chi_j(omega_i) - delta_ij|. */
    private fun biorthDefect(basis: MinimalSplineBasis, funcs: FunctionalFamily): Double {
        val n = basis.n
        var worst = 0.0
        for (j in -2..n - 1) {
            val chi = funcs.chi(j)
            for (i in -2..n - 1) {
                val v = chi.apply(
                    { t -> basis.omega(i, t) },
                    { t -> basis.omegaDeriv(i, t) },
                    { t -> basis.omegaDeriv2(i, t) },
                )
                worst = maxOf(worst, abs(v - if (i == j) 1.0 else 0.0))
            }
        }
        return worst
    }

    /** ||P(P g) - P g||_inf: a check of P^2 = P. */
    private fun projectorDefect(basis: MinimalSplineBasis, funcs: FunctionalFamily): Double {
        val n = basis.n
        val g = { t: Double -> exp(t) * cos(3.0 * t) }
        val gD = { t: Double -> exp(t) * (cos(3.0 * t) - 3.0 * sin(3.0 * t)) }
        val gDD = { t: Double -> exp(t) * (-8.0 * cos(3.0 * t) - 6.0 * sin(3.0 * t)) }
        val c = DoubleArray(n + 2) { funcs.chi(it - 2).apply(g, gD, gDD) }
        val pg = { t: Double -> (0 until n + 2).sumOf { c[it] * basis.omega(it - 2, t) } }
        val pgD = { t: Double -> (0 until n + 2).sumOf { c[it] * basis.omegaDeriv(it - 2, t) } }
        val pgDD = { t: Double -> (0 until n + 2).sumOf { c[it] * basis.omegaDeriv2(it - 2, t) } }
        val c2 = DoubleArray(n + 2) { funcs.chi(it - 2).apply(pg, pgD, pgDD) }
        val p2g = { t: Double -> (0 until n + 2).sumOf { c2[it] * basis.omega(it - 2, t) } }
        val gr = basis.grid
        var worst = 0.0
        val m = 40 * gr.n
        for (k in 0..m) {
            val t = gr.a + (gr.b - gr.a) * k / m
            worst = maxOf(worst, abs(p2g(t) - pg(t)))
        }
        return worst
    }

    private fun healthChecks() {
        hc.append("=== HEALTH CHECKS (before the measurements) ===\n")
        quadratureSmoke()

        for (tag in listOf("B", "H", "T", "T20.0", "T30.0", "T40.0")) {
            val s = system(tag)
            var minAbs = Double.MAX_VALUE
            for (k in 0..100) minAbs = minOf(minAbs, abs(s.wronskian(k / 100.0)))
            hcLine("wronskian min|W| $tag", minAbs, 1e-6, minAbs > 1e-6)
        }

        for (sysTag in listOf("B", "H", "T30.0")) {
            val g = Grid.uniform(32)
            val basis = MinimalSplineBasis(system(sysTag), g)
            for (fam in listOf("theta", "xi0", "xi1", "xi2")) {
                val d = biorthDefect(basis, family(fam, basis))
                hcLine("biorth $fam($sysTag) n=32", d, 1e-9, d <= 1e-9)
            }
            for (fam in listOf("theta", "xi1")) {
                val d = projectorDefect(basis, family(fam, basis))
                hcLine("projector P^2=P $fam($sysTag) n=32", d, 1e-8, d <= 1e-8)
            }
        }

        // xitilde: the biorthogonality MUST be violated (rem:no-biorth).
        for (n in listOf(16, 32, 64, 128)) {
            val basis = MinimalSplineBasis(sysT(30.0), Grid.uniform(n))
            val d = biorthDefect(basis, DiscreteDeBoorFixFunctionals(basis, 1))
            emit("biorth.xit1.T30.n$n", d)
            hcLine("biorth xitilde1(T30) n=$n (O(1) expected)", d, 0.1, d > 0.1)
        }

        // Exact representability: cos(30t) in span{1, sin 30t, cos 30t}.
        run {
            val w = 30.0
            val g = Grid.uniform(32)
            val basis = MinimalSplineBasis(sysT(w), g)
            val funcs = ProjFunctionals(basis)
            val u = { t: Double -> cos(w * t) }
            val uD = { t: Double -> -w * sin(w * t) }
            val uDD = { t: Double -> -w * w * cos(w * t) }
            val c = DoubleArray(g.n + 2) { funcs.chi(it - 2).apply(u, uD, uDD) }
            var worst = 0.0
            val m = 100 * g.n
            for (k in 0..m) {
                val t = g.a + (g.b - g.a) * k / m
                worst = maxOf(worst, abs((0 until g.n + 2).sumOf { c[it] * basis.omega(it - 2, t) } - u(t)))
            }
            hcLine("reproduction cos(30t) by theta(T30) n=32", worst, 1e-10, worst <= 1e-10)
        }

        // The convergence of the right-hand side in the number of quadrature nodes.
        run {
            val g = Grid.uniform(64)
            val vals = listOf(8, 16, 32).map { q ->
                FredholmOperator(m2.kernel, g, GaussLegendre(q)).apply(0.37) { s -> m2.exact(s) }
            }
            val d1 = abs(vals[1] - vals[0])
            val d2 = abs(vals[2] - vals[1])
            emit("rhs.M2.quad8vs16", d1)
            emit("rhs.M2.quad16vs32", d2)
            hcLine("rhs M2 quad 8 vs 16 (n=64)", d1, 1e-12, d1 <= 1e-12)
            hcLine("rhs M2 quad 16 vs 32 (n=64)", d2, 1e-12, d2 <= 1e-12)
        }

        // The memoization of the right-hand side must not change the numbers (the cache is only a speed-up).
        run {
            val g = Grid.uniform(16)
            val op = FredholmOperator(m2.kernel, g, GaussLegendre(8))
            val direct = { t: Double -> m2.exact(t) - op.apply(t) { s -> m2.exact(s) } }
            val cached = memo(direct)
            var worst = 0.0
            for (k in 0..500) {
                val t = k / 500.0
                worst = maxOf(worst, abs(cached(t) - direct(t)))
                worst = maxOf(worst, abs(cached(t) - direct(t))) // the second call comes from the cache
            }
            hcLine("memo consistency (rhs M2, 501 points)", worst, 0.0, worst == 0.0)
        }

        hc.append(if (hcFailures == 0) "RESULT: all health checks PASS\n" else "RESULT: FAIL=$hcFailures\n")
    }

    // ==================== tables ====================

    private fun fredTable(
        tab: String, p: FredProblem, gridKind: String, ns: List<Int>,
        groups: List<Pair<String, String>>, schemes: List<String>,
    ) {
        for ((sysTag, famTag) in groups) {
            val perScheme = LinkedHashMap<String, MutableList<Double>>()
            for (n in ns) {
                val r = runFred(p, gridKind, n, sysTag, famTag, schemes, quadNodes = QUAD)
                for ((s, v) in r) perScheme.getOrPut(s) { mutableListOf() }.add(v)
            }
            for ((s, errs) in perScheme) {
                val ords = orders(errs)
                for ((k, n) in ns.withIndex()) {
                    emit("$tab|$sysTag|$famTag|$n|$s|eh", errs[k])
                    emit("$tab|$sysTag|$famTag|$n|$s|ph", ords[k])
                }
            }
        }
    }

    private fun voltTable(
        tab: String, p: VoltProblem, gridKind: String, ns: List<Int>,
        groups: List<Pair<String, String>>, schemes: List<String>,
    ) {
        for ((sysTag, famTag) in groups) {
            val perScheme = LinkedHashMap<String, MutableList<Double>>()
            for (n in ns) {
                val r = runVolt(p, gridKind, n, sysTag, famTag, schemes, quadNodes = QUAD)
                for ((s, v) in r) perScheme.getOrPut(s) { mutableListOf() }.add(v)
            }
            for ((s, errs) in perScheme) {
                val ords = orders(errs)
                for ((k, n) in ns.withIndex()) {
                    emit("$tab|$sysTag|$famTag|$n|$s|eh", errs[k])
                    emit("$tab|$sysTag|$famTag|$n|$s|ph", ords[k])
                }
            }
        }
    }

    private fun freqTable() {
        val ratios = listOf(0.25, 0.50, 0.75, 0.90, 1.00, 1.05, 1.10, 1.25, 1.40, 1.50, 2.00)
        val schemes = listOf("base", "kulkarni")
        val n = 64
        val refF = runFred(m2, "uniform", n, "B", "theta", schemes)
        val refV = runVolt(m3, "uniform", n, "B", "theta", schemes)
        for (r in ratios) {
            val cf = runFred(m2, "uniform", n, "T${40.0 * r}", "theta", schemes)
            val cv = runVolt(m3, "uniform", n, "T${30.0 * r}", "theta", schemes)
            for (s in schemes) {
                emit("tab:freq|M2|" + String.format("%.2f", r) + "|$s", refF.getValue(s) / cf.getValue(s))
                emit("tab:freq|M3|" + String.format("%.2f", r) + "|$s", refV.getValue(s) / cv.getValue(s))
            }
        }
    }

    /**
     * The dump of the results. The path is FIXED and not set by a system property: Gradle
     * does not forward `-D` from the command line into the forked test JVM (verified:
     * `build.gradle.kts` passes only `scipy.*` and `numerics.backend`), while editing
     * `build.gradle.kts` would be a change of the repository outside the scope of role 12.
     * The scope of the run is chosen BY METHOD through `--tests ...Sec4VerificationTool.<method>`.
     */
    private fun dump(tag: String) {
        log.append("# elapsed: ").append((System.currentTimeMillis() - t0) / 1000.0).append(" s\n")
        print(hc)
        print(log)
        val dir = File(System.getProperty("user.dir")).resolve("build/sec4")
        dir.mkdirs()
        File(dir, "$tag.txt").writeText(hc.toString() + log.toString())
        File(dir, "$tag.json").writeText(buildString {
            append("{\n")
            append(results.entries.joinToString(",\n") { (k, v) ->
                val jv = if (v.isNaN() || v.isInfinite()) "null" else v.toString()
                " \"" + k + "\": " + jv
            })
            append("\n}\n")
        })
        println("written: " + File(dir, "$tag.txt").absolutePath)
        println("written: " + File(dir, "$tag.json").absolutePath)
        println("health-check FAIL: $hcFailures")
    }

    private val t0 = System.currentTimeMillis()

    private fun header(tag: String) {
        log.append("# Sec4VerificationTool — an independent verification of the §4 tables\n")
        log.append("# scope: ").append(tag).append('\n')
        log.append("# backend: ").append(System.getProperty("numerics.backend", "default")).append('\n')
        log.append("# quad nodes/cell: ").append(QUAD).append('\n')
        log.append("# started: ").append(java.time.Instant.now()).append('\n')
    }

    /** The number of nodes of the composite Gauss--Legendre quadrature per cell (-Dsec4.quad=16). */
    private val QUAD: Int = (System.getProperty("sec4.quad") ?: "8").toInt()

    private val four = listOf("base", "sloan", "kulkarni", "iterKulkarni")
    private val ny = listOf("nystrom", "iterNystrom")
    private val nyc = listOf("combinedNystrom", "iterCombinedNystrom")

    @Test
    fun sec4HealthChecks() {
        header("hc")
        healthChecks()
        dump("hc")
    }

    @Test
    fun sec4TablesM1() {
        header("m1")
        fredTable("tab:h1h2", m1, "uniform", listOf(8, 16, 32, 64),
            listOf("B" to "theta", "B" to "xi1", "H" to "xi1"), four)
        fredTable("tab:h1h2-extra", m1, "uniform", listOf(8, 16, 32, 64),
            listOf("B" to "xi0", "H" to "xi0", "B" to "xi2", "H" to "xi2"), four)
        fredTable("tab:nonuni-m1", m1, "graded", listOf(8, 16, 32, 64),
            listOf("B" to "theta", "B" to "xi1", "H" to "xi1"), four)
        fredTable("tab:h4-m1", m1, "uniform", listOf(8, 16, 32, 64),
            listOf("B" to "theta", "B" to "xit1", "H" to "xit1"), ny)
        fredTable("tab:h4c-m1", m1, "uniform", listOf(4, 8, 16, 32),
            listOf("B" to "theta", "B" to "xit1", "H" to "xit1"), nyc)
        dump("m1")
    }

    @Test
    fun sec4TablesM2() {
        header("m2")
        fredTable("tab:h3-m2", m2, "uniform", listOf(32, 64, 128),
            listOf("B" to "theta", "T40.0" to "xi1", "T40.0" to "theta"), four)
        fredTable("tab:nonuni-m2", m2, "graded", listOf(32, 64, 128),
            listOf("B" to "theta", "T40.0" to "xi1", "T40.0" to "theta"), four)
        dump("m2")
    }

    @Test
    fun sec4TablesM3() {
        header("m3")
        voltTable("tab:h3-m3", m3, "uniform", listOf(32, 64, 128),
            listOf("B" to "theta", "T30.0" to "xi1", "T30.0" to "theta"), four)
        voltTable("tab:nonuni-m3", m3, "graded", listOf(32, 64, 128),
            listOf("B" to "theta", "T30.0" to "xi1", "T30.0" to "theta"), four)
        voltTable("tab:h4-m3", m3, "uniform", listOf(16, 32, 64, 128),
            listOf("B" to "theta", "B" to "xit1", "T30.0" to "xit1"), ny)
        voltTable("tab:h4c-m3", m3, "uniform", listOf(16, 32, 64, 128),
            listOf("B" to "theta", "B" to "xit1", "T30.0" to "xit1"), nyc)
        voltTable("tab:nystrom-theta-c", m3, "uniform", listOf(16, 32, 64),
            listOf("B" to "theta", "T30.0" to "theta", "H" to "theta"), nyc)
        dump("m3")
    }

    @Test
    fun sec4TablesFirstKind() {
        header("m45")
        fredTable("tab:m4", m4, "uniform", listOf(16, 32, 64, 128),
            listOf("B" to "xi1", "T20.0" to "xi1"), listOf("base", "sloan"))
        voltTable("tab:m5", m5, "uniform", listOf(32, 64, 128),
            listOf("B" to "xi1", "T30.0" to "xi1"), four)
        dump("m45")
    }

    /**
     * The error of the numerical differentiation of the right-hand side in the reduction of M5 (Volterra I).
     *
     * The library computes g'(t) = s'(t) + deriv4(t, g - s), where s is the smooth part
     * (in our runs the exact solution), and deriv4 is a five-point difference of the 4th order
     * with the step 1e-3. Here the same quantity is compared with the analytic g' = f''.
     */
    @Test
    fun sec4M5DerivError() {
        header("m5deriv")
        val w = 30.0
        val fdStep = 1e-3
        val g = Grid.uniform(128)
        val op = VolterraOperator(m5.kernel, g, GaussLegendre(8))
        // g_eff(t) = f'(t) / K(t,t), K(t,t) = 1 for M5
        val gEff = memo { t: Double -> op.applyDeriv(t) { s -> m5.exact(s) } / m5.kernel.k(t, t) }
        val residual = { t: Double -> gEff(t) - m5.exact(t) }
        // deriv4 is a reproduction of the library formulas (central + one-sided near the ends)
        val tolB = 1e-9 * (g.b - g.a)   // BOUNDARY_RELATIVE_TOLERANCE of the library
        fun deriv4(t: Double, f: (Double) -> Double): Double = when {
            t - 2 * fdStep < g.a - tolB ->
                (-25 * f(t) + 48 * f(t + fdStep) - 36 * f(t + 2 * fdStep) +
                    16 * f(t + 3 * fdStep) - 3 * f(t + 4 * fdStep)) / (12 * fdStep)
            t + 2 * fdStep > g.b + tolB ->
                (25 * f(t) - 48 * f(t - fdStep) + 36 * f(t - 2 * fdStep) -
                    16 * f(t - 3 * fdStep) + 3 * f(t - 4 * fdStep)) / (12 * fdStep)
            else ->
                (-f(t + 2 * fdStep) + 8 * f(t + fdStep) - 8 * f(t - fdStep) + f(t - 2 * fdStep)) / (12 * fdStep)
        }
        fun err(t: Double): Double {
            val fd = m5.exactD(t) + deriv4(t, residual)
            val exactGD = op.applyDeriv2(t, { s -> m5.exact(s) }, m5.exactD) / m5.kernel.k(t, t)
            return abs(fd - exactGD)
        }
        // (a) over the whole control grid
        var wAll = 0.0; var atAll = 0.0
        val m = 1281
        for (k in 0..m) {
            val t = g.a + (g.b - g.a) * k / m
            val e = err(t); if (e > wAll) { wAll = e; atAll = t }
        }
        // (b) outside the five-point boundary layer (|t-a|,|t-b| > 4h_fd)
        var wIn = 0.0
        for (k in 0..m) {
            val t = g.a + (g.b - g.a) * k / m
            if (t - g.a <= 4 * fdStep || g.b - t <= 4 * fdStep) continue
            wIn = maxOf(wIn, err(t))
        }
        // (c) ONLY at the grid nodes — it is there that the xi functionals take the derivative
        var wNodes = 0.0; var atNodes = 0.0
        for (j in -2..g.n + 1) {
            val t = g.a + j * (g.b - g.a) / g.n
            if (t < g.a || t > g.b) continue
            val e = err(t); if (e > wNodes) { wNodes = e; atNodes = t }
        }
        emit("m5.derivError.max", wAll)
        emit("m5.derivError.at", atAll)
        emit("m5.derivError.interior", wIn)
        emit("m5.derivError.nodes", wNodes)
        emit("m5.derivError.nodesAt", atNodes)
        println("eps_D over the whole grid    = " + fmt(wAll) + "  at t = " + fmt(atAll))
        println("eps_D outside the layer     = " + fmt(wIn))
        println("eps_D at the grid nodes     = " + fmt(wNodes) + "  at t = " + fmt(atNodes))
        dump("m5deriv")
    }

    /**
     * A CONTROL: are the numbers of tab:m5 determined by the error of the numerical differentiation?
     *
     * The reduced equation M5 is written out analytically:
     *   K(t,t)=1, K_t(t,s)=1-cos(w(t-s))  =>  u - W u = g,
     *   W: the kernel -(1-cos(w(t-s))), d/dt[-(1-cos)] = -w sin(w(t-s)),
     *   g = f' = u + \int (1-cos) u,  g' = f''.
     * Here g and g' are given ANALYTICALLY (through applyDeriv/applyDeriv2 of the original
     * operator), that is, the five-point difference does not take part at all. The comparison with
     * tab:m5 shows the contribution of the differentiation error.
     */
    @Test
    fun sec4M5AnalyticVsFD() {
        header("m5cmp")
        val w = 30.0
        val reduced = KernelV(
            k = { t, s -> -(1.0 - cos(w * (t - s))) },
            kT = { t, s -> -w * sin(w * (t - s)) },
        )
        for (n in listOf(32, 64, 128)) {
            for (sysTag in listOf("B", "T30.0")) {
                val g = Grid.uniform(n)
                val basis = MinimalSplineBasis(system(sysTag), g)
                val funcs = family("xi1", basis)
                val opOrig = VolterraOperator(m5.kernel, g, GaussLegendre(8))
                val opRed = VolterraOperator(reduced, g, GaussLegendre(8))
                // g = f'/K(t,t) and g' = f''/K(t,t) — analytically (K(t,t)=1)
                val gEff = memo { t: Double -> opOrig.applyDeriv(t) { s -> m5.exact(s) } }
                val gEffD = memo { t: Double -> opOrig.applyDeriv2(t, { s -> m5.exact(s) }, m5.exactD) }
                val solver = VolterraSecondKindSolver(
                    basis, funcs, opRed, 1.0, RhsWithDerivatives(gEff, gEffD),
                )
                for (tag in listOf("base", "sloan", "kulkarni", "iterKulkarni")) {
                    val ev = when (tag) {
                        "base" -> solver.base().eval
                        "sloan" -> solver.sloan().eval
                        "kulkarni" -> solver.kulkarni().eval
                        else -> solver.iteratedKulkarni().eval
                    }
                    emit("m5analytic|$sysTag|xi1|$n|$tag|eh", errorEh(m5.exact, ev, g))
                }
            }
        }
        dump("m5cmp")
    }

    @Test
    fun sec4TableFreq() {
        header("freq")
        freqTable()
        dump("freq")
    }
}
