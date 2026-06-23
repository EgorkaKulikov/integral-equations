package bench

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.ParallelAssembly
import numerics.functionals.ProjFunctionals
import solvers.fredholm.FredholmOperator
import solvers.fredholm.ModelProblem
import solvers.fredholm.SecondKindSolver

// ============================================================================
// Lightweight, dependency-free HPC benchmark harness.
//
//   (A) Scaling study   : wall-clock time vs N for the full Fredholm F2 solve.
//   (B) Speedup study   : sequential vs parallel matrix assembly on the SAME
//                         code path (toggled via ParallelAssembly.parallelEnabled).
//
// No JMH, no coroutines. Plain System.nanoTime, warm-up + repetitions + median.
// Does NOT change solver logic; mirrors FredholmGoldenTest's construction API.
// ============================================================================

/** Builds the representative Fredholm F2 second-kind solver for system size dim = n+2. */
private fun buildF2Solver(n: Int): SecondKindSolver {
    val grid = Grid.uniform(n)
    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
    val funcs = ProjFunctionals(basis)
    val op = FredholmOperator(ModelProblem.F2.kernel, grid, GaussLegendre(8))
    return SecondKindSolver(
        basis, funcs, op, 1.0,
        { t -> ModelProblem.F2.rhsExact(t, op) },
        { t -> ModelProblem.F2.rhsExactDeriv(t, op) },
    )
}

/** Median of a (small) sample, in nanoseconds. */
private fun medianNanos(samples: LongArray): Long {
    val s = samples.copyOf()
    s.sort()
    val mid = s.size / 2
    return if (s.size % 2 == 1) s[mid] else (s[mid - 1] + s[mid]) / 2
}

private fun ms(nanos: Long): Double = nanos / 1_000_000.0

/** Time a single full solve (matrix assembly + dense solve). Returns elapsed nanos. */
private fun timeSolve(n: Int): Long {
    val solver = buildF2Solver(n)
    val t0 = System.nanoTime()
    val sol = solver.base()
    // Touch the result so JIT cannot elide the solve.
    sinkBlackhole += sol.eval(0.37)
    return System.nanoTime() - t0
}

/** Black-hole accumulator: prevents dead-code elimination of the solve result. */
@Volatile
private var sinkBlackhole: Double = 0.0

/** (A) Scaling study: time vs N. */
private fun scalingStudy(ns: List<Int>, reps: Int) {
    println("== (A) Scaling study: full Fredholm F2 solve (matrix assembly + dense solve) ==")
    // Warm up at the largest n: let the JIT and OpenBLAS warm.
    val warmN = ns.maxOrNull() ?: return
    repeat(3) { timeSolve(warmN) }

    println(String.format("%6s | %6s | %12s | %12s", "n", "dim", "median_ms", "min_ms"))
    println("-".repeat(46))
    for (n in ns) {
        val samples = LongArray(reps) { timeSolve(n) }
        val med = medianNanos(samples)
        val mn = samples.min()
        println(String.format("%6d | %6d | %12.3f | %12.3f", n, n + 2, ms(med), ms(mn)))
    }
    println()
}

/** Time only the matrix-assembly hot path (matrixM) of the F2 solver. Returns elapsed nanos. */
private fun timeAssembly(solver: SecondKindSolver): Long {
    val t0 = System.nanoTime()
    val m = solver.matrixM()
    sinkBlackhole += m[0][0]
    return System.nanoTime() - t0
}

/** (B) Parallel speedup study on the assembly hot path (same code, toggled flag). */
private fun speedupStudy(n: Int, reps: Int) {
    println("== (B) Parallel speedup study: matrix assembly (matrixM), n=$n dim=${n + 2} ==")
    val solver = buildF2Solver(n)
    val saved = ParallelAssembly.parallelEnabled
    try {
        // --- Sequential baseline ---
        ParallelAssembly.parallelEnabled = false
        repeat(3) { timeAssembly(solver) } // warm up
        val seq = LongArray(reps) { timeAssembly(solver) }
        val seqMed = medianNanos(seq)

        // --- Parallel ---
        ParallelAssembly.parallelEnabled = true
        repeat(3) { timeAssembly(solver) } // warm up
        val par = LongArray(reps) { timeAssembly(solver) }
        val parMed = medianNanos(par)

        val speedup = seqMed.toDouble() / parMed.toDouble()
        println(String.format("%18s | %12s", "metric", "value"))
        println("-".repeat(34))
        println(String.format("%18s | %12.3f", "sequential_ms", ms(seqMed)))
        println(String.format("%18s | %12.3f", "parallel_ms", ms(parMed)))
        println(String.format("%18s | %12.2f", "speedup (seq/par)", speedup))
        println(String.format("%18s | %12d", "cores", Runtime.getRuntime().availableProcessors()))
    } finally {
        // Restore production default (parallel ON).
        ParallelAssembly.parallelEnabled = saved
    }
    println()
}

/**
 * Benchmark entry point.
 *
 * args (all optional):
 *   args[0] = comma-separated list of n for the scaling study (default 16,32,64,128,256)
 *   args[1] = repetitions K (default 7)
 *   args[2] = n for the speedup study (default 256)
 */
fun main(args: Array<String>) {
    val ns = if (args.isNotEmpty()) args[0].split(",").map { it.trim().toInt() }
    else listOf(16, 32, 64, 128, 256)
    val reps = if (args.size > 1) args[1].toInt() else 7
    val speedupN = if (args.size > 2) args[2].toInt() else 256

    println("============================================================")
    println(" HPC benchmark harness (minimal splines / Fredholm F2)")
    println(" cores=${Runtime.getRuntime().availableProcessors()}" +
            "  java=${System.getProperty("java.version")}" +
            "  os.arch=${System.getProperty("os.arch")}")
    println(" reps=$reps  scaling n=$ns  speedup n=$speedupN  dim=n+2")
    println("============================================================")
    println()

    scalingStudy(ns, reps)
    speedupStudy(speedupN, reps)

    // Defensive: production default must be parallel.
    check(ParallelAssembly.parallelEnabled) { "parallelEnabled must be restored to true" }
    println("Done. ParallelAssembly.parallelEnabled=${ParallelAssembly.parallelEnabled} (production default).")
}
