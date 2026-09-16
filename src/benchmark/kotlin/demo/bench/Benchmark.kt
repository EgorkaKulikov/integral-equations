package demo.bench

import numerics.DenseMatrix
import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import numerics.LinearAlgebra
import numerics.NumericsContext
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import splines.functionals.ProjFunctionals
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import java.io.File
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

// ============================================================================
// Benchmark of computational efficiency (no external dependencies).
//
//   (A) Scaling with problem size: time of the full solve versus N.
//   (B) Scalability with the number of threads: 1, 2, 4, 8, ... on ONE AND THE SAME
//       matrix assembly code, with speedup and efficiency computed.
//
// System.nanoTime, warm-up, repetitions and robust aggregation are used.
// The results can be exported to CSV to be attached to a report.
// ============================================================================

/**
 * Benchmark run parameters.
 *
 * @param problemSizes grid sizes n for the scaling study (system size dim = n+2)
 * @param threadCounts thread counts for the scalability study
 * @param speedupSize grid size at which the thread scalability is measured
 * @param repetitions number of measured repetitions (after the warm-up)
 * @param warmupRuns number of warm-up runs before every measurement series
 * @param csvPath path for exporting the results to CSV; null — do not export
 */
private data class BenchmarkConfig(
    val problemSizes: List<Int> = listOf(16, 32, 64, 128, 256),
    val threadCounts: List<Int> = defaultThreadCounts(),
    val speedupSize: Int = 256,
    val repetitions: Int = 7,
    val warmupRuns: Int = 3,
    val csvPath: String? = null,
)

/**
 * Thread counts for the scalability study: powers of two up to the number of
 * available cores inclusive. One or two points are not enough to conclude anything about
 * parallelization efficiency, so the whole sweep is taken.
 */
private fun defaultThreadCounts(): List<Int> {
    val cores = Runtime.getRuntime().availableProcessors()
    val counts = generateSequence(1) { it * 2 }.takeWhile { it <= cores }.toMutableList()
    if (counts.last() != cores) counts.add(cores)
    return counts
}

/** Summary of a measurement sample: all quantities in milliseconds. */
private data class Measurement(
    val median: Double,
    val min: Double,
    val max: Double,
    val standardDeviation: Double,
) {
    /** Spread relative to the median, in percent — an indicator of measurement stability. */
    val spreadPercent: Double get() = if (median > 0.0) 100.0 * (max - min) / median else 0.0
}

private fun summarize(samplesNanos: LongArray): Measurement {
    val sortedMs = samplesNanos.map { it / 1_000_000.0 }.sorted()
    val mid = sortedMs.size / 2
    val median = if (sortedMs.size % 2 == 1) sortedMs[mid] else 0.5 * (sortedMs[mid - 1] + sortedMs[mid])
    val mean = sortedMs.average()
    val variance = sortedMs.sumOf { (it - mean) * (it - mean) } / sortedMs.size
    return Measurement(median, sortedMs.first(), sortedMs.last(), sqrt(variance))
}

/** Result accumulator: keeps the JIT from eliminating the computation as unused. */
@Volatile
private var blackHole: Double = 0.0

/** Builds a representative solver of the Fredholm equation (problem F2) of size dim = n+2. */
private fun buildSolver(n: Int, ctx: NumericsContext = NumericsContext.default()): FredholmSecondKindSolver {
    val grid = Grid.uniform(n)
    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
    val funcs = ProjFunctionals(basis, ctx)
    val op = FredholmOperator(FredholmProblem.F2.kernel, grid, GaussLegendre(8))
    return FredholmSecondKindSolver(
        basis, funcs, op, 1.0,
        RhsWithDerivatives(
            { t -> FredholmProblem.F2.rhsExact(t, op) },
            { t -> FredholmProblem.F2.rhsExactDeriv(t, op) },
        ),
        ctx = ctx,
    )
}

/**
 * Measures the time of the FULL solve: building the solver (precomputing the images
 * of the basis splines at the quadrature nodes), matrix assembly and solving the linear system.
 *
 * Formerly the solver was built before the timer started, so the measurement excluded
 * the precomputation phase, which dominates for large n. Now
 * all the work a user needs to obtain the solution is measured.
 */
private fun timeFullSolve(n: Int): Long {
    val start = System.nanoTime()
    val solver = buildSolver(n)
    val solution = solver.base()
    blackHole += solution.eval(0.37)
    return System.nanoTime() - start
}

/** Measures the time of the matrix assembly only (without precomputation and the linear solve). */
private fun timeMatrixAssembly(solver: FredholmSecondKindSolver): Long {
    val start = System.nanoTime()
    val matrix = solver.matrixM()
    blackHole += matrix[0, 0]
    return System.nanoTime() - start
}

private fun repeatMeasurement(repetitions: Int, warmupRuns: Int, action: () -> Long): Measurement {
    repeat(warmupRuns) { action() }
    val samples = LongArray(repetitions) { action() }
    return summarize(samples)
}

/**
 * Study (A): time of the full solve versus problem size.
 *
 * The warm-up is performed before EVERY size, not once: otherwise the JIT stays
 * optimized for the profile of another size and small n are measured unreliably.
 */
private fun runScalingStudy(config: BenchmarkConfig): List<Pair<Int, Measurement>> {
    println()
    println("(A) Scaling: full solve of the Fredholm equation (problem F2, basis B)")
    println("-".repeat(78))
    println("%6s %8s %12s %12s %12s %12s".format("n", "dim", "median,ms", "min,ms", "max,ms", "spread,%"))
    val results = config.problemSizes.map { n ->
        val measurement = repeatMeasurement(config.repetitions, config.warmupRuns) { timeFullSolve(n) }
        println(
            "%6d %8d %12.3f %12.3f %12.3f %12.1f".format(
                n, n + 2, measurement.median, measurement.min, measurement.max, measurement.spreadPercent,
            ),
        )
        n to measurement
    }
    return results
}

/**
 * Study (B): scalability of the matrix assembly with the number of threads.
 *
 * The thread count is set EXPLICITLY through a dedicated [ForkJoinPool] with the required
 * parallelism instead of being taken from the common pool — otherwise the thread count is not
 * a controlled parameter of the experiment and the result is not reproducible.
 *
 * For every point the speedup S(p) = T(1)/T(p) and the efficiency
 * E(p) = S(p)/p are computed, as well as the serial fraction estimated by the Karp–Flatt formula
 * f = (1/S(p) - 1/p) / (1 - 1/p), which allows judging the scalability limit.
 */
private fun runScalabilityStudy(config: BenchmarkConfig): List<Triple<Int, Measurement, Double>> {
    val solver = buildSolver(config.speedupSize)
    println()
    println("(B) Scalability of the matrix assembly with the number of threads (n = ${config.speedupSize})")
    println("-".repeat(78))
    println(
        "%8s %12s %12s %12s %10s %10s %10s".format(
            "threads", "median,ms", "min,ms", "spread,%", "speedup", "effic.", "serial.fr",
        ),
    )

    // Reference point: strictly sequential execution of THE SAME code — a separate
    // solver with the context `parallel = false`, not a global switch:
    // that way the benchmark does not change the behaviour of other code in the same JVM.
    val sequentialSolver = buildSolver(config.speedupSize, NumericsContext(parallel = false))
    val sequential =
        repeatMeasurement(config.repetitions, config.warmupRuns) { timeMatrixAssembly(sequentialSolver) }
    println(
        "%8s %12.3f %12.3f %12.1f %10s %10s %10s".format(
            "1 (seq.)", sequential.median, sequential.min, sequential.spreadPercent, "1.00", "1.00", "-",
        ),
    )

    val results = mutableListOf<Triple<Int, Measurement, Double>>()
    for (threads in config.threadCounts) {
        val pool = ForkJoinPool(threads)
        val measurement = try {
            pool.submit<Measurement> {
                repeatMeasurement(config.repetitions, config.warmupRuns) { timeMatrixAssembly(solver) }
            }.get()
        } finally {
            pool.shutdown()
            pool.awaitTermination(10, TimeUnit.SECONDS)
        }
        val speedup = sequential.median / measurement.median
        val efficiency = speedup / threads
        val karpFlatt = if (threads > 1) {
            (1.0 / speedup - 1.0 / threads) / (1.0 - 1.0 / threads)
        } else {
            Double.NaN
        }
        println(
            "%8d %12.3f %12.3f %12.1f %10.2f %10.2f %10s".format(
                threads, measurement.median, measurement.min, measurement.spreadPercent,
                speedup, efficiency, if (karpFlatt.isNaN()) "-" else "%.3f".format(karpFlatt),
            ),
        )
        results.add(Triple(threads, measurement, speedup))
    }
    return results
}

/**
 * Study (C): comparison of the linear algebra backends.
 *
 * Why: the native backend (multik/OpenBLAS) pays on every call for the conversion
 * `Array<DoubleArray>` <-> `NDArray`, and `fromD2` reads the result element by element. At small
 * dimensions this overhead may outweigh the gain from the native BLAS.
 * An optimization decision must rest on measurements, not on assumptions.
 *
 * Four operations are measured at dimensions 16, 64, 256, 1024. The data is
 * deterministic (fixed seed), and the matrix for `solve` is strictly diagonally
 * dominant — otherwise a random matrix becomes singular at large dimensions.
 */
private fun runBackendComparison(config: BenchmarkConfig) {
    println()
    println("(C) Comparison of the linear algebra backends (median, ms)")
    println("-".repeat(78))
    println(
        "%6s %10s %14s %14s %10s".format(
            "size", "operation", "multik,ms", "reference,ms", "multik/ref",
        ),
    )

    // The backends are compared DIRECTLY, without substituting global state and without
    // restoring it in finally: every measurement calls its own [LinAlgBackend] instance.
    for (size in listOf(16, 64, 256, 1024)) {
        val random = kotlin.random.Random(seed = 20240517 + size)
        val a = DenseMatrix.build(size, size) { _, _ -> random.nextDouble(-1.0, 1.0) }
        val b = DenseMatrix.build(size, size) { _, _ -> random.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(size) { random.nextDouble(-1.0, 1.0) }
        // A strictly diagonally dominant matrix — guaranteed nonsingular.
        val solvable = DenseMatrix.build(size, size) { i, j ->
            if (i == j) size + 1.0 else a[i, j] / size
        }

        val operations = linkedMapOf<String, (LinAlgBackend) -> Double>(
            "matVec" to { backend -> LinearAlgebra.matVec(a, x, backend)[0] },
            "matMat" to { backend -> LinearAlgebra.matMat(a, b, backend)[0, 0] },
            "addScaled" to { backend -> LinearAlgebra.addScaled(a, b, 1.5, backend)[0, 0] },
            "solve" to { backend -> LinearAlgebra.solve(solvable, x, backend)[0] },
        )

        for ((operationName, operation) in operations) {
            val timings = LinkedHashMap<String, Measurement>()
            for (backend in listOf<LinAlgBackend>(Backends.native(), Backends.java())) {
                // matMat and solve are cubic: at 1024 the repetitions are reduced so that the benchmark
                // finishes in reasonable time even on the slow JVM backend.
                val heavy = operationName == "matMat" || operationName == "solve"
                val repetitions = if (size >= 512 && heavy) 3 else config.repetitions
                val warmups = if (size >= 512 && heavy) 1 else config.warmupRuns
                timings[backend.name] = repeatMeasurement(repetitions, warmups) {
                    val start = System.nanoTime()
                    blackHole += operation(backend)
                    System.nanoTime() - start
                }
            }
            val multik = timings.getValue(Backends.native().name)
            val reference = timings.getValue(Backends.java().name)
            val ratio = multik.median / reference.median
            println(
                "%6d %10s %14.4f %14.4f %10.2f".format(
                    size, operationName, multik.median, reference.median, ratio,
                ),
            )
        }
    }
    println()
    println(" Column multik/ref: <1 — the native backend is faster, >1 — pure JVM is faster.")
}

/** Exports the results to CSV, so that the measurements can be attached to a report. */
private fun exportCsv(
    path: String,
    scaling: List<Pair<Int, Measurement>>,
    scalability: List<Triple<Int, Measurement, Double>>,
) {
    val file = File(path)
    file.parentFile?.mkdirs()
    buildString {
        appendLine("study,parameter,median_ms,min_ms,max_ms,stddev_ms,speedup")
        for ((n, m) in scaling) {
            appendLine("scaling,n=$n,${m.median},${m.min},${m.max},${m.standardDeviation},")
        }
        for ((threads, m, speedup) in scalability) {
            appendLine("scalability,threads=$threads,${m.median},${m.min},${m.max},${m.standardDeviation},$speedup")
        }
    }.let { file.writeText(it) }
    println()
    println("Results exported to ${file.absolutePath}")
}

/**
 * Prints the environment configuration. Without it the numbers are not reproducible: the result
 * depends on the processor, the JVM version and — critically — on the active linear algebra
 * backend, which may silently fall back to an implementation without a native BLAS.
 */
private fun printEnvironment(config: BenchmarkConfig) {
    println("=".repeat(78))
    println("Benchmark of computational efficiency")
    println("=".repeat(78))
    println("OS              : ${System.getProperty("os.name")} ${System.getProperty("os.version")}")
    println("Architecture    : ${System.getProperty("os.arch")}")
    println("Available cores : ${Runtime.getRuntime().availableProcessors()}")
    println("JVM             : ${System.getProperty("java.vm.name")} ${System.getProperty("java.version")}")
    println("LinAlg backend  : ${Backends.default().name}")
    println("Repetitions     : ${config.repetitions} (warm-up: ${config.warmupRuns} before every series)")
    println("Aggregation     : median; min, max and spread are reported")
    println("Problem sizes   : ${config.problemSizes.joinToString(", ")}")
    println("Thread counts   : ${config.threadCounts.joinToString(", ")}")
}

/**
 * Benchmark entry point.
 *
 * Arguments (all optional):
 *   1. problem sizes separated by commas, e.g. `16,32,64`;
 *   2. number of repetitions;
 *   3. problem size for the scalability study;
 *   4. path to the CSV file for exporting the results.
 */
fun main(args: Array<String>) {
    val config = BenchmarkConfig(
        problemSizes = args.getOrNull(0)?.split(",")?.map { it.trim().toInt() }
            ?: BenchmarkConfig().problemSizes,
        repetitions = args.getOrNull(1)?.toInt() ?: BenchmarkConfig().repetitions,
        speedupSize = args.getOrNull(2)?.toInt() ?: BenchmarkConfig().speedupSize,
        csvPath = args.getOrNull(3),
    )

    printEnvironment(config)
    val scaling = runScalingStudy(config)
    val scalability = runScalabilityStudy(config)
    runBackendComparison(config)
    config.csvPath?.let { exportCsv(it, scaling, scalability) }

    println()
    println("Notes on interpretation:")
    println(" - the speedup is measured on the matrix assembly; the dense linear solve additionally")
    println("   uses the internal multithreading of the native BLAS and is not separated here;")
    println(" - the serial fraction is estimated by the Karp-Flatt formula: a growth of its value")
    println("   with the thread count indicates parallelization overhead;")
    println(" - at a spread above 10 % the measurement should be taken as indicative only.")
    if (blackHole.isNaN()) println("(unreachable: $blackHole)")
}
