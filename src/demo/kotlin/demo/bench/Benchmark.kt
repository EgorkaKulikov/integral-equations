package demo.bench

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.ParallelAssembly
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MultikCpuBackend
import numerics.backend.ReferenceBackend
import numerics.functionals.ProjFunctionals
import problems.fredholm.FredholmProblem
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import java.io.File
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

// ============================================================================
// Бенчмарк вычислительной эффективности (без внешних зависимостей).
//
//   (A) Масштабирование по размеру задачи: время полного решения от N.
//   (B) Масштабируемость по числу потоков: 1, 2, 4, 8, ... на ОДНОМ И ТОМ ЖЕ
//       коде сборки матрицы, с вычислением ускорения и эффективности.
//
// Используется System.nanoTime, прогрев, повторы и робастная агрегация.
// Результаты можно выгрузить в CSV для приложения к отчёту.
// ============================================================================

/**
 * Параметры запуска бенчмарка.
 *
 * @param problemSizes размеры сетки n для исследования масштабирования (размер системы dim = n+2)
 * @param threadCounts числа потоков для исследования масштабируемости
 * @param speedupSize размер сетки, на котором измеряется масштабируемость по потокам
 * @param repetitions число измеряемых повторов (после прогрева)
 * @param warmupRuns число прогревочных прогонов перед каждой серией измерений
 * @param csvPath путь для выгрузки результатов в CSV; null — не выгружать
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
 * Числа потоков для исследования масштабируемости: степени двойки до числа
 * доступных ядер включительно. Одной или двух точек недостаточно для вывода об
 * эффективности распараллеливания, поэтому берётся вся развёртка.
 */
private fun defaultThreadCounts(): List<Int> {
    val cores = Runtime.getRuntime().availableProcessors()
    val counts = generateSequence(1) { it * 2 }.takeWhile { it <= cores }.toMutableList()
    if (counts.last() != cores) counts.add(cores)
    return counts
}

/** Сводка по выборке измерений: все величины в миллисекундах. */
private data class Measurement(
    val median: Double,
    val min: Double,
    val max: Double,
    val standardDeviation: Double,
) {
    /** Разброс относительно медианы, в процентах — показатель стабильности измерения. */
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

/** Накопитель результата: не даёт JIT удалить вычисление как неиспользуемое. */
@Volatile
private var blackHole: Double = 0.0

/** Строит представительный решатель уравнения Фредгольма (задача F2) размера dim = n+2. */
private fun buildSolver(n: Int): FredholmSecondKindSolver {
    val grid = Grid.uniform(n)
    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
    val funcs = ProjFunctionals(basis)
    val op = FredholmOperator(FredholmProblem.F2.kernel, grid, GaussLegendre(8))
    return FredholmSecondKindSolver(
        basis, funcs, op, 1.0,
        { t -> FredholmProblem.F2.rhsExact(t, op) },
        { t -> FredholmProblem.F2.rhsExactDeriv(t, op) },
    )
}

/**
 * Измеряет время ПОЛНОГО решения: построение решателя (предвычисление образов
 * базисных сплайнов в узлах квадратуры), сборка матрицы и решение СЛАУ.
 *
 * Ранее построение решателя выполнялось до запуска таймера, из-за чего в замер не
 * попадала предвычислительная фаза, которая для больших n доминирует. Теперь
 * измеряется вся работа, необходимая пользователю для получения решения.
 */
private fun timeFullSolve(n: Int): Long {
    val start = System.nanoTime()
    val solver = buildSolver(n)
    val solution = solver.base()
    blackHole += solution.eval(0.37)
    return System.nanoTime() - start
}

/** Измеряет время только сборки матрицы (без предвычислений и решения СЛАУ). */
private fun timeMatrixAssembly(solver: FredholmSecondKindSolver): Long {
    val start = System.nanoTime()
    val matrix = solver.matrixM()
    blackHole += matrix[0][0]
    return System.nanoTime() - start
}

private fun repeatMeasurement(repetitions: Int, warmupRuns: Int, action: () -> Long): Measurement {
    repeat(warmupRuns) { action() }
    val samples = LongArray(repetitions) { action() }
    return summarize(samples)
}

/**
 * Исследование (A): время полного решения в зависимости от размера задачи.
 *
 * Прогрев выполняется перед КАЖДЫМ размером, а не однократно: иначе JIT остаётся
 * оптимизированным под профиль другого размера и малые n измеряются недостоверно.
 */
private fun runScalingStudy(config: BenchmarkConfig): List<Pair<Int, Measurement>> {
    println()
    println("(A) Масштабирование: полное решение уравнения Фредгольма (задача F2, базис B)")
    println("-".repeat(78))
    println("%6s %8s %12s %12s %12s %12s".format("n", "dim", "медиана,мс", "мин,мс", "макс,мс", "разброс,%"))
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
 * Исследование (B): масштабируемость сборки матрицы по числу потоков.
 *
 * Число потоков задаётся ЯВНО через выделенный [ForkJoinPool] с нужным
 * parallelism, а не берётся из общего пула — иначе число потоков не является
 * управляемым параметром эксперимента и результат невоспроизводим.
 *
 * Для каждой точки вычисляются ускорение S(p) = T(1)/T(p) и эффективность
 * E(p) = S(p)/p, а также оценка последовательной доли по формуле Карпа–Флэтта
 * f = (1/S(p) - 1/p) / (1 - 1/p), позволяющая судить о пределе масштабируемости.
 */
private fun runScalabilityStudy(config: BenchmarkConfig): List<Triple<Int, Measurement, Double>> {
    val solver = buildSolver(config.speedupSize)
    println()
    println("(B) Масштабируемость сборки матрицы по числу потоков (n = ${config.speedupSize})")
    println("-".repeat(78))
    println(
        "%8s %12s %12s %12s %10s %10s %10s".format(
            "потоков", "медиана,мс", "мин,мс", "разброс,%", "ускор.", "эффект.", "посл.доля",
        ),
    )

    // Опорная точка: строго последовательное выполнение того же кода.
    val sequential = try {
        ParallelAssembly.parallelEnabled = false
        repeatMeasurement(config.repetitions, config.warmupRuns) { timeMatrixAssembly(solver) }
    } finally {
        ParallelAssembly.parallelEnabled = true
    }
    println(
        "%8s %12.3f %12.3f %12.1f %10s %10s %10s".format(
            "1 (посл.)", sequential.median, sequential.min, sequential.spreadPercent, "1.00", "1.00", "-",
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
 * Исследование (C): сравнение бэкендов линейной алгебры.
 *
 * Зачем: нативный бэкенд (multik/OpenBLAS) платит за каждый вызов конвертацией
 * `Array<DoubleArray>` <-> `NDArray`, а `fromD2` читает результат поэлементно. На малых
 * размерностях эти накладные расходы могут перевешивать выигрыш от нативного BLAS.
 * Решение об оптимизации должно опираться на измерения, а не на предположения.
 *
 * Измеряются четыре операции на размерностях 16, 64, 256, 1024. Данные
 * детерминированы (фиксированное зерно), матрица для `solve` строго диагонально
 * доминирующая — иначе на больших размерностях случайная матрица вырождается.
 */
private fun runBackendComparison(config: BenchmarkConfig) {
    println()
    println("(C) Сравнение бэкендов линейной алгебры (медиана, мс)")
    println("-".repeat(78))
    println(
        "%6s %10s %14s %14s %10s".format(
            "размер", "операция", "multik,мс", "reference,мс", "multik/ref",
        ),
    )

    val savedBackend = Backends.active
    try {
        for (size in listOf(16, 64, 256, 1024)) {
            val random = kotlin.random.Random(seed = 20240517 + size)
            val a = Array(size) { DoubleArray(size) { random.nextDouble(-1.0, 1.0) } }
            val b = Array(size) { DoubleArray(size) { random.nextDouble(-1.0, 1.0) } }
            val x = DoubleArray(size) { random.nextDouble(-1.0, 1.0) }
            // Строго диагонально доминирующая матрица — гарантированно невырожденная.
            val solvable = Array(size) { i ->
                DoubleArray(size) { j -> if (i == j) size + 1.0 else a[i][j] / size }
            }

            val operations = linkedMapOf<String, (LinAlgBackend) -> Double>(
                "matVec" to { backend -> backend.matVec(a, x)[0] },
                "matMat" to { backend -> backend.matMat(a, b)[0][0] },
                "addScaled" to { backend -> backend.addScaled(a, b, 1.5)[0][0] },
                "solve" to { backend -> backend.solve(solvable, x)[0] },
            )

            for ((operationName, operation) in operations) {
                val timings = LinkedHashMap<String, Measurement>()
                for (backend in listOf<LinAlgBackend>(MultikCpuBackend, ReferenceBackend)) {
                    // matMat и solve кубичны: на 1024 повторы сокращаются, чтобы бенчмарк
                    // завершался за разумное время даже на медленном JVM-бэкенде.
                    val heavy = operationName == "matMat" || operationName == "solve"
                    val repetitions = if (size >= 512 && heavy) 3 else config.repetitions
                    val warmups = if (size >= 512 && heavy) 1 else config.warmupRuns
                    timings[backend.name] = repeatMeasurement(repetitions, warmups) {
                        val start = System.nanoTime()
                        blackHole += operation(backend)
                        System.nanoTime() - start
                    }
                }
                val multik = timings.getValue(MultikCpuBackend.name)
                val reference = timings.getValue(ReferenceBackend.name)
                val ratio = multik.median / reference.median
                println(
                    "%6d %10s %14.4f %14.4f %10.2f".format(
                        size, operationName, multik.median, reference.median, ratio,
                    ),
                )
            }
        }
    } finally {
        Backends.use(savedBackend)
    }
    println()
    println(" Колонка multik/ref: <1 — нативный бэкенд быстрее, >1 — быстрее чистый JVM.")
}

/** Выгружает результаты в CSV, чтобы измерения можно было приложить к отчёту. */
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
    println("Результаты выгружены в ${file.absolutePath}")
}

/**
 * Печатает конфигурацию окружения. Без неё числа невоспроизводимы: результат
 * зависит от процессора, версии JVM и — критично — от активного бэкенда линейной
 * алгебры, который может тихо откатиться на реализацию без нативного BLAS.
 */
private fun printEnvironment(config: BenchmarkConfig) {
    println("=".repeat(78))
    println("Бенчмарк вычислительной эффективности")
    println("=".repeat(78))
    println("ОС              : ${System.getProperty("os.name")} ${System.getProperty("os.version")}")
    println("Архитектура     : ${System.getProperty("os.arch")}")
    println("Доступно ядер   : ${Runtime.getRuntime().availableProcessors()}")
    println("JVM             : ${System.getProperty("java.vm.name")} ${System.getProperty("java.version")}")
    println("Бэкенд линалг.  : ${Backends.active.name}")
    println("Повторов        : ${config.repetitions} (прогрев: ${config.warmupRuns} перед каждой серией)")
    println("Агрегация       : медиана; приводятся мин., макс. и разброс")
    println("Размеры задачи  : ${config.problemSizes.joinToString(", ")}")
    println("Числа потоков   : ${config.threadCounts.joinToString(", ")}")
}

/**
 * Точка входа бенчмарка.
 *
 * Аргументы (все необязательные):
 *   1. размеры задачи через запятую, например `16,32,64`;
 *   2. число повторов;
 *   3. размер задачи для исследования масштабируемости;
 *   4. путь к CSV-файлу для выгрузки результатов.
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
    println("Примечания к интерпретации:")
    println(" - ускорение измерено на сборке матрицы; плотное решение СЛАУ дополнительно")
    println("   использует внутреннюю многопоточность нативного BLAS и здесь не разделяется;")
    println(" - последовательная доля оценена по формуле Карпа–Флэтта: рост её значения")
    println("   с числом потоков указывает на накладные расходы распараллеливания;")
    println(" - при разбросе свыше 10 % измерение считать ориентировочным.")
    if (blackHole.isNaN()) println("(недостижимо: $blackHole)")
}
