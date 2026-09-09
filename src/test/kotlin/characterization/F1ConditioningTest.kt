package characterization

import numerics.Conditioning
import numerics.DenseMatrix
import numerics.GaussLegendre
import numerics.LinearAlgebra
import numerics.NumericsContext
import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmFirstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * ДИАГНОСТИКА ОБУСЛОВЛЕННОСТИ систем задачи F1 (уравнение Фредгольма ПЕРВОГО рода,
 * регуляризация Вазваза с `alpha = 1e-10`, `c_L = -1/alpha`).
 *
 * Зачем. Ключи `F1.*` эталона `baseline-eh.tsv` привязаны к реализации LU: любая
 * смена BLAS/LAPACK сдвигает их на величины порядка 1e-3…1e-1 при допуске 1e-9.
 * Чтобы отличить такой сдвиг от регрессии, нужна граница прямой ошибки, которую
 * даёт сама библиотека: `cond_1(A) · omega`, где `omega` — относительная обратная
 * ошибка полученного решения. Если расхождение решений двух реализаций LU лежит в
 * пределах этой границы, обе реализации «одинаково правы», и эталон можно переснимать
 * по протоколу `docs/baseline-changes.md`; если нет — это регрессия.
 *
 * Система строится ТЕМ ЖЕ кодом, что и в [FredholmFirstKindSolver]: внутренний
 * [FredholmSecondKindSolver] с `c_L = -1/alpha` и правой частью `f/alpha` (см. поле
 * `inner` решателя первого рода). Тест пишет таблицу в `build/reports/f1-conditioning.tsv`
 * (источник чисел для записей в `docs/baseline-changes.md`) и утверждает только то, что
 * решение вообще получено с обратной ошибкой уровня машинной точности — то есть LU
 * устойчив, а всё различие между реализациями объясняется числом обусловленности.
 */
@Tag("fast")
class F1ConditioningTest {

    private data class Row(
        val system: GeneratingSystem, val family: String, val n: Int,
        val cond: Double, val omega: Double, val bound: Double, val coeffNormInf: Double, val solutionNormInf: Double,
    )

    @Test
    fun f1SystemsAreSolvedBackwardStablyAndConditionIsReported() {
        val fp = FredholmProblem.F1
        val alpha = FredholmFirstKindSolver.DEFAULT_REGULARIZATION
        val cL = -1.0 / alpha
        val ctx = NumericsContext.default()
        val rows = ArrayList<Row>()
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            for (family in listOf("theta", "xi1", "xi2")) {
                for (n in listOf(8, 16, 32)) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(system, grid)
                    val funcs = BaselineSnapshotTool.familyFor(family, basis)
                    val op = FredholmOperator(fp.kernel, grid, GaussLegendre(8))
                    val inner = FredholmSecondKindSolver(
                        basis, funcs, op, cL,
                        RhsWithDerivatives(
                            { t -> fp.rhsExact(t, op) / alpha },
                            { t -> fp.rhsExactDeriv(t, op) / alpha },
                            { t -> fp.rhsExactDeriv2(t, op) / alpha },
                        ),
                        true, ctx,
                    )
                    val a = inner.baseMatrix()
                    val b = inner.vectorG()
                    val x = LinearAlgebra.solve(a, b, ctx.backend)
                    val dense = DenseMatrix.fromRows(a)
                    val cond = Conditioning.conditionEstimate(dense, ctx).condInf
                    val omega = Conditioning.relativeBackwardError(dense, b, x)
                    val uNorm = grid.breakpoints.maxOf { t -> abs(fp.exact(t)) }
                    rows.add(Row(system, family, n, cond, omega, cond * max(omega, 1e-16), x.maxOf { abs(it) }, uNorm))
                }
            }
        }
        // Таблица идёт в файл, а не в stdout: 27 строк на каждый прогон засоряют отчёт Gradle.
        val report = File("build/reports").apply { mkdirs() }.resolve("f1-conditioning.tsv")
        report.writeText(
            buildString {
                append("# F1: backend=${Backends.describe()}\n")
                append("система\tсемейство\tn\tcond_1\tomega\tcond*max(omega,1e-16)\t||c||inf\t||u||inf\n")
                for (r in rows) {
                    append(
                        "%s\t%s\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\n".format(
                            Locale.ROOT,
                            r.system.name, r.family, r.n, r.cond, r.omega, r.bound, r.coeffNormInf, r.solutionNormInf,
                        ),
                    )
                }
            },
        )
        println("F1: backend=${Backends.describe()}, таблица обусловленности — ${report.path}")
        val worstOmega = rows.maxOf { it.omega }
        val minCond = rows.minOf { it.cond }
        val maxCond = rows.maxOf { it.cond }
        println("F1: cond_1 in [%.3e, %.3e], max omega = %.3e".format(minCond, maxCond, worstOmega))
        assertTrue(
            worstOmega < 1e-12,
            "Обратная ошибка решения F1-системы $worstOmega выходит за уровень машинной точности: LU неустойчив.",
        )
        assertTrue(
            rows.all { it.cond.isFinite() && it.cond > 1.0 },
            "Оценка обусловленности F1-системы не получена (вырождение или NaN).",
        )
    }
}
