package characterization

import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import solvers.fredholm.FredholmFirstKindSolver
import splines.GeneratingSystem
import java.io.File
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * THE CONDITIONING DIAGNOSTICS of the systems of the problem F1 (a Fredholm equation of the FIRST kind,
 * Wazwaz regularization with `alpha = 1e-10`, `c_L = -1/alpha`).
 *
 * Why. The `F1.*` keys of the baseline `baseline-eh.tsv` are bound to the LU implementation: any
 * change of BLAS/LAPACK shifts them by quantities of order 1e-3…1e-1 at a tolerance of 1e-9.
 * To distinguish such a shift from a regression, a forward error bound is needed, and it is
 * given by the library itself: `cond_1(A) · omega`, where `omega` is the relative backward
 * error of the obtained solution. If the discrepancy of the solutions of two LU implementations lies
 * within this bound, both implementations are "equally right", and the baseline may be re-shot
 * by the protocol of `docs/baseline-changes.md`; if not, it is a regression.
 *
 * The system is built by THE SAME code as in [FredholmFirstKindSolver]: an inner
 * [FredholmSecondKindSolver] with `c_L = -1/alpha` and the right-hand side `f/alpha` (see the field
 * `inner` of the first-kind solver). The test writes a table into `build/reports/f1-conditioning.tsv`
 * (the source of the numbers for the entries in `docs/baseline-changes.md`) and asserts only that
 * a solution was obtained at all with a backward error at the level of machine accuracy — that is, the LU
 * is stable, and the whole difference between the implementations is explained by the condition number.
 */
@Tag("fast")
class F1ConditioningTest {

    private data class Row(
        val system: GeneratingSystem, val family: String, val n: Int,
        val cond: Double, val omega: Double, val bound: Double, val coeffNormInf: Double, val solutionNormInf: Double,
    )

    @Test
    fun f1SystemsAreSolvedBackwardStablyAndConditionIsReported() {
        val rows = ArrayList<Row>()
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            for (family in listOf("theta", "xi1", "xi2")) {
                for (n in listOf(8, 16, 32)) {
                    // The same machinery as for the class `sensitive` of the characterization gate:
                    // the single implementation is [F1SystemConditioning], the test is its consumer.
                    val m = F1SystemConditioning.measure(system, family, n)
                    rows.add(Row(system, family, n, m.cond, m.omega, m.bound, m.coeffNormInf, m.uNorm))
                }
            }
        }
        // The table goes into a file and not to stdout: 27 rows per run clutter the Gradle report.
        val report = File("build/reports").apply { mkdirs() }.resolve("f1-conditioning.tsv")
        report.writeText(
            buildString {
                append("# F1: backend=${Backends.describe()}\n")
                append("system\tfamily\tn\tcond_1\tomega\tcond*max(omega,1e-16)\t||c||inf\t||u||inf\n")
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
        println("F1: backend=${Backends.describe()}, the conditioning table — ${report.path}")
        val worstOmega = rows.maxOf { it.omega }
        val minCond = rows.minOf { it.cond }
        val maxCond = rows.maxOf { it.cond }
        println("F1: cond_1 in [%.3e, %.3e], max omega = %.3e".format(minCond, maxCond, worstOmega))
        assertTrue(
            worstOmega < 1e-12,
            "The backward error of the solution of the F1 system $worstOmega goes beyond the level of machine accuracy: the LU is unstable.",
        )
        assertTrue(
            rows.all { it.cond.isFinite() && it.cond > 1.0 },
            "The conditioning estimate of the F1 system was not obtained (a degeneracy or a NaN).",
        )
    }
}
