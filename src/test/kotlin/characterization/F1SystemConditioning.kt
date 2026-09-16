package characterization

import numerics.Conditioning
import numerics.GaussLegendre
import numerics.LinearAlgebra
import numerics.NumericsContext
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmFirstKindSolver
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.abs
import kotlin.math.max

/**
 * THE FORWARD ERROR BOUND for the systems of the problem F1 — the machinery of the class
 * [BaselineClass.SENSITIVE], shared by the gate [EhCharacterizationTest],
 * the diagnostics [F1ConditioningTest] and the tool [BaselineClassifier].
 *
 * The system is built by THE SAME code as in [FredholmFirstKindSolver]: an inner
 * [FredholmSecondKindSolver] with `c_L = -1/alpha` and the right-hand side `f/alpha`.
 * What is measured is `cond_1(A)` (the LAPACK `dgecon` from the LU) and the relative backward error
 * `omega`, both IN THE SAME RUN and on THE SAME backend as the value being cross-checked.
 * Hence the bound `2*cond*max(omega,eps)*||u||inf`: each of the two solutions is at most
 * `cond*omega*||u||` away from the exact one, hence the factor 2.
 *
 * WHAT IS EMPIRICAL HERE AND WHAT IS A THEOREM. The bound `cond*omega*||u||` describes the error
 * OF THE COEFFICIENTS `c`. In the `sloan` scheme the value `E_h` is computed AFTER the solve —
 * as `fEff(t) + c_L*applyNodes(t, .)` with a cancellation by 5e9 times — and is formally not
 * covered by this bound. THE MEASUREMENT showed that in fact it is covered:
 * over all 54 F1 keys the worst ratio `|dlt|/bound` = 0.673 (exactly `sloan`),
 * the median ~0.14, keys above the bound: 0. This is an EMPIRICAL FACT and not a theorem.
 *
 * A MIXING OF NORMS, named explicitly: `Conditioning.conditionEstimate(...).condInf`
 * returns an estimate in the 1-NORM (the field name in numerical-core is misleading), while `omega`
 * is normalized by the infinity norm. For non-symmetric matrices `cond_1 != cond_inf`;
 * the factor 2 and the 1.5-fold margin in the worst key cover this.
 *
 * WHY ONLY F1 (`base`/`sloan`). The system `(I-M)c=g` is available from outside through
 * `SecondKindSolverCore.baseMatrix()`/`vectorG()` only for the schemes `base` and `sloan`
 * (the second solves the same system). For `kulkarni`/`nystrom`/Uryson the matrix is private
 * or there is no linear system at all — and it is not needed: only F1 diverges between the LU paths.
 * A key of another scheme that went beyond the `portable` rule is a DEFECT, and [BaselineClassifier]
 * fails on it instead of widening the class.
 */
object F1SystemConditioning {

    /** The factor "two solutions, each within cond*omega*||u|| of the exact one". */
    const val SAFETY_FACTOR = 2.0

    /** The machine epsilon: the lower cut-off for `omega` (a backward error is never more accurate). */
    const val MACHINE_EPSILON = 2.220446049250313e-16

    /**
     * The upper cut-off of the meaningfulness of the bound. A bound above 10 % of `||u||inf` means
     * that the system stopped being solvable: this is a regression and not a "wide bound".
     */
    const val MAX_BOUND_FRACTION = 0.1

    /** The schemes whose cross-checked value is obtained from the system `(I-M)c=g`. */
    val SUPPORTED_SCHEMES = setOf("base", "sloan")

    /** The result of measuring the system of one combination (system, family, n). */
    data class Measurement(
        val cond: Double,
        val omega: Double,
        val uNorm: Double,
        val coeffNormInf: Double,
    ) {
        /** The forward error bound `2*cond*max(omega,eps)*||u||inf`. */
        val bound: Double get() = SAFETY_FACTOR * cond * max(omega, MACHINE_EPSILON) * uNorm

        /** Whether the bound is trustworthy: a degeneracy and an "infinite" bound must BREAK the gate. */
        val reliable: Boolean
            get() = cond.isFinite() && cond > 1.0 && omega.isFinite() &&
                bound.isFinite() && bound < MAX_BOUND_FRACTION * uNorm
    }

    private val cache = ConcurrentHashMap<String, Measurement>()

    /** Parses the key `F1.<system>.<family>.n<N>.<scheme>`; `null` means the key is not from the F1 systems. */
    fun parseKey(key: String): Triple<GeneratingSystem, String, Int>? {
        val parts = key.split('.')
        if (parts.size != 5 || parts[0] != "F1") return null
        if (parts[4] !in SUPPORTED_SCHEMES) return null
        // `GeneratingSystem` is not an enum, there is no enumeration of its values; the mapping is explicit,
        // as in `verification.PublishedValuesTest.system(name)`.
        val system = when (parts[1]) {
            "B" -> GeneratingSystem.B
            "H" -> GeneratingSystem.H
            "T" -> GeneratingSystem.T
            else -> return null
        }
        val n = parts[3].removePrefix("n").toIntOrNull() ?: return null
        return Triple(system, parts[2], n)
    }

    /** Whether the key is supported by the bound machinery (that is, whether the class `sensitive` is admissible for it). */
    fun supports(key: String): Boolean = parseKey(key) != null

    /** The bound for a baseline key; `null` means the key is not from the F1 systems. */
    fun measureFor(key: String): Measurement? {
        val (system, family, n) = parseKey(key) ?: return null
        return measure(system, family, n)
    }

    /** Measures the system of a combination; the result is cached — 27 combinations cost ~4 s. */
    fun measure(system: GeneratingSystem, family: String, n: Int): Measurement =
        cache.getOrPut("${system.name}.$family.n$n") { compute(system, family, n) }

    private fun compute(system: GeneratingSystem, family: String, n: Int): Measurement {
        val fp = FredholmProblem.F1
        val alpha = FredholmFirstKindSolver.DEFAULT_REGULARIZATION
        val cL = -1.0 / alpha
        val ctx = NumericsContext.default()
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
        val cond = Conditioning.conditionEstimate(a, ctx).condInf
        val omega = Conditioning.relativeBackwardError(a, b, x)
        val uNorm = grid.breakpoints.maxOf { t -> abs(fp.exact(t)) }
        return Measurement(cond, omega, uNorm, x.maxOf { abs(it) })
    }
}
