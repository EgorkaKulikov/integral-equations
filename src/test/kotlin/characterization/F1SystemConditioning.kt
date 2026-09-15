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
 * ГРАНИЦА ПРЯМОЙ ОШИБКИ для систем задачи F1 — механика класса
 * [BaselineClass.SENSITIVE], общая для гейта [EhCharacterizationTest],
 * диагностики [F1ConditioningTest] и инструмента [BaselineClassifier].
 *
 * Система строится ТЕМ ЖЕ кодом, что и в [FredholmFirstKindSolver]: внутренний
 * [FredholmSecondKindSolver] с `c_L = -1/alpha` и правой частью `f/alpha`.
 * Измеряются `cond_1(A)` (LAPACK `dgecon` из LU) и относительная обратная ошибка
 * `omega`, обе — В ТОМ ЖЕ ПРОГОНЕ и на ТОМ ЖЕ бэкенде, что и сверяемое значение.
 * Отсюда граница `2*cond*max(omega,eps)*||u||inf`: каждое из двух решений отстоит
 * от точного не более чем на `cond*omega*||u||`, отсюда множитель 2.
 *
 * ЧТО ЗДЕСЬ ЭМПИРИКА, А ЧТО ТЕОРЕМА. Граница `cond*omega*||u||` описывает ошибку
 * КОЭФФИЦИЕНТОВ `c`. У схемы `sloan` значение `E_h` вычисляется ПОСЛЕ решения —
 * как `fEff(t) + c_L*applyNodes(t, .)` с сокращением в 5e9 раз, — и формально этой
 * границей не покрывается. ИЗМЕРЕНИЕ показало, что фактически покрывается:
 * по всем 54 ключам F1 худшее отношение `|dlt|/граница` = 0.673 (как раз `sloan`),
 * медиана ~0.14, ключей выше границы — 0. Это ЭМПИРИЧЕСКИЙ ФАКТ, а не теорема.
 *
 * СМЕШЕНИЕ НОРМ, названное явно: `Conditioning.conditionEstimate(...).condInf`
 * возвращает оценку в 1-НОРМЕ (имя поля в numerical-core дезориентирует), а `omega`
 * нормирована по бесконечной норме. Для несимметричных матриц `cond_1 != cond_inf`;
 * множитель 2 и полуторакратный запас в худшем ключе это покрывают.
 *
 * ПОЧЕМУ ТОЛЬКО F1 (`base`/`sloan`). Система `(I-M)c=g` доступна снаружи через
 * `SecondKindSolverCore.baseMatrix()`/`vectorG()` только у схем `base` и `sloan`
 * (вторая решает ту же систему). У `kulkarni`/`nystrom`/Урысона матрица приватна
 * либо СЛАУ нет вовсе — и не нужна: расходится между путями LU только F1.
 * Ключ другой схемы, вышедший за правило `portable`, — ДЕФЕКТ, и [BaselineClassifier]
 * на нём падает, а не расширяет класс.
 */
object F1SystemConditioning {

    /** Множитель «два решения, каждое в пределах cond*omega*||u|| от точного». */
    const val SAFETY_FACTOR = 2.0

    /** Машинное эпсилон: нижняя отсечка для `omega` (обратная ошибка не бывает точнее). */
    const val MACHINE_EPSILON = 2.220446049250313e-16

    /**
     * Верхняя отсечка осмысленности границы. Граница выше 10 % от `||u||inf` означает,
     * что система перестала быть разрешимой: это регрессия, а не «широкая граница».
     */
    const val MAX_BOUND_FRACTION = 0.1

    /** Схемы, у которых сверяемое значение получено из системы `(I-M)c=g`. */
    val SUPPORTED_SCHEMES = setOf("base", "sloan")

    /** Результат измерения системы одного сочетания (система, семейство, n). */
    data class Measurement(
        val cond: Double,
        val omega: Double,
        val uNorm: Double,
        val coeffNormInf: Double,
    ) {
        /** Граница прямой ошибки `2*cond*max(omega,eps)*||u||inf`. */
        val bound: Double get() = SAFETY_FACTOR * cond * max(omega, MACHINE_EPSILON) * uNorm

        /** Достоверна ли граница: вырождение и «бесконечная» граница обязаны РОНЯТЬ гейт. */
        val reliable: Boolean
            get() = cond.isFinite() && cond > 1.0 && omega.isFinite() &&
                bound.isFinite() && bound < MAX_BOUND_FRACTION * uNorm
    }

    private val cache = ConcurrentHashMap<String, Measurement>()

    /** Разбирает ключ `F1.<система>.<семейство>.n<N>.<схема>`; `null` — ключ не из F1-систем. */
    fun parseKey(key: String): Triple<GeneratingSystem, String, Int>? {
        val parts = key.split('.')
        if (parts.size != 5 || parts[0] != "F1") return null
        if (parts[4] !in SUPPORTED_SCHEMES) return null
        // `GeneratingSystem` — не enum, перечисления его значений нет; отображение явное,
        // как и в `verification.PublishedValuesTest.system(name)`.
        val system = when (parts[1]) {
            "B" -> GeneratingSystem.B
            "H" -> GeneratingSystem.H
            "T" -> GeneratingSystem.T
            else -> return null
        }
        val n = parts[3].removePrefix("n").toIntOrNull() ?: return null
        return Triple(system, parts[2], n)
    }

    /** Поддерживается ли ключ механикой границы (то есть допустим ли для него класс `sensitive`). */
    fun supports(key: String): Boolean = parseKey(key) != null

    /** Граница для ключа эталона; `null` — ключ не из F1-систем. */
    fun measureFor(key: String): Measurement? {
        val (system, family, n) = parseKey(key) ?: return null
        return measure(system, family, n)
    }

    /** Измеряет систему сочетания; результат кэшируется — 27 сочетаний стоят ~4 с. */
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
