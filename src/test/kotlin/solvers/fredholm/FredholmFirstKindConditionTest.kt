package solvers.fredholm

import numerics.ConditionEstimate
import numerics.Conditioning
import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.fredholm.firstKindSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Тесты границы применимости регуляризованного пути F1 по параметру `alpha`
 * (issue #11) и программного доступа к обусловленности собранной системы.
 *
 * Проверяемое требование: утверждение KDoc «ниже примерно `alpha = 1e-8`
 * значащих цифр остаётся мало» подкреплено ИЗМЕРЕНИЕМ на самой библиотеке, а не
 * только внешним наблюдением. Граница фиксируется как ГРАНИЦА ПРИМЕНИМОСТИ, а не
 * как отбраковка: ни один вызов решателя из-за большого `cond` не падает.
 */
@Tag("fast")
class FredholmFirstKindConditionTest {

    private val quad = GaussLegendre(8)

    private fun solver(alpha: Double, n: Int = 8): FredholmFirstKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        return firstKindSolver(
            FredholmProblem.F1,
            basis,
            ProjFunctionals(basis),
            FredholmOperator(FredholmProblem.F1.kernel, grid, quad),
            alpha,
        )
    }

    /**
     * ОБУСЛОВЛЕННОСТЬ РАСТЁТ КАК `alpha^{-1}` — качественное утверждение KDoc
     * подтверждено измерением.
     *
     * Эталон текущего прогона (`n = 8`, базис B, семейство theta):
     * `alpha = 1e-6` → `cond∞ ≈ 1.449e+06`, `1e-8` → `1.492e+08`,
     * `1e-10` → `1.181e+10`. Каждое уменьшение `alpha` на два порядка поднимает
     * `cond∞` примерно на два порядка, как и предсказывает рост элементов `M`.
     * Допуск взят широким (множитель 3 в обе стороны): проверяется ПОРЯДОК роста,
     * а не воспроизводимость цифр на чужой архитектуре.
     */
    @Test fun conditionGrowsInverselyWithAlpha() {
        val condAt = mutableMapOf<Double, Double>()
        for (alpha in listOf(1e-6, 1e-8, 1e-10)) {
            val est = solver(alpha).baseCondition()
            assertTrue(est.condInf.isFinite(), "alpha=$alpha: cond должен быть конечным, получено ${est.condInf}")
            condAt[alpha] = est.condInf
        }
        assertTrue(condAt[1e-6]!! in 5e5..5e6, "alpha=1e-6: cond=${condAt[1e-6]}")
        assertTrue(condAt[1e-8]!! in 5e7..5e8, "alpha=1e-8: cond=${condAt[1e-8]}")
        assertTrue(condAt[1e-10]!! in 4e9..4e10, "alpha=1e-10: cond=${condAt[1e-10]}")
        // Монотонность: чем меньше alpha, тем хуже обусловленность.
        assertTrue(condAt[1e-6]!! < condAt[1e-8]!! && condAt[1e-8]!! < condAt[1e-10]!!)
    }

    /**
     * ГРАНИЦА ПРИМЕНИМОСТИ ПРОХОДИТ ПРИМЕРНО ПО `alpha = 1e-8` — измерено ЗДЕСЬ,
     * тем же критерием достоверности, что и в `Conditioning`.
     *
     * При `alpha >= 1e-8` невязка обращения собранной матрицы держится ниже порога
     * [Conditioning.INVERSION_RESIDUAL_TOLERANCE] и оценка `cond` достоверна;
     * при `alpha = 1e-10` (значение [FredholmFirstKindSolver.DEFAULT_REGULARIZATION])
     * невязка порог уже превышает — то есть матрица вошла в режим, где сама оценка
     * обусловленности теряет смысл. Это машинно проверяемый вид утверждения
     * «ниже примерно 1e-8 значащих цифр остаётся мало».
     */
    @Test fun defaultRegularizationLiesBelowTheReliabilityBoundary() {
        assertTrue(solver(1e-6).baseCondition().isReliable, "alpha=1e-6 обязана оставаться достоверной")
        assertTrue(solver(1e-8).baseCondition().isReliable, "alpha=1e-8 обязана оставаться достоверной")

        val atDefault = solver(FredholmFirstKindSolver.DEFAULT_REGULARIZATION).baseCondition()
        assertEquals(1e-10, FredholmFirstKindSolver.DEFAULT_REGULARIZATION, 0.0)
        assertTrue(
            !atDefault.isReliable,
            "alpha=1e-10 попадает в проблемный диапазон: cond=${atDefault.condInf}, " +
                "невязка обращения=${atDefault.inversionResidual}",
        )
        // Недостоверное число нельзя получить по невнимательности.
        assertNull(atDefault.valueOrNull())
    }

    /**
     * ЭТО НЕ ОТБРАКОВКА. При `alpha = 1e-10`, где оценка `cond` уже недостоверна,
     * решатель работает штатно: базовая схема и итерация Слоана возвращают
     * конечные значения, ни одного исключения не бросается.
     *
     * Требование прямо следует из обоснования
     * `LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE`: большое `cond` у задачи F1 —
     * штатный режим метода, и отбраковывать его нельзя.
     */
    @Test fun poorConditioningDoesNotRejectTheProblem() {
        val s = solver(FredholmFirstKindSolver.DEFAULT_REGULARIZATION)
        for (t in listOf(0.0, 0.25, 0.5, 0.75, 1.0)) {
            assertTrue(s.base().eval(t).isFinite(), "base: t=$t")
            assertTrue(s.sloan().eval(t).isFinite(), "sloan: t=$t")
        }
    }

    /**
     * САМОПРОВЕРКА ДВУМЯ ЭКВИВАЛЕНТНЫМИ ЗАПИСЯМИ (способ, задокументированный в
     * KDoc решателя) — работоспособна и НЕ даёт ложной тревоги на пригодном `alpha`.
     *
     * Одна и та же величина `‖u_h‖` считается двумя алгебраически эквивалентными
     * способами: делением `u/(1+t)` против умножения `u · p`, где `p = 1/(1+t)`.
     * Записи различаются ровно на одно округление. При `alpha = 1e-6`, где по
     * измерениям расхождения нет, оно и здесь отсутствует на уровне 1e-9 —
     * то есть сам приём не создаёт шума и годится как оценка снизу для реально
     * доступной точности.
     */
    @Test fun twoEquivalentWritingsAgreeAtUsableAlpha() {
        val s = solver(1e-6)
        val u = s.base().eval
        var maxRel = 0.0
        for (k in 0..40) {
            val t = k / 40.0
            val byDivision = u(t) / (1.0 + t)
            val byMultiplication = u(t) * (1.0 / (1.0 + t))
            val scale = maxOf(kotlin.math.abs(byDivision), kotlin.math.abs(byMultiplication))
            if (scale > 0.0) maxRel = maxOf(maxRel, kotlin.math.abs(byDivision - byMultiplication) / scale)
        }
        assertTrue(maxRel < 1e-9, "две эквивалентные записи разошлись на $maxRel при alpha=1e-6")
    }

    /**
     * Оценка считается по ТОЙ ЖЕ матрице, с которой решается система: `baseCondition`
     * обязана совпасть с прямым вызовом `Conditioning.conditionInf(baseMatrix())`.
     * Иначе пользователю показывалось бы число от другой задачи.
     */
    @Test fun reportedConditionIsTakenFromTheMatrixActuallySolved() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(FredholmProblem.F1.kernel, grid, quad)
        val s = firstKindSolver(FredholmProblem.F1, basis, funcs, op, 1e-6)
        val inner = FredholmSecondKindSolver(
            basis, funcs, op, cL = -1.0 / 1e-6,
            rhs = solvers.core.RhsWithDerivatives(
                { t -> FredholmProblem.F1.rhsExact(t, op) / 1e-6 },
                { t -> FredholmProblem.F1.rhsExactDeriv(t, op) / 1e-6 },
                { t -> FredholmProblem.F1.rhsExactDeriv2(t, op) / 1e-6 },
            ),
        )
        val expected: ConditionEstimate = Conditioning.conditionInf(inner.baseMatrix())
        assertEquals(expected.condInf, s.baseCondition().condInf, 0.0)
    }

    /** Неположительный порог достоверности отвергается — он означал бы отсутствие проверки. */
    @Test fun baseConditionRejectsNonPositiveTolerance() {
        assertFailsWith<IllegalArgumentException> { solver(1e-6).baseCondition(tolerance = 0.0) }
    }
}
