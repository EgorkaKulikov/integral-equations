package convergence

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import solvers.fredholm.FredholmOperator
import solvers.fredholm.KernelF
import solvers.fredholm.SecondKindSolver
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * ЕДИНЫЙ КОНТРАКТ ДИАГНОСТИКИ РАСХОДИМОСТИ (задача 4).
 *
 * До введения контракта итерационные схемы вели себя по-разному: комбинированный
 * Nyström бросал исключение, `kulkarniQuasi` в обоих линейных решателях МОЛЧА
 * возвращал последний итерант, а схемы Урысона лишь писали предупреждение в лог.
 * Пользователь библиотеки не мог отличить сошедшийся результат от расходящегося:
 * при программном использовании запись в журнал остаётся незамеченной.
 *
 * Здесь проверяется, что контракт соблюдается: по умолчанию — исключение, а в режиме
 * `throwOnDivergence = false` — результат с `converged = false` и содержательной
 * невязкой.
 *
 * Задача для проверки расходимости выбрана НЕ произвольно: ядро `K = 4` на `[0,1]`
 * даёт норму оператора `||L|| = 4 > 1`, поэтому простая итерация, лежащая в основе
 * схемы Кулкарни для квазиинтерполянтов и комбинированного Nyström, заведомо
 * расходится при любом числе шагов. Это свойство самой задачи, а не следствие
 * малого предела итераций.
 */
@Tag("fast")
class DivergenceContractTest {

    private companion object {
        /** Ядро с нормой оператора 4 > 1: простая итерация обязана расходиться. */
        val DIVERGENT_KERNEL = KernelF(k = { _, _ -> 4.0 })
    }

    /** Строит решатель заведомо расходящейся задачи с заданной политикой. */
    private fun divergentSolver(
        throwOnDivergence: Boolean,
        useQuasiInterpolant: Boolean,
    ): SecondKindSolver {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        // mu — квазиинтерполянт: kulkarni() уходит в итерационную ветвь kulkarniQuasi.
        // theta — проектор: kulkarni() решает СЛАУ, итераций нет.
        val funcs = if (useQuasiInterpolant) AveragingFunctionals(basis) else ProjFunctionals(basis)
        val op = FredholmOperator(DIVERGENT_KERNEL, grid, GaussLegendre(8))
        return SecondKindSolver(
            basis, funcs, op, cL = 1.0,
            fEff = { t -> t },
            fEffDeriv = { 1.0 },
            fEffDeriv2 = { 0.0 },
            throwOnDivergence = throwOnDivergence,
        )
    }

    /**
     * По умолчанию схема Кулкарни для квазиинтерполянта обязана БРОСАТЬ исключение,
     * а не возвращать последний итерант, как было ранее.
     */
    @Test
    fun quasiKulkarniThrowsByDefaultOnDivergence() {
        val solver = divergentSolver(throwOnDivergence = true, useQuasiInterpolant = true)
        val failure = assertFailsWith<IllegalStateException> { solver.kulkarni() }
        val message = failure.message ?: ""
        // Сообщение обязано быть диагностическим: без числа итераций и достигнутой
        // невязки пользователь не может понять, что произошло.
        assertTrue(
            message.contains("сходимость не достигнута", ignoreCase = true),
            "Сообщение должно явно называть причину, получено: $message",
        )
        assertTrue(
            message.contains("итераций") && message.contains("требуется"),
            "Сообщение должно содержать число итераций и требуемую точность, получено: $message",
        )
    }

    /**
     * В режиме `throwOnDivergence = false` тот же вызов обязан вернуть результат,
     * ЯВНО помеченный как несошедшийся, с содержательной невязкой. Это и есть
     * осознанный доступ к несошедшемуся результату для исследовательских сценариев.
     */
    @Test
    fun quasiKulkarniReportsNotConvergedWhenAllowed() {
        val solver = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = true)
        val solution = solver.kulkarni()
        assertFalse(solution.converged, "Расходящийся результат обязан быть помечен converged = false")
        assertTrue(
            solution.iterations > 0,
            "Число выполненных итераций обязано быть осмысленным, получено ${solution.iterations}",
        )
        assertTrue(
            solution.residual > 0.0 && !solution.residual.isNaN(),
            "Достигнутая невязка обязана быть содержательной, получено ${solution.residual}",
        )
    }

    /** Тот же контракт для комбинированного оператора Nyström. */
    @Test
    fun combinedNystromFollowsSameContract() {
        assertFailsWith<IllegalStateException>(
            "По умолчанию комбинированный Nyström обязан бросать исключение при расходимости",
        ) {
            divergentSolver(throwOnDivergence = true, useQuasiInterpolant = false).combinedNystrom()
        }

        val solution = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = false)
            .combinedNystrom()
        assertFalse(solution.converged, "Расходящийся результат обязан быть помечен converged = false")
        assertTrue(solution.residual > 0.0, "Невязка обязана быть содержательной")
    }

    /**
     * Признак сходимости НАСЛЕДУЕТСЯ производными схемами. Итерация Слоана поверх
     * расходящегося приближения сама итераций не делает, но её результат осмыслен
     * лишь тогда, когда осмыслено исходное приближение, — иначе `converged = true`
     * у итерированной схемы скрывал бы расходимость базовой.
     */
    @Test
    fun iteratedSchemesInheritConvergenceFlag() {
        val solver = divergentSolver(throwOnDivergence = false, useQuasiInterpolant = true)
        val iterated = solver.iteratedKulkarni()
        assertFalse(
            iterated.converged,
            "Итерированная схема обязана наследовать признак расходимости базовой",
        )
    }

    /**
     * ПРЯМЫЕ схемы (СЛАУ без итераций) на штатной задаче сообщают тривиальную
     * сходимость: `converged = true`, `iterations = 0`. Проверка защищает от
     * противоположной ошибки — пометки корректного результата как несошедшегося.
     */
    @Test
    fun directSchemesReportTrivialConvergence() {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = SecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        for ((name, solution) in listOf(
            "base" to solver.base(),
            "sloan" to solver.sloan(),
            "kulkarni (проектор)" to solver.kulkarni(),
            "nystrom" to solver.nystrom(),
        )) {
            assertTrue(solution.converged, "Прямая схема $name обязана сообщать converged = true")
            assertTrue(
                solution.iterations == 0,
                "Прямая схема $name итераций не выполняет, получено ${solution.iterations}",
            )
        }
    }
}
