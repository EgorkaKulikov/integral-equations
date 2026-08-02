package numerics

import numerics.backend.MultikCpuBackend
import numerics.backend.ReferenceBackend
import numerics.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.uryson.UrysonProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * ПРОВОДКА [NumericsContext] через решатели.
 *
 * Мотивация. Контекст проходит через пять классов ([solvers.core.SecondKindSolverCore],
 * оба решателя второго рода, `UrysonSolver`/`CollocationCore`/`SplineSpace`,
 * [numerics.functionals.FunctionalFamily]), и до появления этих тестов НИ ОДИН тест не
 * строил решатель с НЕдефолтным контекстом. Поэтому ошибка проводки — например, если бы
 * где-то внутри остался `NumericsContext.default()` вместо переданного значения —
 * не проявлялась бы никак: все прогоны шли на одном и том же дефолтном бэкенде.
 *
 * Задача выбрана СОЗНАТЕЛЬНО хорошо обусловленной (Фредгольм II рода, F2:
 * `K(t,s) = 1/(1+t+s)`, гладкое решение). На уравнениях ПЕРВОГО рода смена бэкенда
 * штатно даёт расхождение до 5.7e-2 из-за плохой обусловленности — там сравнение
 * бэкендов проверяло бы обусловленность задачи, а не проводку контекста.
 */
@Tag("fast")
class NumericsContextWiringTest {

    private val problem = FredholmProblem.F2

    /** Строит решатель F2 целиком в ОДНОМ контексте (семейство функционалов — тоже). */
    private fun solver(ctx: NumericsContext, n: Int = 16): FredholmSecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis, ctx)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        return FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            ctx = ctx,
        )
    }

    /** Точки сравнения: узлы и внутренние точки, включая оба конца отрезка. */
    private val samplePoints = doubleArrayOf(0.0, 0.13, 0.37, 0.5, 0.71, 0.99, 1.0)

    /**
     * Решатель на reference-бэкенде согласуется с решателем на multik.
     *
     * Это и есть проверка, что переданный бэкенд ДОХОДИТ до всех мест, где решается СЛАУ:
     * если бы контекст где-то терялся, обе ветви считались бы одним бэкендом и тест
     * проходил бы тривиально, — поэтому ниже отдельно проверяется, что бэкенды РАЗНЫЕ
     * и что результаты при этом НЕ побитово равны (то есть разный путь LU реально задействован).
     */
    @Test
    fun solverAgreesAcrossBackends() {
        val multik = solver(NumericsContext(backend = MultikCpuBackend)).base()
        val reference = solver(NumericsContext(backend = ReferenceBackend)).base()
        for (t in samplePoints) {
            val a = multik.eval(t)
            val b = reference.eval(t)
            assertTrue(
                kotlin.math.abs(a - b) <= 1e-9,
                "t=$t: multik=$a, reference=$b, |разность|=${kotlin.math.abs(a - b)} > 1e-9",
            )
        }
    }

    /**
     * Контекст РЕАЛЬНО долетает до бэкенда: два бэкенда дают ЧИСЛЕННО РАЗНЫЙ (хоть и
     * согласованный) результат хотя бы в одной точке.
     *
     * Без этой проверки предыдущий тест был бы самоподтверждающимся: если бы проводка
     * была сломана и оба решателя считали дефолтным бэкендом, расхождение было бы РОВНО
     * нулевым и допуск 1e-9 выполнился бы автоматически. Здесь фиксируется обратное:
     * пути вычисления различаются, значит параметр `backend` действительно используется.
     */
    @Test
    fun differentBackendsTakeDifferentComputationPaths() {
        val multik = solver(NumericsContext(backend = MultikCpuBackend)).base()
        val reference = solver(NumericsContext(backend = ReferenceBackend)).base()
        val anyBitwiseDifference = samplePoints.any { t -> multik.eval(t) != reference.eval(t) }
        assertTrue(
            anyBitwiseDifference,
            "Оба бэкенда дали ПОБИТОВО одинаковый результат во всех точках. Либо проводка " +
                "контекста сломана (оба решателя считают одним бэкендом), либо реализации LU совпали.",
        )
    }

    /**
     * `parallel = false` и `parallel = true` дают ПОБИТОВО идентичное решение.
     *
     * Для самого [ParallelAssembly] это уже доказано напрямую, но здесь свойство
     * проверяется СКВОЗЬ решатель: сборка матрицы M идёт через `ctx.parallel`, и если бы
     * параллельный путь менял порядок накопления, числа разошлись бы в младших битах.
     */
    @Test
    fun parallelFlagDoesNotChangeResultBitwise() {
        val sequential = solver(NumericsContext(parallel = false)).base()
        val parallel = solver(NumericsContext(parallel = true)).base()
        for (t in samplePoints) {
            val s = sequential.eval(t)
            val p = parallel.eval(t)
            assertTrue(s == p, "t=$t: seq=$s, par=$p — сборка обязана быть побитово идентичной")
        }
    }

    /** Контекст по умолчанию — РАЗДЕЛЯЕМЫЙ экземпляр, а не новый объект на каждый вызов. */
    @Test
    fun defaultContextIsSharedInstance() {
        assertSame(
            NumericsContext.default(), NumericsContext.default(),
            "NumericsContext.default() обязан отдавать один и тот же объект: иначе дефолтное " +
                "значение параметра аллоцировало бы объект на каждое построение решателя.",
        )
    }

    /** Равенство контекстов — ПО ЗНАЧЕНИЮ: два одинаково настроенных контекста совместимы. */
    @Test
    fun contextEqualityIsByValue() {
        assertEquals(NumericsContext(backend = ReferenceBackend), NumericsContext(backend = ReferenceBackend))
        assertTrue(NumericsContext(parallel = false) != NumericsContext(parallel = true))
    }

    /**
     * НЕГАТИВНЫЙ: решатель и семейство функционалов с РАЗНЫМИ контекстами — громкий отказ.
     *
     * Ровно тот сценарий, ради которого добавлена валидация: семейство решает свои
     * крошечные СЛАУ одним бэкендом, решатель — другим, и части ОДНОЙ задачи считаются
     * разными реализациями LU. Раньше это проходило молча.
     */
    @Test
    fun mismatchedFunctionalsContextFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        // Семейство построено на reference, решатель просят считать на multik.
        val funcs = ProjFunctionals(basis, NumericsContext(backend = ReferenceBackend))
        val ex = assertFailsWith<IllegalArgumentException> {
            FredholmSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives({ t -> problem.rhsExact(t, op) }, { t -> problem.rhsExactDeriv(t, op) }),
                ctx = NumericsContext(backend = MultikCpuBackend),
            )
        }
        val message = ex.message!!
        assertTrue(message.contains("funcs"), "Сообщение обязано называть зависимость: $message")
        assertTrue(
            message.contains(ReferenceBackend.name) && message.contains(MultikCpuBackend.name),
            "Сообщение обязано называть ОБА бэкенда, чтобы расхождение было видно: $message",
        )
    }

    /** Несовпадение по флагу `parallel` тоже отбраковывается: контекст сравнивается целиком. */
    @Test
    fun mismatchedParallelFlagFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val funcs = ProjFunctionals(basis, NumericsContext(parallel = false))
        assertFailsWith<IllegalArgumentException> {
            FredholmSecondKindSolver(
                basis, funcs, op, 1.0,
                RhsWithDerivatives({ t -> problem.rhsExact(t, op) }, { t -> problem.rhsExactDeriv(t, op) }),
                ctx = NumericsContext(parallel = true),
            )
        }
    }

    /** Согласованные контексты (в том числе дефолтные у всех участников) проходят. */
    @Test
    fun matchingContextsAreAccepted() {
        for (ctx in listOf(
            NumericsContext.default(),
            NumericsContext(backend = ReferenceBackend),
            NumericsContext(backend = MultikCpuBackend, parallel = false),
        )) {
            val solution = solver(ctx, n = 8).base()
            assertTrue(solution.eval(0.5).isFinite(), "ctx=${ctx.describe()}: решение обязано быть числом")
        }
    }

    // ==== Урысон: у него ДВЕ зависимости с контекстом — funcs И space ==============

    /**
     * Решатель Урысона II рода согласован между бэкендами.
     *
     * Отдельно от Фредгольма: у Урысона своя ветвь проводки — через `CollocationCore`
     * (якобиан и шаг Ньютона) и `SplineSpace`.
     */
    @Test
    fun urysonSolverAgreesAcrossBackends() {
        fun solve(ctx: NumericsContext): (Double) -> Double {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = ProjFunctionals(basis, ctx)
            val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8), ctx)
            val op = solvers.uryson.UrysohnOperator(UrysonProblem.A.kernel, grid, GaussLegendre(8))
            return problems.uryson.secondKindSolver(UrysonProblem.A, basis, funcs, space, op, ctx = ctx)
                .base().eval
        }
        val multik = solve(NumericsContext(backend = MultikCpuBackend))
        val reference = solve(NumericsContext(backend = ReferenceBackend))
        for (t in samplePoints) {
            val a = multik(t)
            val b = reference(t)
            assertTrue(
                kotlin.math.abs(a - b) <= 1e-9,
                "Урысон t=$t: multik=$a, reference=$b, |разность|=${kotlin.math.abs(a - b)} > 1e-9",
            )
        }
    }

    /**
     * НЕГАТИВНЫЙ для `space`: именно это расхождение нашло ревью.
     *
     * `UrysonFirstKindSolver.solveMorozov` считает стабилизатор `Omega` через
     * `space.ctx.backend`, а систему Гаусса–Ньютона — через свой `ctx.backend`.
     * До валидации две части одного критерия Морозова могли считаться разными LU.
     */
    @Test
    fun mismatchedSplineSpaceContextFailsLoudly() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val ctx = NumericsContext(backend = MultikCpuBackend)
        val funcs = ProjFunctionals(basis, ctx)
        // space построен на ДРУГОМ бэкенде, чем решатель и семейство.
        val space = solvers.uryson.SplineSpace(
            basis, GaussLegendre(8), NumericsContext(backend = ReferenceBackend),
        )
        val op = solvers.uryson.UrysohnOperator(UrysonProblem.A.kernel, grid, GaussLegendre(8))
        val ex = assertFailsWith<IllegalArgumentException> {
            solvers.uryson.UrysonFirstKindSolver(basis, funcs, space, op, ctx = ctx)
        }
        assertTrue(ex.message!!.contains("space"), "Сообщение обязано назвать 'space': ${ex.message}")
    }
}
