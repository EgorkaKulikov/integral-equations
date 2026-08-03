package solvers.uryson

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.LinearAlgebra
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.uryson.UrysonProblem
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * КОНТРАКТ ОТЧЁТНОСТИ ньютоновских итераций решателя Урысона (пункт 8.4 спека).
 *
 * Проверяются три утверждения, которые до правки НЕ выполнялись ни в одной из трёх
 * итерационных схем (`solveBase`, `kulkarni`, `nystrom`):
 *
 *  1. `iterations` — число ФАКТИЧЕСКИ выполненных шагов Ньютона, а не число проверок
 *     критерия. Раньше счётчик увеличивался ДО проверки невязки, поэтому при
 *     мгновенной сходимости наружу уходила единица при нуле шагов (замер: `lambda = 0`
 *     давал `iterations = 1` при неизменном векторе коэффициентов).
 *  2. `residual` относится к ВОЗВРАЩАЕМОЙ точке. Раньше при выходе по критерию шага
 *     возвращалась невязка в точке ДО шага, то есть систематически ЗАВЫШЕННАЯ.
 *     Оба фактических случая выхода по шагу (базовая схема, задача B, n=8):
 *     при `tol = 1e-2` сообщалось `2.1409e-2` вместо `4.9945e-5` (в 429 раз хуже),
 *     при `tol = 1e-10` — `2.7402e-10` вместо `4.4409e-15` (в 6.2e4 раз хуже).
 *  3. `converged` достоверен: он выставляется по ФАКТИЧЕСКОЙ невязке, а не по тому,
 *     какой критерий прервал цикл. Раньше малый шаг объявлялся успехом безусловно,
 *     поэтому схема сообщала `converged = true` при невязке ВЫШЕ затребованного `tol`.
 *
 * ЗАЩИТА ОТ ВЫРОЖДЕНИЯ. Три теста ниже перебирают конфигурации и проверяют условие
 * не в каждой из них (например, только там, где схема сошлась, или только там, где
 * выход произошёл по шагу). Такой перебор МОЛЧА обнуляется, если поведение изменится
 * и подходящих конфигураций не останется: тест продолжит быть зелёным, ничего не
 * проверяя. Поэтому каждый из них считает ФАКТИЧЕСКИ ПРОВЕРЕННЫЕ случаи и требует,
 * чтобы их было больше нуля.
 */
@Tag("fast")
class UrysonNewtonReportingTest {

    private val quad = GaussLegendre(8)

    /**
     * Допуск сравнения сообщённой невязки с независимо пересчитанной.
     *
     * Оба числа считаются одним и тем же кодом (`CollocationCore.xiVector`) в одной
     * и той же точке, поэтому фактически совпадают побитово; допуск оставлен на случай
     * иного порядка суммирования при смене бэкенда. Он на много порядков меньше
     * зафиксированных выше расхождений (429 и 6.2e4 раза), поэтому старое поведение
     * этот тест заведомо не прошло бы.
     */
    private val residualMatchTolerance = 1e-12

    /** Нижняя граница допуска базовой схемы — `UrysonSecondKindSolver.NEWTON_TOLERANCE_FLOOR`. */
    private val newtonToleranceFloor = 1e-13

    private fun solverFor(
        problem: UrysonProblem,
        n: Int,
        tol: Double,
        lambdaOverride: Double? = null,
    ): UrysonSecondKindSolver {
        val basis = MinimalSplineBasis(GeneratingSystem.B, Grid.uniform(n))
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, basis.grid, quad)
        return UrysonSecondKindSolver(
            basis = basis,
            funcs = funcs,
            space = space,
            op = op,
            lambda = lambdaOverride ?: problem.lambda,
            rhs = { t -> problem.rhsExact(t, op) },
            tol = tol,
        )
    }

    /** Вектор `theta_j(f)` — он же начальное приближение базовой схемы. */
    private fun thetaOf(solver: UrysonSecondKindSolver): DoubleArray {
        val n = solver.grid.n
        return DoubleArray(n + 2) { solver.funcs.chi(it - 2).apply(solver.rhs, { 0.0 }, { 0.0 }) }
    }

    /**
     * НЕЗАВИСИМЫЙ пересчёт невязки базовой схемы `F(c) = c - theta(f) - lambda Xi(c)`
     * в заданной точке — эталон для проверки поля `residual`.
     */
    private fun baseResidualAt(solver: UrysonSecondKindSolver, coeffs: DoubleArray): Double {
        val core = CollocationCore(solver.basis, solver.funcs, solver.op)
        val thetaF = thetaOf(solver)
        val xi = core.xiVector(coeffs)
        return LinearAlgebra.normInf(
            DoubleArray(coeffs.size) { coeffs[it] - thetaF[it] - solver.lambda * xi[it] },
        )
    }

    /** Итог воспроизведения базовой схемы: то же, что вернул решатель, плюс НАБЛЮДЕНИЯ за циклом. */
    private class Replay(
        val steps: Int,
        val residual: Double,
        val converged: Boolean,
        /** Сколько раз ФАКТИЧЕСКИ вызывался расчёт шага — независимый счётчик шагов. */
        val stepCalls: Int,
        /** Норма последнего шага: если она ниже допуска, цикл вышел ПО ШАГУ. */
        val lastStepNorm: Double,
        val coeffs: DoubleArray,
    )

    /**
     * Воспроизводит базовую схему ЧЕРЕЗ ТОТ ЖЕ хелпер [runNewtonIterations], но с
     * обёртками, наблюдающими за циклом: считает вызовы расчёта шага и запоминает
     * норму последнего шага.
     *
     * Нужно для двух вещей, недоступных снаружи по публичному API:
     *  - узнать, каким из двух критериев вышел цикл (без этого нельзя утверждать,
     *    что предмет правки — выход ПО ШАГУ — вообще был затронут);
     *  - получить НЕЗАВИСИМЫЙ от поля `performedSteps` счётчик выполненных шагов.
     *
     * Верность воспроизведения не постулируется, а ПРОВЕРЯЕТСЯ: каждый тест,
     * использующий этот метод, сверяет `steps` и `residual` с тем, что вернул
     * настоящий `solveBase()`.
     */
    private fun replayBase(solver: UrysonSecondKindSolver, tol: Double): Replay {
        val n = solver.grid.n
        val core = CollocationCore(solver.basis, solver.funcs, solver.op)
        val thetaF = thetaOf(solver)
        val c = thetaF.copyOf()
        var stepCalls = 0
        var lastStepNorm = Double.NaN
        val run = runNewtonIterations(
            x = c,
            maxSteps = UrysonSecondKindSolver.DEFAULT_MAX_ITERATIONS,
            tolerance = maxOf(tol, newtonToleranceFloor),
            residualAt = { current ->
                val xi = core.xiVector(current)
                DoubleArray(n + 2) { current[it] - thetaF[it] - solver.lambda * xi[it] }
            },
            stepAt = { current, f ->
                stepCalls++
                val b = core.bMatrix(current)
                val jacobian = Array(n + 2) { r ->
                    DoubleArray(n + 2) { col -> -solver.lambda * b[r][col] }.also { it[r] += 1.0 }
                }
                val delta = LinearAlgebra.solve(jacobian, DoubleArray(n + 2) { -f[it] })
                lastStepNorm = LinearAlgebra.normInf(delta)
                delta
            },
        )
        return Replay(run.performedSteps, run.residual, run.converged, stepCalls, lastStepNorm, c)
    }

    /**
     * МГНОВЕННАЯ СХОДИМОСТЬ: при `lambda = 0` уравнение превращается в `x = f`, а
     * начальное приближение базовой схемы `c_0 = theta(f)` УЖЕ является решением.
     * Значит, ни одного шага Ньютона не требуется, и честный счётчик обязан дать 0.
     *
     * Это не искусственная конфигурация ради теста, а вырожденный случай самого
     * уравнения (нулевой множитель перед интегральным оператором), достижимый через
     * штатный публичный конструктор.
     *
     * Утверждение усилено проверкой, что вектор коэффициентов НЕ СДВИНУЛСЯ побитово:
     * иначе «ноль шагов» можно было бы сообщать и после фактически сделанного шага.
     */
    @Test
    fun instantConvergenceReportsZeroNewtonSteps() {
        val solver = solverFor(UrysonProblem.A, n = 8, tol = 1e-12, lambdaOverride = 0.0)
        val expectedCoeffs = thetaOf(solver)

        val newton = solver.solveBase()
        assertTrue(newton.converged, "При lambda = 0 базовая схема обязана сойтись")
        assertEquals(
            0,
            newton.iterations,
            "Начальное приближение УЖЕ решение: шагов Ньютона ноль, получено ${newton.iterations}",
        )
        assertEquals(0.0, newton.residual, "Невязка в точном решении обязана быть нулевой")
        for (i in expectedCoeffs.indices) {
            assertEquals(
                expectedCoeffs[i].toRawBits(),
                newton.coeffs[i].toRawBits(),
                "Коэффициент $i изменился, значит шаг всё-таки был выполнен",
            )
        }

        // Схема Кулкарни при lambda = 0 приходит к тому же: G_K(c) = theta(f).
        val kulkarni = solver.kulkarni()
        assertEquals(
            0,
            kulkarni.iterations,
            "Кулкарни при lambda = 0 шагов не делает, получено ${kulkarni.iterations}",
        )
        assertEquals(0.0, kulkarni.residual)

        // Nyström СОЗНАТЕЛЬНО стартует не с theta(f), а с проекции постоянной функции,
        // поэтому один шаг здесь ОБЯЗАН быть — и он ровно один, а не два.
        val nystrom = solver.nystrom()
        assertEquals(
            1,
            nystrom.iterations,
            "Nyström стартует с проекции единицы: нужен ровно один шаг, получено ${nystrom.iterations}",
        )
    }

    /**
     * ВЫХОД ПО КРИТЕРИЮ ШАГА: сообщённая невязка обязана относиться к ВОЗВРАЩАЕМОЙ
     * точке. Проверяется независимым пересчётом.
     *
     * ПОЧЕМУ НУЖЕН СЧЁТЧИК `stepExits`. Для конфигураций, вышедших ПО НЕВЯЗКЕ,
     * утверждение ТАВТОЛОГИЧНО: сообщённое число и есть та самая невязка, посчитанная
     * тем же кодом в той же точке, — совпадение гарантировано независимо от правки.
     * Содержательным тест становится ТОЛЬКО на конфигурациях, вышедших ПО ШАГУ:
     * именно там старый код возвращал невязку предыдущей точки. Поэтому тест
     * подсчитывает такие конфигурации и требует, чтобы их было больше нуля, — иначе
     * он молча выродился бы в проверку тождества.
     *
     * Допуски подобраны замером: на `tol = 1e-2` и `tol = 1e-10` базовая схема для
     * задачи B выходит именно по норме шага (кубическая нелинейность при `lambda = 1`).
     */
    @Test
    fun reportedResidualIsMeasuredAtReturnedPoint() {
        var checked = 0
        var stepExits = 0
        for (tol in listOf(1e-2, 1e-5, 1e-10, 1e-12)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val label = "задача ${problem.name}, tol=$tol"
                val newton = solverFor(problem, n = 8, tol = tol).solveBase()
                val solver = solverFor(problem, n = 8, tol = tol)
                val recomputed = baseResidualAt(solver, newton.coeffs)
                val scale = maxOf(abs(recomputed), abs(newton.residual), 1e-300)
                assertTrue(
                    abs(newton.residual - recomputed) / scale < residualMatchTolerance,
                    "$label: сообщена невязка ${newton.residual}, " +
                        "а в ВОЗВРАЩАЕМОЙ точке она равна $recomputed",
                )
                checked++

                // Классифицируем критерий выхода воспроизведением цикла.
                val replay = replayBase(solverFor(problem, n = 8, tol = tol), tol)
                assertEquals(
                    newton.iterations,
                    replay.steps,
                    "$label: воспроизведение цикла разошлось с solveBase по числу шагов",
                )
                if (replay.steps > 0 && replay.lastStepNorm < maxOf(tol, newtonToleranceFloor)) {
                    stepExits++
                }
            }
        }
        assertTrue(checked > 0, "Тест выродился: ни одна конфигурация не проверена")
        assertTrue(
            stepExits > 0,
            "Тест выродился: НИ ОДНА конфигурация не вышла по критерию ШАГА, а именно этот " +
                "случай и есть предмет правки — на выходе по невязке утверждение тавтологично. " +
                "Подберите допуски заново (замером подтверждены tol=1e-2 и tol=1e-10 на задаче B)",
        )
    }

    /**
     * ДОСТОВЕРНОСТЬ ФЛАГА `converged`: если сходимость объявлена, невязка ОБЯЗАНА быть
     * ниже затребованного допуска.
     *
     * Это не следствие, а самостоятельный контракт, обеспеченный кодом: при выходе по
     * норме шага невязка пересчитывается в новой точке и СРАВНИВАЕТСЯ с допуском.
     * Малый шаг сам по себе успехом не считается — он штатно возникает на плохо
     * обусловленном якобиане и на приближённом разностном якобиане схемы `nystrom`.
     *
     * До правки противоречие воспроизводилось фактически на всех трёх схемах (замер,
     * n=8, задача B): базовая при `tol = 1e-2` сообщала `2.14e-2 > 1e-2`, схема
     * Кулкарни при `tol = 1e-5` — `3.19e-5 > 1e-5`, Nyström при `tol = 1e-8` —
     * `2.56e-8 > 1e-8`. Все три случая включены в перебор ниже.
     *
     * Счётчик `checked` обязателен: без него отказ схем сходиться (`continue` по
     * каждой конфигурации) обнулил бы тест, оставив его зелёным.
     */
    @Test
    fun convergedResultNeverReportsResidualAboveTolerance() {
        var checked = 0
        for (tol in listOf(1e-2, 1e-5, 1e-8, 1e-10)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val solver = solverFor(problem, n = 8, tol = tol)
                val checks = listOf(
                    "base" to solver.base(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                )
                for ((name, solution) in checks) {
                    if (!solution.converged) continue
                    assertTrue(
                        solution.residual < tol,
                        "Задача ${problem.name}, tol=$tol, схема $name: сходимость объявлена, " +
                            "но сообщена невязка ${solution.residual} выше допуска",
                    )
                    checked++
                }
            }
        }
        assertTrue(
            checked > 0,
            "Тест выродился: ни одна схема не сообщила о сходимости, поэтому утверждение " +
                "не проверено ни разу",
        )
    }

    /**
     * СЧЁТЧИК РАВЕН ЧИСЛУ ФАКТИЧЕСКИ ВЫПОЛНЕННЫХ ШАГОВ — сверка с НЕЗАВИСИМЫМ счётом.
     *
     * Независимый счёт — это число вызовов лямбды расчёта шага: она вызывается ровно
     * один раз на каждый выполненный шаг, и её счётчик живёт в тесте, а не в хелпере.
     * Поэтому сверка не зависит от того, как хелпер считает сам, и ловит расхождение
     * даже на единицу.
     *
     * ПОЧЕМУ НЕ МОНОТОННОСТЬ ПО `tol`. Прежняя редакция этого теста проверяла, что
     * ужесточение допуска не уменьшает число шагов, и утверждала, что «ловит сдвиг
     * счётчика на единицу». Это было НЕВЕРНО: старая семантика отличается от новой
     * РАВНОМЕРНЫМ сдвигом на единицу, а монотонность к постоянному сдвигу инвариантна,
     * поэтому мутацию тест не ловил по построению. Сверка с независимым счётом от
     * сдвига не инвариантна и ловит его сразу.
     */
    @Test
    fun stepCountEqualsNumberOfPerformedSteps() {
        var checked = 0
        var withSteps = 0
        for (tol in listOf(1e-1, 1e-2, 1e-5, 1e-10, 1e-12)) {
            for (problem in listOf(UrysonProblem.A, UrysonProblem.B)) {
                val label = "задача ${problem.name}, tol=$tol"
                val reported = solverFor(problem, n = 8, tol = tol).solveBase()
                val replay = replayBase(solverFor(problem, n = 8, tol = tol), tol)

                assertEquals(
                    replay.stepCalls,
                    replay.steps,
                    "$label: хелпер сообщил ${replay.steps} шагов, а расчёт шага вызывался " +
                        "${replay.stepCalls} раз",
                )
                assertEquals(
                    replay.stepCalls,
                    reported.iterations,
                    "$label: solveBase сообщил ${reported.iterations} шагов, а фактически шаг " +
                        "выполнялся ${replay.stepCalls} раз",
                )
                checked++
                if (replay.stepCalls > 0) withSteps++
            }
        }
        assertTrue(checked > 0, "Тест выродился: ни одна конфигурация не проверена")
        assertTrue(
            withSteps > 0,
            "Тест выродился: во всех конфигурациях шагов оказалось ноль, поэтому сверка " +
                "счётчиков не различает семантики",
        )
    }

    /**
     * ЗАСТОЙ: малый шаг при большой невязке НЕ объявляется сходимостью.
     *
     * Штатные задачи A и B этого режима не дают — перебор 216 сочетаний
     * (задача × базис × сетка × допуск) дал НОЛЬ случаев. Поэтому режим проверяется
     * прямым вызовом [runNewtonIterations] с вырожденными `F` и шагом: невязка держится
     * равной `5.0`, а шаг — `1e-15`. Такая подстановка не подменяет физику задачи,
     * а изолирует ровно то управляющее решение, которое здесь проверяется.
     *
     * Именно этот случай отделяет обеспеченный кодом контракт от эмпирического
     * наблюдения: без сравнения с допуском `converged` был бы `true` при невязке `5.0`.
     */
    @Test
    fun negligibleStepWithLargeResidualIsNotConvergence() {
        val x = doubleArrayOf(1.0)
        val run = runNewtonIterations(
            x = x,
            maxSteps = 100,
            tolerance = 1e-8,
            residualAt = { doubleArrayOf(5.0) },
            stepAt = { _, _ -> doubleArrayOf(1e-15) },
        )
        assertFalse(
            run.converged,
            "Шаг пренебрежимо мал, но невязка 5.0 выше допуска 1e-8: это НЕ сходимость",
        )
        assertTrue(run.stalled, "Случай обязан быть помечен как ЗАСТОЙ, а не как исчерпание предела")
        assertEquals(1, run.performedSteps, "Застой распознаётся сразу после первого же шага")
        assertEquals(5.0, run.residual, "Невязка обязана быть сообщена фактическая")
        assertTrue(run.performedSteps < 100, "Предел итераций НЕ исчерпан — счёт прерван досрочно")

        // Контроль в обратную сторону: тот же выход по шагу, но невязка МАЛА, — успех.
        val converging = doubleArrayOf(10.0)
        val ok = runNewtonIterations(
            x = converging,
            maxSteps = 100,
            tolerance = 1e-8,
            residualAt = { current -> doubleArrayOf(abs(current[0])) },
            stepAt = { current, _ -> doubleArrayOf(-current[0]) },
        )
        assertTrue(ok.converged, "Малый шаг при НУЛЕВОЙ невязке обязан считаться сходимостью")
        assertFalse(ok.stalled, "Успешный исход застоем не является")
        assertEquals(0.0, ok.residual, "Невязка обязана быть измерена в возвращаемой точке")
    }
}
