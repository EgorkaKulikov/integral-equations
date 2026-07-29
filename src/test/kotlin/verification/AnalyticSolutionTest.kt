package verification

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import problems.analytic.AnalyticFredholmProblem
import problems.analytic.AnalyticVolterraProblem
import problems.analytic.analyticFredholmSolver
import problems.analytic.analyticVolterraSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * НЕЗАВИСИМАЯ ВЕРИФИКАЦИЯ по аналитически точным решениям (задание, п. 2.1).
 *
 * Все прочие проверки проекта замкнуты на собственную реализацию: характеризационный
 * тест фиксирует, что результат НЕ ИЗМЕНИЛСЯ, но не доказывает, что он ПРАВИЛЬНЫЙ.
 * Здесь эталон получен вне кода — решением конечных систем и взятием интегралов
 * ВРУЧНУЮ (см. выкладки в KDoc задач [AnalyticFredholmProblem],
 * [AnalyticVolterraProblem]).
 *
 * Тесты разделены на два уровня, и это разделение принципиально.
 *
 *  1. ТОЖДЕСТВО `u* - K u* = f` ([fredholmAnalyticIdentityHolds],
 *     [volterraAnalyticIdentityHolds]). Проверяет САМИ ВЫКЛАДКИ, а не решатель:
 *     если в ручном выводе допущена ошибка, она будет обнаружена здесь, а не станет
 *     молча «эталоном». Используется высокоточная квадратура [PRECISE_QUADRATURE_ORDER]
 *     узлов на подынтервал с мелким разбиением — она не участвует в работе схем и
 *     служит только независимой сверкой.
 *
 *  2. СХОДИМОСТЬ схем к аналитическому решению. Проверяет уже сам решатель против
 *     эталона, доверие к которому обеспечено уровнем 1.
 *
 * Ключевое отличие от `problems.fredholm.FredholmProblem`: там правая часть строится
 * численно тем же оператором, который затем проверяется, из-за чего погрешность
 * квадратуры входит в обе части сравнения и частично сокращается. Здесь `f` выписана
 * аналитически, поэтому такой взаимной компенсации нет.
 */
class AnalyticSolutionTest {

    private companion object {
        /**
         * Порядок квадратуры для НЕЗАВИСИМОЙ проверки тождества. Он вдвое выше
         * рабочего (8) и применяется на мелком разбиении: цель — чтобы погрешность
         * самой проверки была заведомо ниже проверяемого допуска.
         */
        const val PRECISE_QUADRATURE_ORDER = 16

        /** Число подынтервалов составного разбиения при проверке тождества. */
        const val PRECISE_SUBDIVISIONS = 32

        /**
         * Допуск проверки тождества `u* - K u* = f`. Величина отражает предел
         * точности составной квадратуры высокого порядка на гладких данных.
         */
        const val IDENTITY_TOLERANCE = 1e-12

        /** Число точек, в которых проверяется тождество. */
        const val IDENTITY_SAMPLE_COUNT = 21

        /** Шаг центральной разности при сверке аналитических производных `f'`. */
        const val DERIVATIVE_STEP = 1e-5

        /** Допуск сверки `f'` с центральной разностью (ошибка разности ~ h^2). */
        const val FIRST_DERIVATIVE_TOLERANCE = 1e-6

        /** Шаг центральной разности при сверке `f''` (второй порядок требует большего шага). */
        const val SECOND_DERIVATIVE_STEP = 1e-4

        /** Допуск сверки `f''` со второй центральной разностью. */
        const val SECOND_DERIVATIVE_TOLERANCE = 1e-4

        /**
         * Допуск для задачи, решение которой лежит в `span{1, t, t^2}`: метод обязан
         * воспроизводить его практически точно.
         */
        const val SPAN_EXACTNESS_TOLERANCE = 1e-10

        /**
         * Верхняя граница погрешности на самой мелкой сетке набора. Значение выбрано
         * с большим запасом относительно наблюдаемых величин (порядка 1e-6 и ниже):
         * тест обязан реагировать на поломку схемы, а не на колебания константы.
         */
        const val MAX_FINE_GRID_ERROR = 1e-3

        /** Сетки для проверки сходимости схем к аналитическому решению. */
        val GRID_SIZES = listOf(8, 16, 32)
    }

    /** Квадратура для независимой проверки тождества (в схемах НЕ используется). */
    private val preciseQuadrature = GaussLegendre(PRECISE_QUADRATURE_ORDER)

    /** Высокоточный интеграл по `[lo, hi]` составной квадратурой. */
    private fun preciseIntegral(lo: Double, hi: Double, integrand: (Double) -> Double): Double {
        if (hi <= lo) return 0.0
        val breakpoints = DoubleArray(PRECISE_SUBDIVISIONS + 1) {
            lo + (hi - lo) * it / PRECISE_SUBDIVISIONS
        }
        return preciseQuadrature.integrate(breakpoints, integrand)
    }

    /** Равномерная выборка точек отрезка `[0, 1]`. */
    private fun samplePoints(): List<Double> =
        (0 until IDENTITY_SAMPLE_COUNT).map { it.toDouble() / (IDENTITY_SAMPLE_COUNT - 1) }

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        else -> error("Неизвестное семейство функционалов: '$name'")
    }

    private fun reportIfAny(failures: List<String>) {
        assertTrue(
            failures.isEmpty(),
            "Обнаружено ${failures.size} расхождений с аналитическим эталоном:\n" +
                failures.joinToString("\n").take(6000),
        )
    }

    // ------------------------------------------------------------------------
    // Уровень 1: проверка САМИХ ВЫКЛАДОК (не решателя)
    // ------------------------------------------------------------------------

    /**
     * Тождество `u*(t) - \int_0^1 K(t,s) u*(s) ds = f(t)` для всех аналитических
     * задач Фредгольма.
     *
     * Это страховка от ошибки в ручном выводе: и `u*`, и `f` выписаны на бумаге,
     * и если хотя бы одна из выкладок неверна, тождество нарушится. Проверка не
     * обращается ни к базису, ни к функционалам, ни к решателю — только к ядру и
     * независимой высокоточной квадратуре.
     */
    @Test
    fun fredholmAnalyticIdentityHolds() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticFredholmProblem.ALL) {
            var worstDeviation = 0.0
            var worstPoint = Double.NaN
            for (t in samplePoints()) {
                val image = preciseIntegral(0.0, 1.0) { s -> problem.kernel.k(t, s) * problem.exact(s) }
                val residual = problem.exact(t) - image - problem.rhs(t)
                if (abs(residual) > worstDeviation) {
                    worstDeviation = abs(residual)
                    worstPoint = t
                }
            }
            if (worstDeviation > IDENTITY_TOLERANCE) {
                failures += "${problem.name}: |u* - K u* - f| = $worstDeviation при t=$worstPoint " +
                    "(допуск $IDENTITY_TOLERANCE). Вывод: ${problem.derivation}"
            }
        }
        reportIfAny(failures)
    }

    /**
     * Тождество `u*(t) - \int_0^t K(t,s) u*(s) ds = f(t)` для всех аналитических
     * задач Вольтерры.
     *
     * Для задач с ядром свёртки это независимая проверка решения, полученного
     * преобразованием Лапласа: образ считается прямым интегрированием, а не через
     * операционное исчисление.
     */
    @Test
    fun volterraAnalyticIdentityHolds() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticVolterraProblem.ALL) {
            var worstDeviation = 0.0
            var worstPoint = Double.NaN
            for (t in samplePoints()) {
                val image = preciseIntegral(0.0, t) { s -> problem.kernel.k(t, s) * problem.exact(s) }
                val residual = problem.exact(t) - image - problem.rhs(t)
                if (abs(residual) > worstDeviation) {
                    worstDeviation = abs(residual)
                    worstPoint = t
                }
            }
            if (worstDeviation > IDENTITY_TOLERANCE) {
                failures += "${problem.name}: |u* - V u* - f| = $worstDeviation при t=$worstPoint " +
                    "(допуск $IDENTITY_TOLERANCE). Вывод: ${problem.derivation}"
            }
        }
        reportIfAny(failures)
    }

    /**
     * Согласованность аналитических производных правой части `f'` и `f''` с самой `f`.
     *
     * Производные выписаны вручную и используются семействами функционалов
     * де Бура–Фикса. Ошибка в них не проявилась бы в семействах `theta`/`mu`
     * (они производных не читают) и осталась бы незамеченной. Сверка выполняется
     * конечными разностями — независимо от какой-либо части проекта.
     */
    @Test
    fun analyticRightHandSideDerivativesAreConsistent() {
        val failures = mutableListOf<String>()

        fun check(
            name: String,
            rhs: (Double) -> Double,
            rhsDeriv: (Double) -> Double,
            rhsDeriv2: (Double) -> Double,
        ) {
            for (i in 1 until IDENTITY_SAMPLE_COUNT - 1) {
                val t = i.toDouble() / (IDENTITY_SAMPLE_COUNT - 1)
                val numericFirst =
                    (rhs(t + DERIVATIVE_STEP) - rhs(t - DERIVATIVE_STEP)) / (2 * DERIVATIVE_STEP)
                if (abs(rhsDeriv(t) - numericFirst) > FIRST_DERIVATIVE_TOLERANCE) {
                    failures += "$name: f'($t)=${rhsDeriv(t)} расходится с разностной $numericFirst"
                }
                val h = SECOND_DERIVATIVE_STEP
                val numericSecond = (rhs(t + h) - 2 * rhs(t) + rhs(t - h)) / (h * h)
                if (abs(rhsDeriv2(t) - numericSecond) > SECOND_DERIVATIVE_TOLERANCE) {
                    failures += "$name: f''($t)=${rhsDeriv2(t)} расходится с разностной $numericSecond"
                }
            }
        }

        for (problem in AnalyticFredholmProblem.ALL) {
            check(problem.name, problem.rhs, problem.rhsDeriv, problem.rhsDeriv2)
        }
        for (problem in AnalyticVolterraProblem.ALL) {
            check(problem.name, problem.rhs, problem.rhsDeriv, problem.rhsDeriv2)
        }
        reportIfAny(failures)
    }

    // ------------------------------------------------------------------------
    // Уровень 2: сходимость схем к аналитическому решению
    // ------------------------------------------------------------------------

    /**
     * Задача из постановки задания (`K = t*s`, `f = t`, точное решение `u* = (3/2)t`)
     * решается ТОЧНО: линейная функция лежит в `span{1, t, t^2}` полиномиальной
     * порождающей системы.
     *
     * Проверка охватывает всю цепочку — базис, функционалы, квадратуру, сборку
     * матрицы и решение СЛАУ — против результата, полученного вручную на бумаге.
     */
    @Test
    fun separableRank1ExampleIsReproducedExactly() {
        val problem = AnalyticFredholmProblem.SEPARABLE_RANK1_LINEAR
        val failures = mutableListOf<String>()
        for (familyName in listOf("theta", "mu")) {
            for (n in listOf(8, 16)) {
                val grid = Grid.uniform(n)
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                val funcs = family(familyName, basis)
                val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                val solver = analyticFredholmSolver(problem, basis, funcs, op)
                val error = errorEh(problem.exact, solver.base().eval, grid)
                if (!(error < SPAN_EXACTNESS_TOLERANCE)) {
                    failures += "${problem.name}/$familyName/n=$n: E_h=$error должно быть " +
                        "ниже $SPAN_EXACTNESS_TOLERANCE (u*=(3/2)t лежит в span порождающей системы B)"
                }
            }
        }
        reportIfAny(failures)
    }

    /**
     * Схемы `base`, `sloan`, `kulkarni` сходятся к АНАЛИТИЧЕСКОМУ решению задач
     * Фредгольма с вырожденным ядром и задачи MMS.
     *
     * Проверяются два содержательных условия: погрешность убывает при измельчении
     * сетки и на самой мелкой сетке достигает разумной абсолютной величины.
     * Измерение порядка сходимости — предмет отдельного теста `ConvergenceOrderTest`.
     *
     * ВАЖНО о семействе `mu`. Для квазиинтерполянтов схема Кулкарни реализована
     * ПРОСТОЙ ИТЕРАЦИЕЙ (`kulkarniQuasi`), которая требует `rho(K) < 1`. На задачах
     * [AnalyticFredholmProblem.SEPARABLE_RANK2] (`rho ~ 1.27`) и
     * [AnalyticFredholmProblem.SEPARABLE_RANK3] (`rho ~ 1.41`) она расходится. Это
     * ограничение МЕТОДА, а не дефект реализации и не свойство задач: сами задачи
     * однозначно разрешимы, и прямые схемы решают их штатно. Поэтому схема Кулкарни
     * для `mu` проверяется только на задачах с `rho < 1`, а расхождение на остальных
     * зафиксировано отдельным тестом [quasiKulkarniDivergesWhenSpectralRadiusExceedsOne]
     * — это осознанное ограничение, а не подгонка допуска.
     */
    @Test
    fun fredholmSchemesConvergeToAnalyticSolution() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticFredholmProblem.CONVERGENT) {
            for (familyName in listOf("theta", "mu")) {
                // Проектор theta использует редукцию Кулкарни с ПРЯМЫМ решением СЛАУ
                // и пригоден всегда; квазиинтерполянт mu — простую итерацию.
                val iterativeKulkarni = familyName == "mu"
                val includeKulkarni = !iterativeKulkarni || problem.supportsFixedPointSchemes
                val errors = linkedMapOf<String, MutableList<Double>>(
                    "base" to mutableListOf(),
                    "sloan" to mutableListOf(),
                )
                if (includeKulkarni) errors["kulkarni"] = mutableListOf()
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = analyticFredholmSolver(problem, basis, funcs, op)
                    errors.getValue("base") += errorEh(problem.exact, solver.base().eval, grid)
                    errors.getValue("sloan") += errorEh(problem.exact, solver.sloan().eval, grid)
                    if (includeKulkarni) {
                        errors.getValue("kulkarni") += errorEh(problem.exact, solver.kulkarni().eval, grid)
                    }
                }
                collectConvergenceFailures("F.${problem.name}.$familyName", errors, failures)
            }
        }
        reportIfAny(failures)
    }

    /**
     * ДОКУМЕНТИРОВАННОЕ ОГРАНИЧЕНИЕ: схема Кулкарни для квазиинтерполянтов
     * расходится при `rho(K) > 1`, тогда как прямые схемы решают те же задачи штатно.
     *
     * Найдено при разработке настоящего теста: на [AnalyticFredholmProblem.SEPARABLE_RANK2]
     * и [AnalyticFredholmProblem.SEPARABLE_RANK3] итерация даёт величины порядка 1e21
     * и 1e29 соответственно (не зависящие от `n` — признак расходимости итерации,
     * а не ошибки аппроксимации, которая убывала бы).
     *
     * Тест закрепляет три факта: (а) прямые схемы работают — значит, задача
     * корректна и эталон верен; (б) итерационная по умолчанию СИГНАЛИЗИРУЕТ
     * об ошибке исключением (единый контракт сходимости); (в) в режиме
     * `throwOnDivergence = false` та же схема возвращает результат, помеченный
     * `converged = false`, и его погрешность действительно катастрофическая.
     *
     * ИСТОРИЯ: изначально тест проверял, что `kulkarni()` МОЛЧА возвращает
     * расходящийся результат с `E_h ~ 1e21`. После введения единого контракта
     * такое поведение стало недопустимым, и тест обновлён осознанно: именно
     * такого падения и требовала задача 4 — молчаливый возврат больше невозможен.
     */
    @Test
    fun quasiKulkarniDivergesWhenSpectralRadiusExceedsOne() {
        val failures = mutableListOf<String>()
        val divergent = AnalyticFredholmProblem.ALL.filterNot { it.supportsFixedPointSchemes }
        assertTrue(
            divergent.isNotEmpty(),
            "Набор задач обязан содержать хотя бы одну с rho > 1 для этой проверки",
        )
        for (problem in divergent) {
            val grid = Grid.uniform(8)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))

            // (а) Прямая схема на проекторе theta — задача решается штатно.
            val projectorSolver = analyticFredholmSolver(problem, basis, ProjFunctionals(basis), op)
            val directError = errorEh(problem.exact, projectorSolver.base().eval, grid)
            if (!(directError < MAX_FINE_GRID_ERROR)) {
                failures += "${problem.name}: прямая схема (theta/base) обязана решать задачу " +
                    "с rho=${problem.spectralRadius}, но E_h=$directError"
            }

            // (б) По умолчанию расходимость обязана быть явной ошибкой.
            val strictSolver = analyticFredholmSolver(problem, basis, AveragingFunctionals(basis), op)
            assertFailsWith<IllegalStateException>(
                "${problem.name}: при rho=${problem.spectralRadius} > 1 схема обязана " +
                    "сообщить о расходимости исключением, а не возвращать результат",
            ) { strictSolver.kulkarni() }

            // (в) В явно разрешённом режиме результат доступен, но помечен как несошедшийся.
            val lenientSolver = analyticFredholmSolver(
                problem, basis, AveragingFunctionals(basis), op, throwOnDivergence = false,
            )
            val solution = lenientSolver.kulkarni()
            if (solution.converged) {
                failures += "${problem.name}: результат при rho=${problem.spectralRadius} > 1 " +
                    "обязан быть помечен converged = false"
            }
            val iterativeError = errorEh(problem.exact, solution.eval, grid)
            if (iterativeError < 1.0) {
                failures += "${problem.name}: ожидалась расходимость kulkarniQuasi при " +
                    "rho=${problem.spectralRadius} > 1, но получено E_h=$iterativeError. " +
                    "Если схема улучшена — обновите ожидание осознанно"
            }
        }
        reportIfAny(failures)
    }

    /**
     * Схемы `base`, `sloan`, `kulkarni` сходятся к АНАЛИТИЧЕСКОМУ решению задач
     * Вольтерры: трёх с ядром свёртки (решены преобразованием Лапласа) и одной MMS.
     */
    @Test
    fun volterraSchemesConvergeToAnalyticSolution() {
        val failures = mutableListOf<String>()
        for (problem in AnalyticVolterraProblem.ALL) {
            for (familyName in listOf("theta", "mu")) {
                val errors = linkedMapOf<String, MutableList<Double>>(
                    "base" to mutableListOf(),
                    "sloan" to mutableListOf(),
                    "kulkarni" to mutableListOf(),
                )
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = analyticVolterraSolver(problem, basis, funcs, op)
                    errors.getValue("base") += errorEh(problem.exact, solver.base().eval, grid)
                    errors.getValue("sloan") += errorEh(problem.exact, solver.sloan().eval, grid)
                    errors.getValue("kulkarni") += errorEh(problem.exact, solver.kulkarni().eval, grid)
                }
                collectConvergenceFailures("V.${problem.name}.$familyName", errors, failures)
            }
        }
        reportIfAny(failures)
    }

    /**
     * Общая проверка набора погрешностей: монотонное убывание и достижение разумной
     * абсолютной точности на самой мелкой сетке.
     *
     * Значения ниже [SPAN_EXACTNESS_TOLERANCE] от проверки убывания освобождаются:
     * там доминирует шум округления, и требовать монотонности бессмысленно.
     */
    private fun collectConvergenceFailures(
        tag: String,
        errors: Map<String, List<Double>>,
        failures: MutableList<String>,
    ) {
        for ((scheme, values) in errors) {
            val formatted = values.joinToString(", ") { "%.3e".format(it) }
            val detail = "$tag.$scheme: E_h(n=${GRID_SIZES.joinToString(",")}) = [$formatted]"
            if (values.any { !it.isFinite() }) {
                failures += "$detail — присутствует нечисловое значение"
                continue
            }
            val fine = values.last()
            if (fine > MAX_FINE_GRID_ERROR) {
                failures += "$detail — погрешность на мелкой сетке превышает $MAX_FINE_GRID_ERROR"
            }
            // Убывание проверяем только пока не достигнут уровень округления.
            for (i in 0 until values.size - 1) {
                if (values[i] < SPAN_EXACTNESS_TOLERANCE) break
                if (values[i + 1] >= values[i]) {
                    failures += "$detail — погрешность не убывает на шаге ${GRID_SIZES[i]}->" +
                        "${GRID_SIZES[i + 1]}"
                    break
                }
            }
        }
    }
}
