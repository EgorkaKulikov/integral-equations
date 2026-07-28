package regression

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.ReferenceLinearAlgebra
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * REGRESSION-ТЕСТЫ обнаруженных и исправленных дефектов.
 *
 * Каждый тест назван по дефекту и снабжён описанием: в чём была ошибка, почему она
 * не проявлялась в существующих тестах и что именно проверяется теперь. Все эти
 * тесты ПАДАЮТ на коде до исправления.
 */
class DefectRegressionTest {

    /**
     * ДЕФЕКТ 1. Модельные задачи F2span/V2span/F1 объявляли производные ядра
     * (K_s, K_tt) и вторую производную решения нулевыми, хотя истинные значения
     * ненулевые. Семейство функционалов xi^<0> (де Бура--Фикса, r=0) читает вторые
     * производные — и молча получало неверную правую часть.
     *
     * Проявление: для задачи F2span точное решение u*(t)=t^2 принадлежит
     * span{1,t,t^2} = порождающей системе B, поэтому метод ОБЯЗАН воспроизводить
     * его с машинной точностью. До исправления ошибка составляла ~1e-2, то есть
     * была на двенадцать порядков больше должной.
     *
     * Ранее дефект не ловился: тесты для xi^<0> проверяли только конечность
     * результата и его биортогональность, но не точность на span-задаче.
     */
    @Test
    fun defect1_xi0OnSpanProblemIsExactForFredholm() {
        val problem = solvers.fredholm.ModelProblem.F2span
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = DeBoorFixFunctionals(basis, 0)
            val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
            val solver = solvers.fredholm.SecondKindSolver(
                basis, funcs, op, 1.0,
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            )
            val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
            assertTrue(
                error < 1e-10,
                "F2span (u*=t^2 в span порождающей системы B) должна решаться точно " +
                    "семейством xi^<0>, но E_h(n=$n)=$error (до исправления было ~1e-2)",
            )
        }
    }

    /** Тот же дефект для решателя Вольтерры (задача V2span). */
    @Test
    fun defect1_xi0OnSpanProblemIsExactForVolterra() {
        val problem = solvers.volterra.ModelProblem.V2span
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = DeBoorFixFunctionals(basis, 0)
            val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
            val solver = solvers.volterra.SecondKindSolver(
                basis, funcs, op, 1.0,
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            )
            val error = errorEh({ t -> problem.exact(t) }, solver.base().eval, grid)
            assertTrue(
                error < 1e-10,
                "V2span (u*=t^2 в span порождающей системы B) должна решаться точно " +
                    "семейством xi^<0>, но E_h(n=$n)=$error",
            )
        }
    }

    /**
     * ДЕФЕКТ 1 (проверка самих коэффициентов ядра). Прямая проверка того, что
     * производные ядра заданы согласованно с самим ядром: сравнение аналитических
     * K_s и K_tt с центральной конечной разностью.
     */
    @Test
    fun defect1_kernelDerivativesAreConsistentWithKernel() {
        val step = 1e-5
        val samplePoints = listOf(0.1 to 0.2, 0.5 to 0.3, 0.8 to 0.75)

        val fredholmKernels = listOf(
            "F2span" to solvers.fredholm.ModelProblem.F2span.kernel,
            "F1" to solvers.fredholm.ModelProblem.F1.kernel,
            "F2" to solvers.fredholm.ModelProblem.F2.kernel,
            "F2exp" to solvers.fredholm.ModelProblem.F2exp.kernel,
        )
        for ((name, kernel) in fredholmKernels) {
            for ((t, s) in samplePoints) {
                val numericKs = (kernel.k(t, s + step) - kernel.k(t, s - step)) / (2 * step)
                assertTrue(
                    abs(kernel.kS(t, s) - numericKs) < 1e-6,
                    "$name: K_s($t,$s)=${kernel.kS(t, s)} расходится с разностной ${numericKs}",
                )
                val numericKtt = (kernel.kT(t + step, s) - kernel.kT(t - step, s)) / (2 * step)
                assertTrue(
                    abs(kernel.kTT(t, s) - numericKtt) < 1e-6,
                    "$name: K_tt($t,$s)=${kernel.kTT(t, s)} расходится с разностной ${numericKtt}",
                )
            }
        }

        val volterraKernels = listOf(
            "V2span" to solvers.volterra.ModelProblem.V2span.kernel,
            "V2" to solvers.volterra.ModelProblem.V2.kernel,
            "V2exp" to solvers.volterra.ModelProblem.V2exp.kernel,
        )
        for ((name, kernel) in volterraKernels) {
            for ((t, s) in samplePoints) {
                val numericKs = (kernel.k(t, s + step) - kernel.k(t, s - step)) / (2 * step)
                assertTrue(
                    abs(kernel.kS(t, s) - numericKs) < 1e-6,
                    "$name: K_s($t,$s)=${kernel.kS(t, s)} расходится с разностной ${numericKs}",
                )
                val numericKtt = (kernel.kT(t + step, s) - kernel.kT(t - step, s)) / (2 * step)
                assertTrue(
                    abs(kernel.kTT(t, s) - numericKtt) < 1e-6,
                    "$name: K_tt($t,$s)=${kernel.kTT(t, s)} расходится с разностной ${numericKtt}",
                )
            }
        }
    }

    /**
     * ДЕФЕКТ 2. `FirstKindSolver` (Фредгольм) не передавал вторую производную
     * правой части во внутренний решатель II рода, из-за чего семейство xi^<0>
     * молча получало f'' = 0.
     *
     * Проверяется, что решатель I рода с xi^<0> теперь даёт конечный и осмысленный
     * результат, согласованный с решением через семейство theta (которое вторых
     * производных не использует и потому дефектом не затрагивалось).
     */
    @Test
    fun defect2_firstKindSolverPassesSecondDerivative() {
        val problem = solvers.fredholm.ModelProblem.F1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))

        val viaXi0 = solvers.fredholm.FirstKindSolver(problem, basis, DeBoorFixFunctionals(basis, 0), op)
        val errorXi0 = errorEh({ t -> problem.exact(t) }, viaXi0.base().eval, grid)
        assertTrue(errorXi0.isFinite(), "Решение I рода через xi^<0> должно быть конечным, получено $errorXi0")

        val viaTheta = solvers.fredholm.FirstKindSolver(problem, basis, ProjFunctionals(basis), op)
        val errorTheta = errorEh({ t -> problem.exact(t) }, viaTheta.base().eval, grid)
        assertTrue(
            errorXi0 < 100.0 * maxOf(errorTheta, 1e-12),
            "Ошибка xi^<0> ($errorXi0) не должна катастрофически превосходить theta ($errorTheta)",
        )
    }

    /**
     * ДЕФЕКТ 2 (Вольтерра). После редукции I->II рода вторая производная
     * редуцированного ядра аналитически недоступна, поэтому семейство xi^<0>
     * на этом пути должно ЯВНО отвергаться, а не давать молча неверный результат.
     */
    @Test
    fun defect2_volterraFirstKindRejectsSecondDerivativeFamilies() {
        val problem = solvers.volterra.ModelProblem.V1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(8))
        assertFailsWith<IllegalArgumentException>(
            "Решатель Вольтерры I рода обязан отвергать семейства, требующие второй производной",
        ) {
            solvers.volterra.FirstKindSolver(problem, basis, DeBoorFixFunctionals(basis, 0), op)
        }
    }

    /**
     * ДЕФЕКТ 3. `FirstKindSolver` (Фредгольм) не проверял положительность параметра
     * регуляризации: при alpha = 0 множитель c_L = -1/alpha обращался в бесконечность,
     * при alpha < 0 менялся смысл регуляризации — в обоих случаях без диагностики.
     */
    @Test
    fun defect3_firstKindSolverRejectsNonPositiveAlpha() {
        val problem = solvers.fredholm.ModelProblem.F1
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        for (badAlpha in listOf(0.0, -1e-10)) {
            assertFailsWith<IllegalArgumentException>("alpha=$badAlpha должен отвергаться") {
                solvers.fredholm.FirstKindSolver(problem, basis, funcs, op, badAlpha)
            }
        }
    }

    /**
     * ДЕФЕКТ 4. Порог вырожденности в эталонной линейной алгебре был абсолютным
     * (1e-300) и фактически проверял лишь строгий машинный ноль: практически
     * вырожденная матрица решалась молча и возвращала бессмысленный результат.
     *
     * Теперь порог относителен норме матрицы, поэтому вырожденность распознаётся
     * независимо от масштаба данных.
     */
    @Test
    fun defect4_singularityDetectedRegardlessOfScale() {
        // Точно вырожденная матрица (вторая строка кратна первой) в разных масштабах.
        for (scale in listOf(1.0, 1e6, 1e-6)) {
            val singular = arrayOf(
                doubleArrayOf(1.0 * scale, 2.0 * scale),
                doubleArrayOf(2.0 * scale, 4.0 * scale),
            )
            assertFailsWith<IllegalStateException>("Масштаб $scale: вырожденность должна распознаваться") {
                ReferenceLinearAlgebra.solve(singular, doubleArrayOf(1.0 * scale, 2.0 * scale))
            }
        }
        // Невырожденная матрица того же масштаба обязана решаться штатно.
        for (scale in listOf(1.0, 1e6, 1e-6)) {
            val regular = arrayOf(
                doubleArrayOf(1.0 * scale, 2.0 * scale),
                doubleArrayOf(3.0 * scale, 4.0 * scale),
            )
            val solution = ReferenceLinearAlgebra.solve(regular, doubleArrayOf(1.0 * scale, 1.0 * scale))
            assertTrue(solution.all { it.isFinite() }, "Масштаб $scale: решение должно быть конечным")
        }
    }

    /**
     * ДЕФЕКТ 5. Бэкенды линейной алгебры расходились на нечисловом входе:
     * multik/OpenBLAS бросал `IllegalStateException`, а эталонная реализация молча
     * возвращала вектор из NaN. Наблюдаемое поведение обязано совпадать.
     */
    @Test
    fun defect5_backendsAgreeOnNonFiniteInput() {
        val withNaN = arrayOf(
            doubleArrayOf(Double.NaN, 1.0),
            doubleArrayOf(1.0, 1.0),
        )
        assertFailsWith<IllegalStateException>(
            "Эталонная реализация обязана сигнализировать об ошибке так же, как нативный бэкенд",
        ) {
            ReferenceLinearAlgebra.solve(withNaN, doubleArrayOf(1.0, 1.0))
        }
    }

    /**
     * ДЕФЕКТ 6. Схема Nyström для уравнения Урысона стартовала с ТОЧНОГО решения
     * задачи (`problem.exact`), недоступного в реальном применении: метод был
     * невоспроизводим, а возможная расходимость из нейтрального приближения
     * маскировалась.
     *
     * Проверяется, что метод сходится из нейтрального начального приближения
     * (проекции постоянной функции) и достигает точности, сопоставимой с базовой
     * схемой. Тест не зависит от `problem.exact` как стартовой точки — только как
     * от эталона для измерения ошибки.
     */
    @Test
    fun defect6_urysonNystromConvergesFromNeutralStart() {
        for (problem in listOf(solvers.uryson.ModelProblem.A, solvers.uryson.ModelProblem.B)) {
            for (n in listOf(8, 16)) {
                val grid = Grid.uniform(n)
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                val funcs = solvers.uryson.ProjFunctionals(basis)
                val space = solvers.uryson.SplineSpace(basis, GaussLegendre(8))
                val op = solvers.uryson.UrysohnOperator(problem.kernel, grid, GaussLegendre(8))
                val solver = solvers.uryson.SecondKindSolver(problem, basis, funcs, space, op)

                val nystromSolution = solver.nystrom()
                val exact = { t: Double -> problem.exact(t) }
                val nystromError = solvers.uryson.errorEhEval(exact, nystromSolution.eval, grid)
                val baseError = solvers.uryson.errorEhEval(exact, solver.base().eval, grid)

                assertTrue(
                    nystromError.isFinite() && nystromError < 1.0,
                    "Задача ${problem.name}, n=$n: Nyström обязан сходиться из нейтрального " +
                        "начального приближения, получено E_h=$nystromError",
                )
                assertTrue(
                    nystromError < 100.0 * maxOf(baseError, 1e-14),
                    "Задача ${problem.name}, n=$n: точность Nyström ($nystromError) не должна " +
                        "катастрофически уступать базовой схеме ($baseError)",
                )
                assertTrue(
                    nystromSolution.iterations >= 1,
                    "Задача ${problem.name}, n=$n: счётчик итераций обязан быть осмысленным",
                )
            }
        }
    }
}
