package solvers.uryson

import numerics.CheckResult
import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyRightHandSide
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Характеризационные тесты решателей нелинейного уравнения Урысона и вспомогательных типов.
 *
 * ВАЖНО: числовые пороги здесь не являются аналитической истиной. Эталонные значения
 * зафиксированы однократным запуском текущей реализации и служат сетью безопасности
 * от регрессий: они ловят появление NaN, расходимость и «взрыв» решения, но не
 * подтверждают теоретические порядки сходимости.
 */
class UrysonCoverageTest {

    private val quad = GaussLegendre(8)

    private fun finite(x: Double) = !x.isNaN() && !x.isInfinite()

    private fun solverFor(
        problem: UrysonProblem,
        system: GeneratingSystem,
        n: Int,
    ): SecondKindSolver {
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(problem.kernel, grid, quad)
        return secondKindSolver(problem, basis, funcs, space, op)
    }

    /** Проверка флага `ok` у [CheckResult]: превышение порога означает провал. */
    @Test
    fun checkResultFlags() {
        assertTrue(CheckResult("x", 1e-12, 1e-10, true).ok)
        assertFalse(CheckResult("y", 1.0, 1e-10, false).ok)
    }

    /** Поля и вычислитель [FirstKindSolution] — простой носитель данных. */
    @Test
    fun firstKindSolutionFields() {
        val solution = FirstKindSolution(doubleArrayOf(1.0, 2.0), { t -> t * t }, 1e-3, 1e-4, 0.5)
        assertTrue(solution.coeffs.size == 2)
        assertTrue(solution.alpha == 1e-3 && solution.resid == 1e-4 && solution.omega == 0.5)
        assertTrue(abs(solution.eval(3.0) - 9.0) < 1e-12)
    }

    /** Поля [SolutionFunc]: вычислитель и счётчик итераций. */
    @Test
    fun solutionFuncFields() {
        val solution = SolutionFunc({ t -> t + 1 }, 7)
        assertTrue(abs(solution.eval(1.0) - 2.0) < 1e-12 && solution.iterations == 7)
    }

    /**
     * Свойства [SplineSpace]: сумма весов равна длине отрезка, матрица Грама
     * симметрична, квадратичная форма стабилизатора неотрицательна.
     */
    @Test
    fun splineSpaceWeightsGramAndRegularizer() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val space = SplineSpace(basis, quad)
        assertTrue(abs(space.weightsSum() - 1.0) < 1e-12)
        assertTrue(space.weights.sum() > 0 && space.wInt.all { finite(it) })
        for (i in 0 until space.dim) {
            for (j in 0 until space.dim) {
                assertTrue(abs(space.gramR[i][j] - space.gramR[j][i]) < 1e-12)
            }
        }
        assertTrue(abs(space.omegaReg(DoubleArray(space.dim))) < 1e-15)
        assertTrue(space.omegaReg(DoubleArray(space.dim) { 1.0 }) > 0.0)
    }

    /**
     * Согласованность интерфейсов [UrysohnOperator] и правой части для всех четырёх
     * модельных задач: вычисление по предвычисленным узлам совпадает с вычислением
     * через замыкание.
     */
    @Test
    fun operatorApisAgreeForAllProblems() {
        val grid = Grid.uniform(8)
        val problems = listOf(
            UrysonProblem.A, UrysonProblem.B, UrysonProblem.C, UrysonProblem.D,
        )
        for (problem in problems) {
            val op = UrysohnOperator(problem.kernel, grid, quad)
            val t = 0.4
            val xNodes = DoubleArray(op.gNode.size) { problem.exact(op.gNode[it]) }
            val viaNodes = op.applyNodes(t, xNodes)
            val viaClosure = op.apply(t) { s -> problem.exact(s) }
            assertTrue(
                abs(viaNodes - viaClosure) < 1e-9,
                "${problem.name}: applyNodes и apply должны совпадать",
            )
            assertTrue(finite(op.frechet(t, { s -> problem.exact(s) }, { 1.0 })))
            assertTrue(finite(problem.rhsExact(t, op)))
            assertTrue(finite(problem.kernel.dkdu(t, 0.3, 1.0)))
        }
    }

    /** Вектор `Xi` и якобиан `B` имеют правильные размеры и конечны. */
    @Test
    fun collocationCoreProducesFiniteXiAndJacobian() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.A.exact(t) })
        val xi = core.xiVector(coefficients)
        val jacobian = core.bMatrix(coefficients)
        assertTrue(xi.size == grid.n + 2 && xi.all { finite(it) })
        assertTrue(
            jacobian.size == grid.n + 2 &&
                jacobian.all { row -> row.size == grid.n + 2 && row.all { finite(it) } },
        )
        assertTrue(core.uAtSupport(coefficients).all { finite(it) })
    }

    /** Все четыре схемы второго рода дают конечный результат на сжимающей задаче A. */
    @Test
    fun problemAllSchemesAreFinite() {
        val solver = solverFor(UrysonProblem.A, GeneratingSystem.B, 8)
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val solutions = listOf(solver.base(), solver.sloan(), solver.kulkarni(), solver.nystrom())
        for (solution in solutions) {
            val error = errorEh(exact, solution.eval, solver.grid)
            assertTrue(finite(error) && error < 1e-2, "Задача A: E_h = $error")
        }
    }

    /**
     * Задача B с кубическим ядром при `lambda = 1` НЕсжимающая: простая итерация
     * расходится, поэтому тест проверяет именно ньютоновский путь решателя.
     */
    @Test
    fun problemBNonContractiveSchemesConverge() {
        val solver = solverFor(UrysonProblem.B, GeneratingSystem.H, 8)
        val exact = { t: Double -> UrysonProblem.B.exact(t) }
        for (solution in listOf(solver.base(), solver.sloan(), solver.nystrom())) {
            val error = errorEh(exact, solution.eval, solver.grid)
            assertTrue(finite(error) && error < 1e-1, "Задача B: E_h = $error")
        }
    }

    /** Базовая схема сходится: погрешность убывает при переходе n = 8 -> 16. */
    @Test
    fun problemAConverges() {
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val errorN8 = errorEh(exact, solverFor(UrysonProblem.A, GeneratingSystem.B, 8).base().eval, Grid.uniform(8))
        val errorN16 = errorEh(exact, solverFor(UrysonProblem.A, GeneratingSystem.B, 16).base().eval, Grid.uniform(16))
        assertTrue(errorN16 < errorN8, "Нет сходимости: E_8 = $errorN8, E_16 = $errorN16")
    }

    /**
     * Регуляризованный решатель на задаче C: ветви генератора шума при нулевом и
     * ненулевом уровне, шаг Гаусса–Ньютона, вычисление невязки и путь Морозова без шума.
     */
    @Test
    fun firstKindSolverOnProblemCWithoutNoise() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)

        val thetaExact = noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 0.0, 1L)
        assertTrue(thetaExact.size == grid.n + 2 && thetaExact.all { finite(it) })

        val thetaNoisy = noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 1e-3, 42L)
        assertTrue(thetaNoisy.all { finite(it) })

        val start = funcs.projectorCoeffs({ 1.0 })
        val coefficients = solver.solveFixedAlpha(thetaExact, 1e-4, start)
        assertTrue(coefficients.all { finite(it) })
        assertTrue(finite(solver.residual(coefficients, thetaExact)))

        val solution = solver.solveMorozov(thetaExact, 0.0)
        assertTrue(solution.alpha > 0 && finite(solution.resid) && finite(solution.eval(0.5)))
    }

    /**
     * Путь Морозова с шумом на задаче D с кубическим ядром: `dK/du(t,s,0) = 0`,
     * поэтому проверяется ветвь перезапуска с ненулевого начального приближения.
     */
    @Test
    fun firstKindSolverOnProblemDWithNoise() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.D.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta = noisyThetaCoefficients(UrysonProblem.D, solver, op, grid, quad, 1e-2, 7L)
        val solution = solver.solveMorozov(thetaFDelta, 1e-2)
        assertTrue(solution.alpha > 0 && finite(solution.resid) && finite(solution.omega))
        assertTrue(finite(solution.eval(0.3)))
    }

    /**
     * Генератор шума детерминирован: при одном зерне результат воспроизводится
     * побитово, при разных — различается.
     */
    @Test
    fun noiseGeneratorIsReproducible() {
        val grid = Grid.uniform(8)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val exactRhs = { t: Double -> UrysonProblem.C.rhsExact(t, op) }
        val first = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 123L)
        val second = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 123L)
        val other = noisyRightHandSide(exactRhs, grid, quad, 1e-2, 456L)
        var differsFromOther = false
        for (i in 0..20) {
            val t = i / 20.0
            assertTrue(first(t) == second(t), "Одно зерно должно давать идентичный шум в точке $t")
            if (abs(first(t) - other(t)) > 1e-15) differsFromOther = true
        }
        assertTrue(differsFromOther, "Разные зёрна должны давать различный шум")
        // При нулевом уровне шума возвращается исходная правая часть.
        val noiseless = noisyRightHandSide(exactRhs, grid, quad, 0.0, 1L)
        assertTrue(noiseless(0.5) == exactRhs(0.5))
    }
}
