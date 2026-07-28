package healthchecks

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.LinearAlgebra
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import problems.uryson.UrysonProblem
import problems.uryson.firstKindSolver
import problems.uryson.noisyThetaCoefficients
import problems.uryson.secondKindSolver
import solvers.uryson.CollocationCore
import solvers.uryson.SplineSpace
import solvers.uryson.UrysohnOperator
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Health-checks, СПЕЦИФИЧНЫЕ для решателя нелинейного уравнения Урысона.
 *
 * Общие проверки вычислительного ядра (сплайны, функционалы, квадратура) вынесены
 * в [SplineCoreHealthCheckTest] и здесь не дублируются.
 *
 * Ранее эти проверки были оформлены как объект `HealthChecks` в производственном
 * коде и возвращали числовую «измеренную величину», сравниваемую с порогом. Часть
 * из них при этом была замаскированными БИНАРНЫМИ проверками: при невыполнении
 * условия величина искусственно приравнивалась единице, чтобы превысить порог.
 * Здесь такие проверки записаны явными утверждениями.
 */
class UrysonHealthCheckTest {

    private companion object {
        /** Порог для тождеств, выполняющихся точно (с точностью до округления). */
        const val EXACT_IDENTITY_TOLERANCE = 1e-10

        /** Порог для суммы весов: величина вычисляется без интегрирования, ошибка минимальна. */
        const val WEIGHTS_SUM_TOLERANCE = 1e-12

        /** Верхняя граница погрешности базовой схемы на грубой сетке (защита от расходимости). */
        const val MAX_COARSE_GRID_ERROR = 1e-2

        /** Допустимое превышение невязки при сгущении сетки (невязка не обязана падать строго монотонно). */
        const val RESIDUAL_GROWTH_TOLERANCE = 1.1

        /** Параметр регуляризации для проверок, где важен сам факт положительной определённости. */
        const val PROBE_ALPHA = 1e-3
    }

    private val quad = GaussLegendre(8)

    /**
     * Матрица стабилизатора `R_h` симметрична, положительно определена и полосная.
     *
     * Эти три свойства обеспечивают разрешимость системы Гаусса–Ньютона: без
     * положительной определённости регуляризованная задача перестаёт быть выпуклой.
     * Полосность (`|i-j| <= 2`) следует из того, что носители далёких сплайнов не
     * пересекаются, и её нарушение означало бы ошибку в вычислении общего носителя.
     */
    @Test
    fun gramMatrixIsSymmetricPositiveDefiniteAndBanded() {
        for (grid in listOf(Grid.uniform(8), Grid.quasiUniform(8))) {
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                val basis = MinimalSplineBasis(system, grid)
                val space = SplineSpace(basis, quad)
                val gram = space.gramR

                val asymmetry = LinearAlgebra.maxAsymmetry(gram)
                assertTrue(
                    asymmetry < EXACT_IDENTITY_TOLERANCE,
                    "Базис ${system.name}: матрица R_h должна быть симметричной, " +
                        "max|R - R^T| = $asymmetry",
                )
                assertNotNull(
                    LinearAlgebra.cholesky(gram),
                    "Базис ${system.name}: матрица R_h должна быть положительно определена " +
                        "(разложение Холецкого не существует)",
                )
                for (i in 0 until space.dim) {
                    for (j in 0 until space.dim) {
                        if (abs(i - j) > 2) {
                            assertTrue(
                                abs(gram[i][j]) < EXACT_IDENTITY_TOLERANCE,
                                "Базис ${system.name}: элемент R_h[$i][$j] вне полосы должен быть " +
                                    "нулевым, получено ${gram[i][j]}",
                            )
                        }
                    }
                }
            }
        }
    }

    /**
     * Сумма весов дискретной нормы равна длине отрезка.
     *
     * Веса `w_j = (x_{j+3} - x_j)/3` в сумме дают `b - a`, поскольку каждый интервал
     * входит в носители ровно трёх сплайнов. Нарушение означало бы, что дискретная
     * норма не согласована с `L^2`, и оценка невязки в принципе Морозова была бы
     * систематически смещена.
     */
    @Test
    fun weightsSumEqualsIntervalLength() {
        for (grid in listOf(Grid.uniform(8), Grid.quasiUniform(8))) {
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val space = SplineSpace(basis, quad)
            val deviation = abs(space.weightsSum() - (grid.b - grid.a))
            assertTrue(
                deviation < WEIGHTS_SUM_TOLERANCE,
                "Сумма весов должна равняться длине отрезка ${grid.b - grid.a}, " +
                    "отклонение $deviation",
            )
        }
    }

    /**
     * Матрица системы Гаусса–Ньютона `B^T W_h B + alpha R_h` симметрична и положительно
     * определена при любом `alpha > 0`.
     *
     * Слагаемое `B^T W_h B` неотрицательно определено, но может быть вырожденным
     * (задача первого рода некорректна); именно добавление `alpha R_h` делает систему
     * разрешимой. Проверка подтверждает, что регуляризация действительно выполняет
     * свою роль.
     */
    @Test
    fun gaussNewtonMatrixIsPositiveDefinite() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val core = CollocationCore(basis, funcs, op)
        val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.C.exact(t) })
        val b = core.bMatrix(coefficients)
        val normalMatrix = LinearAlgebra.atWa(b, space.weights)
        val regularized = LinearAlgebra.addScaled(normalMatrix, space.gramR, PROBE_ALPHA)

        val asymmetry = LinearAlgebra.maxAsymmetry(regularized)
        assertTrue(
            asymmetry < EXACT_IDENTITY_TOLERANCE,
            "Матрица B^T W B + alpha R должна быть симметричной, max|A - A^T| = $asymmetry",
        )
        assertNotNull(
            LinearAlgebra.cholesky(regularized),
            "Матрица B^T W B + alpha R при alpha = $PROBE_ALPHA должна быть положительно определена",
        )
    }

    /**
     * Согласованность правой части с оператором: подстановка точного решения в
     * дискретную невязку даёт малую величину, которая убывает при сгущении сетки.
     *
     * Проверка ловит рассогласование между способом построения правой части
     * `f = U x*` и способом её дискретизации функционалами `theta_j`.
     */
    @Test
    fun rightHandSideIsConsistentWithOperator() {
        fun residualOn(n: Int): Double {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
            val core = CollocationCore(basis, funcs, op)
            val coefficients = funcs.projectorCoeffs({ t -> UrysonProblem.C.exact(t) })
            val xi = core.xiVector(coefficients)
            val thetaF = firstKindSolver(basis, funcs, space, op)
                .thetaOf { t -> UrysonProblem.C.rhsExact(t, op) }
            var sum = 0.0
            for (j in 0 until grid.n + 2) {
                val d = xi[j] - thetaF[j]
                sum += space.weights[j] * d * d
            }
            return Math.sqrt(sum)
        }

        val coarse = residualOn(8)
        val fine = residualOn(16)
        assertTrue(
            fine <= coarse * RESIDUAL_GROWTH_TOLERANCE,
            "Невязка на точном решении должна убывать при сгущении сетки: " +
                "res(n=8) = $coarse, res(n=16) = $fine",
        )
        assertTrue(
            fine.isFinite() && fine < MAX_COARSE_GRID_ERROR,
            "Невязка на точном решении должна быть малой, получено $fine",
        )
    }

    /**
     * Базовая схема второго рода сходится на модельной задаче A.
     *
     * Ранее эта проверка возвращала «измеренную величину», равную погрешности при
     * выполнении условия и единице при невыполнении, — то есть была бинарной под
     * видом числовой. Здесь условие записано прямо.
     */
    @Test
    fun secondKindSchemeConverges() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val solver = secondKindSolver(UrysonProblem.A, basis, funcs, space, op)
        val error = errorEh({ t -> UrysonProblem.A.exact(t) }, solver.base().eval, grid)
        assertTrue(
            error.isFinite() && error < MAX_COARSE_GRID_ERROR,
            "Базовая схема на задаче A должна давать малую погрешность, получено E_h = $error",
        )
    }

    /**
     * Итерация Гаусса–Ньютона уменьшает регуляризованный функционал Тихонова.
     *
     * Это базовое свойство корректности спуска: если функционал растёт, значит
     * система для шага собрана с неверным знаком либо матрица не соответствует
     * градиенту.
     */
    @Test
    fun gaussNewtonStepDecreasesTikhonovFunctional() {
        val grid = Grid.uniform(8)
        val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.C.kernel, grid, quad)
        val solver = firstKindSolver(basis, funcs, space, op)
        val thetaFDelta =
            noisyThetaCoefficients(UrysonProblem.C, solver, op, grid, quad, 1e-2, 999L)
        val core = CollocationCore(basis, funcs, op)

        fun tikhonovFunctional(c: DoubleArray): Double {
            val xi = core.xiVector(c)
            var sum = 0.0
            for (j in 0 until grid.n + 2) {
                val d = xi[j] - thetaFDelta[j]
                sum += space.weights[j] * d * d
            }
            return sum + PROBE_ALPHA * space.omegaReg(c)
        }

        val start = DoubleArray(grid.n + 2)
        val valueBefore = tikhonovFunctional(start)
        val afterStep = solver.solveFixedAlpha(thetaFDelta, PROBE_ALPHA, start)
        val valueAfter = tikhonovFunctional(afterStep)
        assertTrue(
            valueAfter <= valueBefore,
            "Итерации Гаусса–Ньютона должны уменьшать функционал Тихонова: " +
                "было $valueBefore, стало $valueAfter",
        )
    }
}
