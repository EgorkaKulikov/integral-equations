package problems.analytic

import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * Создаёт решатель уравнения Фредгольма II рода для аналитической задачи.
 *
 * В отличие от `problems.fredholm.secondKindSolver`, правая часть и её производные
 * берутся из АНАЛИТИЧЕСКИХ формул задачи и не проходят через квадратуру проекта.
 *
 * @param throwOnDivergence политика обработки расходимости итерационных схем;
 *        значение `false` нужно тестам, изучающим САМУ расходимость на задачах
 *        со спектральным радиусом больше единицы (см. [AnalyticFredholmProblem.supportsFixedPointSchemes]).
 */
fun analyticFredholmSolver(
    problem: AnalyticFredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
    throwOnDivergence: Boolean = true,
): FredholmSecondKindSolver = FredholmSecondKindSolver(
    basis, funcs, op, cL = 1.0,
    rhs = RhsWithDerivatives(
        value = problem.rhs,
        deriv = problem.rhsDeriv,
        deriv2 = problem.rhsDeriv2,
    ),
    throwOnDivergence = throwOnDivergence,
)

/**
 * Создаёт решатель уравнения Вольтерры II рода для аналитической задачи.
 *
 * Правая часть и её производные — аналитические, без обращения к квадратуре проекта.
 */
fun analyticVolterraSolver(
    problem: AnalyticVolterraProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: VolterraOperator,
): VolterraSecondKindSolver = VolterraSecondKindSolver(
    basis, funcs, op, cL = 1.0,
    rhs = RhsWithDerivatives(
        value = problem.rhs,
        deriv = problem.rhsDeriv,
        deriv2 = problem.rhsDeriv2,
    ),
)
