package problems.analytic

import splines.MinimalSplineBasis
import splines.functionals.FunctionalFamily
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.volterra.VolterraOperator
import solvers.volterra.VolterraSecondKindSolver

/**
 * Creates a solver of the Fredholm equation of the second kind for an analytic problem.
 *
 * Unlike `problems.fredholm.secondKindSolver`, the right-hand side and its derivatives
 * come from the ANALYTIC formulas of the problem and do not pass through the project quadrature.
 *
 * @param throwOnDivergence policy for handling divergence of the iterative schemes;
 *        the value `false` is needed by tests that study divergence ITSELF on problems
 *        with spectral radius greater than one (see [AnalyticFredholmProblem.supportsFixedPointSchemes]).
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
 * Creates a solver of the Volterra equation of the second kind for an analytic problem.
 *
 * The right-hand side and its derivatives are analytic, with no recourse to the project quadrature.
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
