package solvers.core

/**
 * Result of the regularized solution of a first-kind equation.
 *
 * @param coeffs coefficients of the spline found.
 * @param eval evaluator of the approximate solution.
 * @param alpha the regularization parameter chosen.
 * @param residual the discrete residual at that parameter.
 * @param omega value of the stabilizer `c^T R_h c`.
 */
public class FirstKindSolution(
    public val coeffs: DoubleArray,
    public val eval: (Double) -> Double,
    public val alpha: Double,
    public val residual: Double,
    public val omega: Double,
)
