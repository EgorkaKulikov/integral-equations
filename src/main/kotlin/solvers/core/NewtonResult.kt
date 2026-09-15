package solvers.core

/**
 * Результат итераций Ньютона в пространстве коэффициентов.
 *
 * Ранее [UrysonSecondKindSolver.solveBase] возвращал `Pair<DoubleArray, Int>`, из которого
 * было невозможно узнать, сошлась ли итерация: число итераций, равное пределу,
 * одинаково возникает и при сходимости на последнем шаге, и при расходимости.
 *
 * @param coeffs найденные коэффициенты сплайна.
 * @param converged признак достижения сходимости.
 * @param iterations число ФАКТИЧЕСКИ ВЫПОЛНЕННЫХ шагов Ньютона (см. [NewtonRun]);
 *        `0` означает, что коэффициенты не изменились относительно начального
 *        приближения.
 * @param residual норма невязки. При `converged == true` — измеренная В ТОЙ ЖЕ
 *        точке, что возвращается в [coeffs]. При исчерпании предела шагов — невязка
 *        ПЕРЕД ПОСЛЕДНИМ шагом, а НЕ в возвращаемой точке: сознательное ограничение,
 *        полное обоснование — в KDoc [runNewtonIterations] и [NewtonRun.residual].
 */
public class NewtonResult(
    public val coeffs: DoubleArray,
    public val converged: Boolean,
    public val iterations: Int,
    public val residual: Double,
)
