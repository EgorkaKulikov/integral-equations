package solvers.core

/**
 * Результат регуляризованного решения уравнения первого рода.
 *
 * @param coeffs коэффициенты найденного сплайна.
 * @param eval вычислитель приближённого решения.
 * @param alpha выбранный параметр регуляризации.
 * @param resid дискретная невязка при этом параметре.
 * @param omega значение стабилизатора `c^T R_h c`.
 */
public class FirstKindSolution(
    public val coeffs: DoubleArray,
    public val eval: (Double) -> Double,
    public val alpha: Double,
    public val resid: Double,
    public val omega: Double,
)
