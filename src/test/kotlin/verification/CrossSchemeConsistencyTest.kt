package verification

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.AveragingFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * ПЕРЕКРЁСТНАЯ ПРОВЕРКА СХЕМ (задача 2.4).
 *
 * Разные схемы приближают ОДНО И ТО ЖЕ решение одного уравнения, поэтому при
 * достаточно мелкой сетке обязаны быть согласованы между собой: попарная разность
 * не может существенно превосходить сумму их собственных погрешностей.
 *
 * Что даёт проверка. Она не доказывает правильность (все схемы могли бы ошибаться
 * согласованно — например, из-за общей ошибки в базисе или квадратуре), но
 * обнаруживает ситуацию, когда ОДНА схема рассогласована с остальными. Это
 * дополняет сверку с публикацией: там проверяются числа, здесь — взаимная
 * непротиворечивость независимо реализованных путей вычисления.
 *
 * Отличие от `PublishedValuesTest`: сравниваются не сводные величины `E_h`,
 * а ЗНАЧЕНИЯ РЕШЕНИЙ в наборе точек. Совпадение `E_h` двух схем ещё не означает,
 * что они дают одну функцию: максимум погрешности мог бы достигаться в разных
 * точках при разном поведении между ними.
 */
class CrossSchemeConsistencyTest {

    private companion object {
        /**
         * Множитель запаса при сравнении двух схем.
         *
         * Обоснование. Из неравенства треугольника
         *   |u_A(t) - u_B(t)| <= |u_A(t) - u*(t)| + |u*(t) - u_B(t)| <= E_A + E_B <= 2 max(E_A, E_B),
         * то есть математически гарантированная граница равна 2. Множитель 4 берёт
         * двукратный запас на то, что `E_h` измеряется на конечной выборке точек
         * (`errorEh` — 100n+1 точка) и может слегка недооценивать истинный максимум
         * равномерной нормы.
         *
         * Больший запас брать нельзя: при множителе порядка десятков проверка
         * перестала бы отличать согласованные схемы от рассогласованной.
         */
        const val PAIRWISE_SAFETY_FACTOR = 4.0

        /**
         * Абсолютный «пол» сравнения.
         *
         * Когда обе схемы вышли на машинную точность, их погрешности определяются
         * округлением, и требование `|u_A - u_B| <= 4 max(E_A, E_B)` превратилось бы
         * в сравнение двух шумов. Ниже этого порога расхождение считается
         * несущественным независимо от отношения величин.
         */
        const val ABSOLUTE_FLOOR = 1e-12

        /** Порядок квадратуры — тот же, что в остальных проверках и демонстрациях. */
        const val QUADRATURE_ORDER = 8

        /** Число точек сравнения решений внутри отрезка. */
        const val COMPARISON_POINTS = 200
    }

    /** Точки сравнения, равномерно покрывающие отрезок сетки (включая концы). */
    private fun comparisonPoints(grid: Grid): DoubleArray =
        DoubleArray(COMPARISON_POINTS + 1) { i ->
            grid.a + (grid.b - grid.a) * i / COMPARISON_POINTS
        }

    /** Именованное решение: вычислитель и его собственная погрешность `E_h`. */
    private class NamedSolution(val name: String, val eval: (Double) -> Double, val error: Double)

    /**
     * Проверяет попарную согласованность всех схем набора.
     *
     * @param context описание сочетания «задача/базис/семейство/сетка» для сообщения.
     * @param solutions схемы, решающие одну и ту же задачу.
     * @param points точки, в которых сравниваются значения.
     * @param failures накопитель обнаруженных рассогласований.
     */
    private fun checkPairwise(
        context: String,
        solutions: List<NamedSolution>,
        points: DoubleArray,
        failures: MutableList<String>,
    ) {
        for (i in solutions.indices) {
            for (j in i + 1 until solutions.size) {
                val a = solutions[i]
                val b = solutions[j]
                var worstDifference = 0.0
                var worstPoint = Double.NaN
                for (t in points) {
                    val difference = abs(a.eval(t) - b.eval(t))
                    if (difference > worstDifference) {
                        worstDifference = difference
                        worstPoint = t
                    }
                }
                val allowed = maxOf(
                    PAIRWISE_SAFETY_FACTOR * maxOf(a.error, b.error),
                    ABSOLUTE_FLOOR,
                )
                if (worstDifference > allowed) {
                    failures += buildString {
                        append(context)
                        append(": схемы '").append(a.name).append("' и '").append(b.name)
                        append("' рассогласованы. max|u_A - u_B| = ").append(worstDifference)
                        append(" в точке t = ").append(worstPoint)
                        append(", допустимо ").append(allowed)
                        append(" (= ").append(PAIRWISE_SAFETY_FACTOR).append(" * max(E_A, E_B)); ")
                        append("E(").append(a.name).append(") = ").append(a.error)
                        append(", E(").append(b.name).append(") = ").append(b.error)
                    }
                }
            }
        }
    }

    private fun families(basis: MinimalSplineBasis): List<FunctionalFamily> = listOf(
        // Только семейства БЕЗ производной: схемы Nyström не поддерживают xi.
        ProjFunctionals(basis),
        AveragingFunctionals(basis),
        ThreePointFunctionals(basis),
    )

    private fun reportFailures(failures: List<String>, title: String) {
        assertTrue(
            failures.isEmpty(),
            "$title: обнаружено рассогласование схем (${failures.size} шт.). " +
                "Разные схемы приближают одно решение и обязаны сходиться к одному пределу:\n" +
                failures.joinToString("\n").take(6000),
        )
    }

    /**
     * Фредгольм II рода, n = 64: base, sloan, kulkarni, nystrom, combinedNystrom.
     *
     * Сетка максимальна из используемых в статье; для уравнения Фредгольма все
     * схемы на ней ещё считаются за разумное время (образы базисных сплайнов
     * предвычисляются в фиксированных узлах квадратуры).
     */
    @Test
    fun fredholmSchemesAgreeAtFinestGrid() {
        val failures = mutableListOf<String>()
        val problem = problems.fredholm.FredholmProblem.F2
        val n = 64
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val points = comparisonPoints(grid)
            val exact = { t: Double -> problem.exact(t) }
            for (funcs in families(basis)) {
                val op = solvers.fredholm.FredholmOperator(
                    problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                )
                val solver = solvers.fredholm.SecondKindSolver(
                    basis, funcs, op, 1.0,
                    { t -> problem.rhsExact(t, op) },
                    { t -> problem.rhsExactDeriv(t, op) },
                    { t -> problem.rhsExactDeriv2(t, op) },
                )
                val solutions = listOf(
                    "base" to solver.base(),
                    "sloan" to solver.sloan(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                    "combinedNystrom" to solver.combinedNystrom(),
                ).map { (name, solution) ->
                    NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
                }
                checkPairwise(
                    "Фредгольм ${problem.name}, базис ${system.name}, " +
                        "семейство ${funcs.name}, n=$n",
                    solutions, points, failures,
                )
            }
        }
        reportFailures(failures, "Фредгольм II рода")
    }

    /**
     * Вольтерра II рода, n = 32.
     *
     * Сетка вдвое грубее, чем для Фредгольма: у оператора Вольтерры область
     * интегрирования зависит от `t`, поэтому предвычисление на фиксированных узлах
     * невозможно, а веса Nyström `W_j(t)` приходится пересчитывать для каждой точки.
     * По этой же причине в статье таблицы Nyström для Вольтерры ограничены n <= 32.
     */
    @Test
    fun volterraSchemesAgreeAtFinestGrid() {
        val failures = mutableListOf<String>()
        val problem = problems.volterra.VolterraProblem.V2
        val n = 32
        for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val points = comparisonPoints(grid)
            val exact = { t: Double -> problem.exact(t) }
            for (funcs in families(basis)) {
                val op = solvers.volterra.VolterraOperator(
                    problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
                )
                val solver = solvers.volterra.SecondKindSolver(
                    basis, funcs, op, 1.0,
                    { t -> problem.rhsExact(t, op) },
                    { t -> problem.rhsExactDeriv(t, op) },
                    { t -> problem.rhsExactDeriv2(t, op) },
                )
                val solutions = listOf(
                    "base" to solver.base(),
                    "sloan" to solver.sloan(),
                    "kulkarni" to solver.kulkarni(),
                    "nystrom" to solver.nystrom(),
                    "combinedNystrom" to solver.combinedNystrom(),
                ).map { (name, solution) ->
                    NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
                }
                checkPairwise(
                    "Вольтерра ${problem.name}, базис ${system.name}, " +
                        "семейство ${funcs.name}, n=$n",
                    solutions, points, failures,
                )
            }
        }
        reportFailures(failures, "Вольтерра II рода")
    }

    /**
     * Согласованность на задаче, решение которой лежит в span порождающей системы.
     *
     * Здесь погрешности схем близки к машинной точности, и проверка вырождается в
     * требование «все схемы дают практически одну и ту же функцию». Случай ценен
     * тем, что порог сравнения определяется абсолютным полом, а не собственными
     * ошибками схем: рассогласование любой из них видно немедленно.
     */
    @Test
    fun schemesAgreeOnSpanProblem() {
        val failures = mutableListOf<String>()
        val problem = problems.fredholm.FredholmProblem.F2span
        val n = 16
        val grid = Grid.uniform(n)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val points = comparisonPoints(grid)
        val exact = { t: Double -> problem.exact(t) }
        val funcs = ProjFunctionals(basis)
        val op = solvers.fredholm.FredholmOperator(
            problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER),
        )
        val solver = solvers.fredholm.SecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        // Схема nystrom исключена намеренно: её приближение лежит ВНЕ сплайнового
        // пространства, поэтому на span-задаче она не обязана давать машинную
        // точность (см. KDoc SecondKindSolver.nystrom) и её погрешность ~5e-5
        // определяет порог сравнения, обесценивая проверку.
        val solutions = listOf(
            "base" to solver.base(),
            "sloan" to solver.sloan(),
            "kulkarni" to solver.kulkarni(),
            "iteratedKulkarni" to solver.iteratedKulkarni(),
        ).map { (name, solution) ->
            NamedSolution(name, solution.eval, errorEh(exact, solution.eval, grid))
        }
        checkPairwise(
            "Фредгольм ${problem.name} (решение в span), базис B, семейство theta, n=$n",
            solutions, points, failures,
        )
        reportFailures(failures, "Задача с решением в span порождающей системы")
    }
}
