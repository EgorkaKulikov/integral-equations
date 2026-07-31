package verification

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import numerics.functionals.errorEh
import problems.fredholm.FredholmProblem
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import java.io.File

/**
 * ВЫГРУЗКА ВНУТРЕННИХ АРТЕФАКТОВ ВЫЧИСЛЕНИЙ для внешней сверки со SciPy/NumPy.
 *
 * Почему выгружаются именно внутренние артефакты, а не только итоговые `E_h`:
 * сверка «числа с числами» на выходе проверяла бы совпадение двух реализаций
 * целиком, и при расхождении было бы неясно, какой слой виноват. Выгрузка узлов
 * квадратуры, значений базиса, собранных матриц и образов операторов позволяет
 * проверить каждый слой ОТДЕЛЬНО и указать место расхождения.
 *
 * Вынесено в отдельный объект (а не оставлено внутри тестового класса), поскольку
 * используется двумя потребителями:
 *  - [ScipyCrossVerificationTest] — смоук-тест, готовит данные сам, чтобы не
 *    зависеть от того, запускалась ли перед ним отдельная задача Gradle;
 *  - [VerificationArtifactDumpTool] — задача `dumpVerificationArtifacts` для
 *    ручной выгрузки и разбора расхождений вне тестового прогона.
 */
object VerificationArtifacts {

    /** Каталог выгрузки по умолчанию; лежит в build/, поэтому в репозиторий не попадает. */
    val DEFAULT_DIR: File = File("build/verification")

    /** Сетка, на которой выгружаются базис и матрицы: достаточно мелкая и быстрая. */
    private const val DUMP_GRID_SIZE = 8

    /** Печать числа с полной точностью: сверка идёт на уровне 1e-15, округлять нельзя. */
    private fun Double.full(): String = "%.17g".format(this)

    /** Выгружает все артефакты в каталог [dir]. */
    fun dumpAll(dir: File = DEFAULT_DIR): List<File> {
        dir.mkdirs()
        return listOf(
            dumpGaussLegendreNodes(dir),
            dumpSplineKnots(dir),
            dumpSplineValues(dir),
            dumpAssembledSystem(dir),
            dumpOperatorImages(dir),
            dumpSolutionErrors(dir),
        )
    }

    /**
     * L1. Узлы и веса квадратуры Гаусса–Лежандра на `[-1,1]` для m = 1..16.
     *
     * Эталон в SciPy — `numpy.polynomial.legendre.leggauss`: принципиально другой
     * алгоритм (проект использует метод Ньютона по нулям многочлена Лежандра).
     */
    fun dumpGaussLegendreNodes(dir: File): File {
        val file = File(dir, "gauss-legendre.tsv")
        file.printWriter().use { out ->
            out.println("# m\tindex\tnode\tweight")
            for (m in 1..16) {
                val (nodes, weights) = GaussLegendre.gaussLegendreReference(m)
                for (i in nodes.indices) {
                    out.println("$m\t$i\t${nodes[i].full()}\t${weights[i].full()}")
                }
            }
        }
        return file
    }

    /** Сетки, на которых сверяется базис: равномерная и три существенно неравномерные. */
    private fun dumpGrids(): Map<String, Grid> = linkedMapOf(
        "uniform" to Grid.uniform(DUMP_GRID_SIZE),
        "quasiUniform" to Grid.quasiUniform(DUMP_GRID_SIZE),
        "geometric" to Grid.geometric(DUMP_GRID_SIZE),
        "graded" to Grid.graded(DUMP_GRID_SIZE),
    )

    /**
     * Полный вектор узлов `x_{-2..n+2}`.
     *
     * Узлы кратности 3 на концах в точности задают клампованный вектор узлов
     * степени 2, поэтому `scipy.interpolate.BSpline` строит на них то же
     * пространство: число базисных функций совпадает, `(n+5) - 2 - 1 = n + 2`.
     */
    fun dumpSplineKnots(dir: File): File {
        val file = File(dir, "spline-knots.tsv")
        file.printWriter().use { out ->
            out.println("# grid\tindex\tknot")
            for ((name, grid) in dumpGrids()) {
                for (j in -2..grid.n + 2) out.println("$name\t$j\t${grid.x(j).full()}")
            }
        }
        return file
    }

    /**
     * L3. Значения базиса минимальных сплайнов и двух его производных для
     * полиномиальной системы `B`.
     *
     * Сверка возможна именно для `B`: для систем `H` и `T` аналога в SciPy нет
     * (см. docs/REFERENCES.md, раздел 6).
     */
    fun dumpSplineValues(dir: File): File {
        val file = File(dir, "spline-values.tsv")
        // Число точек взаимно просто с числом узлов: выборка не попадает на стыки.
        val sampleCount = 97
        file.printWriter().use { out ->
            out.println("# grid\tj\tt\tomega\tomegaDeriv\tomegaDeriv2")
            for ((name, grid) in dumpGrids()) {
                val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                for (i in 0..sampleCount) {
                    val t = grid.a + (grid.b - grid.a) * i / sampleCount
                    for (j in -2..grid.n - 1) {
                        out.println(
                            "$name\t$j\t${t.full()}\t${basis.omega(j, t).full()}\t" +
                                "${basis.omegaDeriv(j, t).full()}\t${basis.omegaDeriv2(j, t).full()}",
                        )
                    }
                }
            }
        }
        return file
    }

    /**
     * L2. Собранные матрицы `M`, `M2`, векторы `g`, `d` и коэффициенты базовой схемы.
     *
     * Эталон в SciPy — `scipy.linalg.solve` на той же системе `(I - M) c = g`;
     * дополнительно сообщается обусловленность (`numpy.linalg.cond`), которая
     * объясняет достижимую точность.
     */
    fun dumpAssembledSystem(dir: File): File {
        val problem = FredholmProblem.F2
        val grid = Grid.uniform(DUMP_GRID_SIZE)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
        val solver = FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            { t -> problem.rhsExact(t, op) },
            { t -> problem.rhsExactDeriv(t, op) },
            { t -> problem.rhsExactDeriv2(t, op) },
        )
        val m = solver.matrixM()
        val m2 = solver.matrixM2()
        val g = solver.vectorG()
        val d = solver.vectorD()
        val coeffs = solver.solveBaseCoeffs()

        val file = File(dir, "assembled-system.tsv")
        file.printWriter().use { out ->
            out.println("# kind\trow\tcol\tvalue")
            for (r in m.indices) for (c in m[r].indices) out.println("M\t$r\t$c\t${m[r][c].full()}")
            for (r in m2.indices) for (c in m2[r].indices) out.println("M2\t$r\t$c\t${m2[r][c].full()}")
            for (i in g.indices) out.println("g\t$i\t0\t${g[i].full()}")
            for (i in d.indices) out.println("d\t$i\t0\t${d[i].full()}")
            for (i in coeffs.indices) out.println("c_base\t$i\t0\t${coeffs[i].full()}")
        }
        return file
    }

    /**
     * L4/L5. Образы интегральных операторов и правые части модельных задач.
     *
     * Эталон в SciPy — `scipy.integrate.quad` (адаптивный QUADPACK), независимый от
     * составной квадратуры проекта. Это единственная численная проверка формул
     * Лейбница для `(Vu)'` и `(Vu)''`, у которых отдельной публикации нет.
     */
    fun dumpOperatorImages(dir: File): File {
        val grid = Grid.uniform(DUMP_GRID_SIZE)
        val quad = GaussLegendre(8)
        val samplePoints = (0..20).map { grid.a + (grid.b - grid.a) * it / 20.0 }

        val file = File(dir, "operator-images.tsv")
        file.printWriter().use { out ->
            out.println("# equation\tproblem\tquantity\tt\tvalue")
            for (problem in listOf(FredholmProblem.F2, FredholmProblem.F2exp)) {
                val op = FredholmOperator(problem.kernel, grid, quad)
                for (t in samplePoints) {
                    val prefix = "F\t${problem.name}"
                    out.println("$prefix\tKu\t${t.full()}\t${op.apply(t) { s -> problem.exact(s) }.full()}")
                    out.println("$prefix\trhs\t${t.full()}\t${problem.rhsExact(t, op).full()}")
                    out.println("$prefix\trhsDeriv\t${t.full()}\t${problem.rhsExactDeriv(t, op).full()}")
                    out.println("$prefix\trhsDeriv2\t${t.full()}\t${problem.rhsExactDeriv2(t, op).full()}")
                }
            }
            for (problem in listOf(
                problems.volterra.VolterraProblem.V2,
                problems.volterra.VolterraProblem.V2exp,
                problems.volterra.VolterraProblem.V2win,
            )) {
                val op = solvers.volterra.VolterraOperator(problem.kernel, grid, quad)
                for (t in samplePoints) {
                    val prefix = "V\t${problem.name}"
                    out.println("$prefix\tVu\t${t.full()}\t${op.apply(t) { s -> problem.exact(s) }.full()}")
                    out.println("$prefix\trhs\t${t.full()}\t${problem.rhsExact(t, op).full()}")
                    out.println("$prefix\trhsDeriv\t${t.full()}\t${problem.rhsExactDeriv(t, op).full()}")
                    out.println("$prefix\trhsDeriv2\t${t.full()}\t${problem.rhsExactDeriv2(t, op).full()}")
                }
            }
        }
        return file
    }

    /**
     * L6. Итоговые погрешности `E_h` базовой схемы и итерации Слоана — для сверки
     * с решением, построенным независимым методом Nyström средствами NumPy.
     */
    fun dumpSolutionErrors(dir: File): File {
        val file = File(dir, "solution-errors.tsv")
        file.printWriter().use { out ->
            out.println("# problem\tsystem\tn\tscheme\tEh")
            for (problem in listOf(FredholmProblem.F2, FredholmProblem.F2exp)) {
                for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                    for (n in listOf(8, 16, 32)) {
                        val grid = Grid.uniform(n)
                        val basis = MinimalSplineBasis(system, grid)
                        val funcs = ProjFunctionals(basis)
                        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                        val solver = FredholmSecondKindSolver(
                            basis, funcs, op, 1.0,
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
                        )
                        val exact = { t: Double -> problem.exact(t) }
                        val prefix = "${problem.name}\t${system.name}\t$n"
                        out.println("$prefix\tbase\t${errorEh(exact, solver.base().eval, grid).full()}")
                        out.println("$prefix\tsloan\t${errorEh(exact, solver.sloan().eval, grid).full()}")
                    }
                }
            }
        }
        return file
    }
}
