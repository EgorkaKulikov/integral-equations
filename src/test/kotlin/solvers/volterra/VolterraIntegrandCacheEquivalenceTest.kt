package solvers.volterra

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import org.junit.jupiter.api.Tag
import problems.volterra.VolterraProblem
import java.util.concurrent.Callable
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * ПОБИТОВАЯ эквивалентность кэшированного и некэшированного применения оператора
 * Вольтерры: [VolterraOperator.apply] с [VolterraOperator.IntegrandCache] обязан давать
 * ТОТ ЖЕ double, что и [VolterraOperator.apply] с функцией.
 *
 * Зачем нужен именно такой тест. Мемоизация значений `u` в узлах полных ячеек — чистая
 * оптимизация горячего пути, и её корректность до сих пор подтверждалась лишь косвенно:
 * characterization-эталон (1316 значений E_h, допуск 1e-9) совпал бит-в-бит. Но эталон
 * покрывает КОНКРЕТНЫЙ набор конфигураций; кэш мог бы разойтись с оригиналом на других
 * входах, прежде всего на границах, где отбор «полных» ячеек зависит от сравнения с
 * порогом [VolterraOperator.breakpointInclusionEps]. Этот тест закрывает дыру
 * прямой проверкой на матрице граничных случаев.
 *
 * О МАСШТАБЕ ОТРЕЗКА: порог включения ОТНОСИТЕЛЕН (пропорционален `b-a`), поэтому
 * сетки берутся не только на дефолтном `[0,1]`, но и на отрезках БОЛЬШОГО масштаба
 * (`[0,1000]`, `[-500,500]`), где абсолютный порог 1e-15 ушёл бы под ulp узлов. Без них
 * новое поведение порога не проверялось бы вовсе (характеризационный эталон тоже снят
 * только на `[0,1]`).
 *
 * Почему сравнение ПОБИТОВОЕ, а не с допуском. Требование к этапам 1-7 рефакторинга —
 * поведенческая нейтральность: эталон не перегенерируется ни на одну строку. Умножение
 * double неассоциативно, поэтому даже перестановка множителей в накоплении суммы
 * (`(half*w*K)*u` вместо `half*w*(K*u)`) меняет последний бит результата и «уплывает»
 * в эталоне. Тест с допуском такую регрессию пропустит, побитовый — поймает.
 *
 * Матрица покрытия по `t` (см. [boundaryTs], [nodeThresholdTs], [interiorTs]):
 * `t = a` (пустой интеграл), `t = b` (все ячейки полные), `t` РОВНО в узле сетки,
 * `t` в окрестности узла (±1e-16, ±1e-15, ±1e-14 — вокруг самого порога),
 * `t` внутри первой ячейки (полных ячеек ещё нет), `t` внутри последней ячейки
 * и несколько произвольных точек в середине отрезка.
 *
 * Матрица по сеткам и данным: равномерная (n = 4, 8, 16), геометрическая и
 * градуированная (существенно неравномерные); ядра трёх модельных задач проекта
 * (`V2`: 1/(1+t+s), `V2exp`: exp(-(t-s)^2), `V2win`: t-s); подынтегральные функции —
 * гладкая, полиномиальная и ДВЕ сплайновых того же вида, что реально приходят в
 * `SecondKindSolver.applyL` (значение сплайна по коэффициентам и базисный `omega_j`).
 */
@Tag("fast")
class VolterraIntegrandCacheEquivalenceTest {

    private val quad = GaussLegendre(8)

    /** Эталонные узлы/веса на [-1,1] — нужны диагностике для воспроизведения арифметики. */
    private val refNodes = quad.refNodesWeights().first
    private val refWeights = quad.refNodesWeights().second

    // --- Описание сочетаний -----------------------------------------------------

    private class GridCase(val name: String, val grid: Grid)
    private class KernelCase(val name: String, val kernel: KernelV)
    private class IntegrandCase(val name: String, val u: (Double) -> Double)
    private class TCase(val t: Double, val label: String)

    /**
     * Сетки берутся МАЛЫЕ, но разных семейств: расхождение кэша с оригиналом — свойство
     * структуры разбиения (порог отбора ячеек), а не размера n, поэтому крупные сетки
     * ничего не добавили бы, но выбили бы тест из бюджета fast-набора.
     */
    private fun gridCases(): List<GridCase> = listOf(
        GridCase("uniform n=4", Grid.uniform(4)),
        GridCase("uniform n=8", Grid.uniform(8)),
        GridCase("uniform n=16", Grid.uniform(16)),
        GridCase("geometric n=8 R=2", Grid.geometric(8, R = 2.0)),
        GridCase("graded n=8 ratio=2", Grid.graded(8, ratio = 2.0)),
        // Отрезки БОЛЬШОГО масштаба: здесь порог включения уже НЕ равен 1e-15
        // (он относительный), и именно они покрывают новое поведение: на них
        // абсолютные 1e-15 лежат ниже ulp узлов (ulp(1e3) ~ 2.3e-13).
        GridCase("uniform n=8 на [0,1000] (большой масштаб)", Grid.uniform(8, 0.0, 1000.0)),
        GridCase("uniform n=8 на [-500,500] (большой масштаб, знакопеременный)", Grid.uniform(8, -500.0, 500.0)),
        GridCase("graded n=8 ratio=2 на [0,1000]", Grid.graded(8, 0.0, 1000.0, ratio = 2.0)),
    )

    /** Ядра — из модельных задач проекта (фикстуры), а не выдуманные. */
    private val kernelCases = listOf(
        KernelCase("V2 K=1/(1+t+s)", VolterraProblem.V2.kernel),
        KernelCase("V2exp K=exp(-(t-s)^2)", VolterraProblem.V2exp.kernel),
        KernelCase("V2win K=t-s", VolterraProblem.V2win.kernel),
    )

    /**
     * Подынтегральные функции. Сплайновые варианты обязательны: в решателе через
     * `applyL` проходят ровно они (`basis.evalSpline(c, s)` в `sloan`/`iteratedKulkarni`
     * и `basis.omega(i, s)` в `matrixM`), а сплайн кусочно-задан и потому чувствителен
     * к тому, по какую сторону узла попал аргумент.
     */
    private fun integrands(grid: Grid): List<IntegrandCase> {
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        // Произвольные, но детерминированные коэффициенты (не «красивые» числа, чтобы
        // в накоплении суммы участвовали все биты мантиссы).
        val coeffs = DoubleArray(grid.n + 2) { i -> Math.sin(0.7 * i + 0.3) }
        return listOf(
            IntegrandCase("гладкая sin(3s)*exp(-s/2)") { s -> Math.sin(3.0 * s) * Math.exp(-0.5 * s) },
            IntegrandCase("полином 1-2s+3s^2-s^3") { s -> 1.0 - 2.0 * s + 3.0 * s * s - s * s * s },
            IntegrandCase("сплайн evalSpline(c,s) (как в sloan)") { s -> basis.evalSpline(coeffs, s) },
            IntegrandCase("базисный сплайн omega_1(s) (как в matrixM)") { s -> basis.omega(1, s) },
        )
    }

    // --- Наборы значений t ------------------------------------------------------

    /** Концы отрезка: при `t=a` интеграл пуст, при `t=b` полными являются ВСЕ ячейки. */
    private fun boundaryTs(grid: Grid): List<TCase> = listOf(
        TCase(grid.a, "t=a=${grid.a} (нижняя граница, интеграл пуст)"),
        TCase(grid.b, "t=b=${grid.b} (верхняя граница, все ячейки полные)"),
    )

    /**
     * Узлы сетки и их окрестность. Это САМОЕ опасное место: оба пути решают «включать ли
     * узел x_j в разбиение» сравнением `x_j < t - 1e-15`, и любое расхождение в этом
     * решении меняет и число полных ячеек, и границы усечённой ячейки. Сдвиги 1e-16 /
     * 1e-15 / 1e-14 выбраны вокруг самого порога: меньше порога, на пороге, больше порога.
     */
    private fun nodeThresholdTs(grid: Grid): List<TCase> {
        val res = ArrayList<TCase>()
        for (j in 1 until grid.n) {
            val x = grid.x(j)
            res.add(TCase(x, "t=x_$j=$x РОВНО узел сетки"))
            for (d in listOf(1e-16, 1e-15, 1e-14)) {
                res.add(TCase(x - d, "t=x_$j-$d (x_$j=$x)"))
                res.add(TCase(x + d, "t=x_$j+$d (x_$j=$x)"))
            }
        }
        return res
    }

    /** Внутренние точки: первая ячейка (полных ячеек нет), последняя ячейка, середина. */
    private fun interiorTs(grid: Grid): List<TCase> {
        val res = ArrayList<TCase>()
        val x1 = grid.x(1)
        val xLast = grid.x(grid.n - 1)
        res.add(TCase(grid.a + 0.25 * (x1 - grid.a), "t в первой четверти ПЕРВОЙ ячейки (полных ячеек нет)"))
        res.add(TCase(0.5 * (grid.a + x1), "t в середине ПЕРВОЙ ячейки (полных ячеек нет)"))
        res.add(TCase(0.5 * (xLast + grid.b), "t в середине ПОСЛЕДНЕЙ ячейки"))
        res.add(TCase(grid.b - 0.25 * (grid.b - xLast), "t в последней четверти ПОСЛЕДНЕЙ ячейки"))
        for (frac in listOf(0.137, 0.381, 0.5, 0.6180339887498949, 0.9042)) {
            res.add(TCase(grid.a + frac * (grid.b - grid.a), "t=a+$frac*(b-a) (произвольная точка внутри)"))
        }
        return res
    }

    // --- Тесты ------------------------------------------------------------------

    /**
     * Границы отрезка. При `t <= a` оба пути обязаны вернуть ровно `0.0` по раннему
     * возврату, при `t = b` работает самый длинный вариант: n-1 полная ячейка из кэша
     * плюс усечённая ячейка `[x_{n-1}, b]`.
     */
    @Test fun boundaries_cachedMatchesUncachedBitwise() {
        val stats = compareAll("границы отрезка") { grid -> boundaryTs(grid) }
        assertNonDegenerate(stats)
    }

    /**
     * Узлы сетки и окрестность порога [VolterraOperator.breakpointInclusionEps].
     * Именно здесь кэшированный путь (счётчик `included` в цикле `while`) мог бы отобрать
     * иное число полных ячеек, чем `subBreakpoints` (цикл `for` с `break`).
     */
    @Test fun gridNodesAndThresholdNeighborhood_cachedMatchesUncachedBitwise() {
        val stats = compareAll("узлы сетки и окрестность порога включения") { grid -> nodeThresholdTs(grid) }
        assertNonDegenerate(stats)
    }

    /**
     * ОКРЕСТНОСТЬ САМОГО ПОРОГА, вычисленного по масштабу конкретной сетки.
     *
     * [nodeThresholdTs] сдвигает `t` на ФИКСИРОВАННЫЕ 1e-16/1e-15/1e-14 — это было
     * окрестностью порога, пока порог был абсолютным. Теперь порог зависит от
     * масштаба отрезка, и на `[0,1000]` те сдвиги на три порядка МЕНЬШЕ порога
     * (и вообще ниже ulp узлов), то есть саму точку переключения решения больше не
     * задевают. Здесь сдвиги берутся ОТ ФАКТИЧЕСКОГО порога оператора, поэтому
     * граница «включать / не включать узел» проверяется на ЛЮБОМ масштабе.
     */
    @Test fun scaledThresholdNeighborhood_cachedMatchesUncachedBitwise() {
        var comparisons = 0
        var crossings = 0
        for (g in gridCases()) {
            val eps = VolterraOperator(kernelCases[0].kernel, g.grid, quad).breakpointInclusionEps
            val ts = ArrayList<TCase>()
            for (j in 1 until g.grid.n) {
                val x = g.grid.x(j)
                // Ровно на пороге, чуть ниже и чуть выше — в единицах САМОГО порога.
                for (mult in listOf(0.5, 1.0, 2.0, 10.0)) {
                    ts.add(TCase(x + mult * eps, "t=x_$j+$mult*eps (eps=$eps, x_$j=$x)"))
                    ts.add(TCase(x - mult * eps, "t=x_$j-$mult*eps (eps=$eps, x_$j=$x)"))
                }
                ts.add(TCase(Math.nextAfter(x + eps, Double.NEGATIVE_INFINITY), "t=nextDown(x_$j+eps)"))
                ts.add(TCase(Math.nextAfter(x + eps, Double.POSITIVE_INFINITY), "t=nextUp(x_$j+eps)"))
            }
            for (k in kernelCases) {
                val op = VolterraOperator(k.kernel, g.grid, quad)
                for (integrand in integrands(g.grid)) {
                    val cache = op.integrandCache(integrand.u)
                    for (tc in ts) {
                        val expected = op.apply(tc.t, integrand.u)
                        val actual = op.apply(tc.t, cache)
                        comparisons++
                        assertBitwise(expected, actual) {
                            report("окрестность ОТНОСИТЕЛЬНОГО порога eps=$eps", g, k, integrand, tc, expected, actual, op, cache)
                        }
                    }
                }
            }
            // Набор не вырожден: среди точек есть и такие, где число полных ячеек РАЗНОЕ
            // по разные стороны порога, иначе проверка не задевала бы точку переключения.
            for (j in 1 until g.grid.n) {
                val x = g.grid.x(j)
                if (fullCellCount(g.grid, x - 0.5 * eps) != fullCellCount(g.grid, x + 2.0 * eps)) crossings++
            }
        }
        assertTrue(comparisons > 0, "Вырожденный тест: ни одной сверки у относительного порога")
        assertTrue(
            crossings > 0,
            "Вырожденный набор: ни на одной сетке переход через порог не меняет числа полных ячеек, "
                + "то есть точка переключения не проверяется",
        )
    }

    /**
     * НА ОТРЕЗКАХ МАСШТАБА <= 1 ПОРОГ СОВПАДАЕТ С ПРЕЖНИМ 1e-15 ПОБИТОВО.
     *
     * Это и есть доказательство нейтральности миграции на относительный порог: характеризационный
     * эталон снят на `[0,1]`, и если порог там тот же до бита, то числа не могли поехать.
     * Сравниваются RAW-биты, а не значения.
     */
    @Test fun thresholdIsBitwiseUnchangedOnUnitScaleSegments() {
        val unitScale = listOf(
            "[0,1] (дефолт, на нём снят эталон)" to Grid.uniform(8),
            "[0,1] n=16" to Grid.uniform(16),
            "[0,0.5] (масштаб МЕНЬШЕ 1)" to Grid.uniform(8, 0.0, 0.5),
            "geometric [0,1]" to Grid.geometric(8, R = 2.0),
            "graded [0,1]" to Grid.graded(8, ratio = 2.0),
        )
        for ((name, grid) in unitScale) {
            val op = VolterraOperator(VolterraProblem.V2.kernel, grid, quad)
            assertBitwise(1e-15, op.breakpointInclusionEps) {
                "На отрезке масштаба <= 1 порог обязан побитово совпадать с прежним абсолютным 1e-15 " +
                    "(иначе миграция не нейтральна и эталон поехал бы): сетка $name, " +
                    "ожидалось " + fmt(1e-15) + ", получено " + fmt(op.breakpointInclusionEps)
            }
        }
    }

    /**
     * На отрезке БОЛЬШОГО масштаба порог ДЕЙСТВИТЕЛЬНО масштабируется и остаётся
     * рабочим допуском — то есть вычитание `t - eps` реально меняет `t`.
     *
     * Именно это было сломано до правки: при абсолютном 1e-15 на `[0,1000]` выражение
     * `t - 1e-15` возвращало РОВНО `t` (сдвиг ниже ulp), и допуск вырождался в строгое `<`.
     */
    @Test fun thresholdActuallyScalesOnLargeSegments() {
        val grid = Grid.uniform(8, 0.0, 1000.0)
        val op = VolterraOperator(VolterraProblem.V2.kernel, grid, quad)
        val eps = op.breakpointInclusionEps
        assertTrue(eps > 1e-15, "На [0,1000] порог обязан быть больше абсолютного 1e-15, получено $eps")
        val x = grid.x(4)
        // Старый абсолютный порог здесь вырожден: вычитание не меняет число ни на бит.
        assertBitwise(x, x - 1e-15) {
            "Ожидалось, что на [0,1000] СТАРЫЙ абсолютный порог вырожден (x-1e-15 == x); " +
                "если это не так, мотивировка правки неверна: x=" + fmt(x)
        }
        // А новый относительный — работает.
        assertTrue(
            x - eps < x,
            "Относительный порог обязан реально сдвигать точку: x=" + fmt(x) + ", eps=" + eps +
                ", x-eps=" + fmt(x - eps),
        )
    }

    /**
     * Внутренние точки, включая первую ячейку (кэш вообще не задействован — путь должен
     * сводиться к одной усечённой ячейке) и последнюю (кэш задействован максимально).
     */
    @Test fun interiorPoints_cachedMatchesUncachedBitwise() {
        val stats = compareAll("внутренние точки") { grid -> interiorTs(grid) }
        assertNonDegenerate(stats)
        // Требуем, чтобы среди сверок были и такие, где полных ячеек НЕТ: иначе ветка
        // «t внутри первой ячейки» осталась бы непройденной, а она отдельная в коде.
        assertTrue(
            stats.comparisons > stats.withFullCells,
            "Вырожденный набор: не осталось ни одной сверки без полных ячеек " +
                "(сверок ${stats.comparisons}, из них с полными ячейками ${stats.withFullCells})",
        )
    }

    /**
     * Независимость от ПОРЯДКА обращений. Кэш переиспользуется одним замыканием при всех
     * `t` (см. `SecondKindSolver.applyL`), а порядок этих `t` определяется вызывающим
     * кодом: сборкой матрицы, итерациями Кулкарни, вычислением невязки. Значения ячеек от
     * `t` не зависят по построению, но именно это и проверяется: прогон в обратном и в
     * перемешанном (с фиксированным зерном) порядке обязан дать те же биты, что и
     * некэшированный путь.
     */
    @Test fun cacheReuseIsOrderIndependent_bitwise() {
        var comparisons = 0
        for (g in gridCases()) {
            val ts = boundaryTs(g.grid) + nodeThresholdTs(g.grid) + interiorTs(g.grid)
            val reversed = ts.reversed()
            val shuffled = ts.shuffled(Random(20240513))
            for (k in kernelCases) {
                val op = VolterraOperator(k.kernel, g.grid, quad)
                for (integrand in integrands(g.grid)) {
                    val expected = HashMap<String, Double>()
                    for (tc in ts) expected[tc.label] = op.apply(tc.t, integrand.u)
                    for ((orderName, order) in listOf("обратный" to reversed, "перемешанный" to shuffled)) {
                        // Свежий кэш на каждый порядок: интересно первое заполнение ячеек
                        // именно в этом порядке обращений.
                        val cache = op.integrandCache(integrand.u)
                        for (tc in order) {
                            val actual = op.apply(tc.t, cache)
                            comparisons++
                            assertBitwise(expected.getValue(tc.label), actual) {
                                report(
                                    "порядок обращений: $orderName", g, k, integrand, tc,
                                    expected.getValue(tc.label), actual, op, cache,
                                )
                            }
                        }
                    }
                }
            }
        }
        assertTrue(comparisons > 0, "Вырожденный тест: ни одной сверки не выполнено")
    }

    /**
     * Детерминизм при КОНКУРЕНТНОМ доступе к одному экземпляру кэша.
     *
     * KDoc [VolterraOperator.IntegrandCache] прямо утверждает: гонка двух потоков на одной
     * ячейке безвредна, потому что `u` чиста и оба вычислят побитово одинаковые числа, а
     * победитель `compareAndSet` определяет, чей массив увидят остальные. Утверждение
     * проверяется ФАКТИЧЕСКИ: несколько потоков стартуют одновременно (барьер) и идут по
     * одному и тому же списку `t` в одном порядке — то есть все сразу претендуют на первое
     * заполнение ячейки 0, потом ячейки 1 и так далее. Это максимизирует конкуренцию.
     *
     * Тест детерминирован: фиксированное число потоков, фиксированный список `t`,
     * фиксированное число повторов; проверяется побитовое равенство с ПОСЛЕДОВАТЕЛЬНЫМ
     * некэшированным эталоном, а не «отсутствие исключений», поэтому исход не зависит от
     * того, случилась ли гонка на самом деле.
     */
    @Test fun concurrentCacheUse_isBitwiseDeterministic() {
        val grid = Grid.uniform(8)
        val op = VolterraOperator(VolterraProblem.V2exp.kernel, grid, quad)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val coeffs = DoubleArray(grid.n + 2) { i -> Math.sin(0.7 * i + 0.3) }
        // Сплайн, а не тригонометрия: именно такие функции кэшируются в решателе.
        val u = { s: Double -> basis.evalSpline(coeffs, s) }
        val ts = boundaryTs(grid) + nodeThresholdTs(grid) + interiorTs(grid)
        val expected = DoubleArray(ts.size) { i -> op.apply(ts[i].t, u) }

        val threads = 4
        val repeats = 3
        val pool = Executors.newFixedThreadPool(threads)
        try {
            repeat(repeats) { iteration ->
                val cache = op.integrandCache(u)
                val barrier = CyclicBarrier(threads)
                val futures = (0 until threads).map {
                    pool.submit(
                        Callable {
                            barrier.await(30, TimeUnit.SECONDS)
                            DoubleArray(ts.size) { i -> op.apply(ts[i].t, cache) }
                        },
                    )
                }
                for ((tid, future) in futures.withIndex()) {
                    val got = future.get(60, TimeUnit.SECONDS)
                    for (i in ts.indices) {
                        assertBitwise(expected[i], got[i]) {
                            "Конкурентный кэш: поток #$tid, повтор #$iteration, uniform n=8, " +
                                "ядро V2exp, сплайн evalSpline: " + tsDescription(ts[i]) + "\n" +
                                "  последовательный некэшированный: " + fmt(expected[i]) + "\n" +
                                "  конкурентный кэшированный:       " + fmt(got[i]) + "\n" +
                                cacheContentsReport(op, grid, u, cache)
                        }
                    }
                }
                // Дополнительно: после гонки САМО содержимое кэша обязано совпадать с
                // прямым вычислением u в узлах — иначе «безвредность» гонки не доказана,
                // а совпадение интегралов могло бы оказаться случайным сокращением.
                val contents = cacheContentsReport(op, grid, u, cache)
                assertTrue(
                    contents.isEmpty(),
                    "После конкурентного заполнения содержимое кэша разошлось с прямым " +
                        "вычислением u (повтор #$iteration):\n$contents",
                )
            }
        } finally {
            pool.shutdownNow()
        }
    }

    /**
     * Точки ЗА верхней границей и в её окрестности: непокрытая ветка `included == bp.size`.
     *
     * Зачем отдельно от [boundaryTs]: при `t > b` в кэшированном пути счётчик `included`
     * упирается в ГРАНИЦУ МАССИВА (`included < bp.size`), а не в условие по `t`,
     * тогда как в `subBreakpoints` цикл завершается естественным концом `for`. Это
     * единственное место, где два пути останавливаются по РАЗНЫМ причинам, поэтому
     * тождество результатов требует прямой проверки. Точки `t > b` достижимы
     * на практике: шаблон конечной разности в `FirstKindSolver.deriv4` берёт `t + k*h`.
     */
    private fun beyondUpperBoundTs(grid: Grid): List<TCase> {
        val res = ArrayList<TCase>()
        val span = grid.b - grid.a
        for (d in listOf(1e-16, 1e-15, 1e-14, 1e-9, 1e-3, 0.25 * span, span)) {
            res.add(TCase(grid.b + d, "t=b+$d ЗА верхней границей (b=${grid.b})"))
        }
        return res
    }

    /**
     * Окрестность САМОЙ границы `b` СНИЗУ и соседнего узла `x_n`.
     *
     * [nodeThresholdTs] обходит только ВНУТРЕННИЕ узлы (`j in 1 until n`), так что
     * самая чувствительная точка — `b` минус величина порядка порога 1e-15 — не
     * проверялась. Именно здесь решается, вошла ли ПОСЛЕДНЯЯ ячейка целиком
     * или осталась усечённой шириной в единицы ulp.
     */
    private fun upperBoundNeighborhoodTs(grid: Grid): List<TCase> {
        val res = ArrayList<TCase>()
        for (d in listOf(1e-16, 1e-15, 1e-14)) {
            res.add(TCase(grid.b - d, "t=b-$d (b=${grid.b}, окрестность верхней границы)"))
        }
        res.add(TCase(Math.nextAfter(grid.b, Double.NEGATIVE_INFINITY), "t=nextDown(b) (ближайший double левее b)"))
        res.add(TCase(Math.nextAfter(grid.b, Double.POSITIVE_INFINITY), "t=nextUp(b) (ближайший double правее b)"))
        return res
    }

    /**
     * `t` ЗА верхней границей: ветка исчерпания массива узлов (`included == bp.size`).
     */
    @Test fun beyondUpperBound_cachedMatchesUncachedBitwise() {
        val stats = compareAll("точки за верхней границей t>b") { grid -> beyondUpperBoundTs(grid) }
        assertNonDegenerate(stats)
        // Все ячейки обязаны быть полными в КАЖДОЙ сверке, иначе набор t подобран неверно.
        assertTrue(
            stats.comparisons == stats.withFullCells,
            "Вырожденный набор: при t>b полными обязаны быть все ячейки, но из сверок " +
                "${stats.comparisons} таких лишь ${stats.withFullCells}",
        )
    }

    /** Окрестность самой верхней границы `b` — не покрытая [nodeThresholdTs]. */
    @Test fun upperBoundNeighborhood_cachedMatchesUncachedBitwise() {
        val stats = compareAll("окрестность верхней границы b") { grid -> upperBoundNeighborhoodTs(grid) }
        assertNonDegenerate(stats)
    }

    /**
     * ВЛОЖЕННЫЙ кэш: подынтегральная функция сама вычисляется через кэш.
     *
     * В решателе это достигается в `matrixM2`: `applyL(image)`, где `image = applyL(omega)`,
     * то есть внешний кэш заполняется ЗНАЧЕНИЯМИ КЭШИРОВАННОГО образа. Потенциальная
     * опасность: внутренний кэш вычисляется в узлах, а не в тех `t`, которые запрашивал
     * бы некэшированный путь, и ошибка здесь удвоила бысь по глубине вложения.
     * Оракул — ПОЛНОСТЬЮ некэшированное двойное применение оператора.
     */
    @Test fun nestedCache_cachedMatchesUncachedBitwise() {
        var comparisons = 0
        var nonZero = 0
        for (g in gridCases()) {
            val ts = boundaryTs(g.grid) + interiorTs(g.grid) +
                upperBoundNeighborhoodTs(g.grid) + beyondUpperBoundTs(g.grid)
            for (k in kernelCases) {
                val op = VolterraOperator(k.kernel, g.grid, quad)
                for (integrand in integrands(g.grid)) {
                    // Оракул: оба уровня без кэша.
                    val innerUncached = { s: Double -> op.apply(s, integrand.u) }
                    // Проверяемое: оба уровня через кэш (как в matrixM2).
                    val innerCache = op.integrandCache(integrand.u)
                    val innerCached = { s: Double -> op.apply(s, innerCache) }
                    val outerCache = op.integrandCache(innerCached)
                    for (tc in ts) {
                        val expected = op.apply(tc.t, innerUncached)
                        val actual = op.apply(tc.t, outerCache)
                        comparisons++
                        if (expected != 0.0) nonZero++
                        assertBitwise(expected, actual) {
                            "Вложенный кэш (как в matrixM2: applyL(applyL(omega))) разошёлся с полностью " +
                                "некэшированным двойным применением ПОБИТОВО.\n" +
                                "  сетка:   " + g.name + "\n" +
                                "  ядро:    " + k.name + "\n" +
                                "  функция: " + integrand.name + "\n" +
                                "  точка:   " + tsDescription(tc) + "\n" +
                                "  без кэша (оба уровня): " + fmt(expected) + "\n" +
                                "  с кэшем (оба уровня):  " + fmt(actual) + "\n" +
                                "  внутренний кэш против прямого u(узел):\n" +
                                cacheContentsReport(op, g.grid, integrand.u, innerCache)
                        }
                    }
                }
            }
        }
        assertTrue(comparisons > 0, "Вырожденный тест: ни одной сверки вложенного кэша")
        assertTrue(nonZero > 0, "Вырожденный тест: все значения вложенного кэша оказались нулевыми")
    }

    /**
     * Кэш привязан к ЭКЗЕМПЛЯРУ оператора, а не только к функции.
     *
     * Зачем. `IntegrandCache` — `inner class`, а в Kotlin тип `inner class` НЕ
     * параметризован внешним ЭКЗЕМПЛЯРОМ, поэтому `op2.apply(t, op1.integrandCache(u))`
     * компилируется. Без проверки владельца такой вызов давал бы молча неверные
     * числа (значения `u` — в узлах сетки `op1`, ядро — в узлах сетки `op2`) либо
     * `IndexOutOfBoundsException` при более грубой сетке владельца. Сегодня такой путь в
     * проекте не достижим, но перенос кода в общее ядро решателей сделал бы его таковым.
     *
     * Тест не вырожден: сначала доказывается, что сетки двух операторов ДЕЙСТВИТЕЛЬНО
     * разные и что оба оператора поотдельности работоспособны со СВОИМ кэшем
     * (иначе исключение могло бы бросаться по любой другой причине).
     */
    @Test fun cacheIsBoundToOwningOperator_foreignCacheRejected() {
        val gridFine = Grid.uniform(16)
        val gridCoarse = Grid.uniform(4)
        val u = { s: Double -> Math.sin(3.0 * s) * Math.exp(-0.5 * s) }
        val opFine = VolterraOperator(VolterraProblem.V2.kernel, gridFine, quad)
        val opCoarse = VolterraOperator(VolterraProblem.V2.kernel, gridCoarse, quad)

        // Сетки обязаны быть разными — иначе дефект, от которого защищаемся, был бы мнимым.
        assertTrue(
            gridFine.breakpoints.size != gridCoarse.breakpoints.size,
            "Вырожденный тест: сетки операторов совпадают, подмена кэша была бы безвредна",
        )
        val probe = 0.5 * (gridFine.a + gridFine.b)
        // Каждый оператор со СВОИМ кэшем работает и даёт правильные числа.
        assertBitwise(opFine.apply(probe, u), opFine.apply(probe, opFine.integrandCache(u))) {
            "Оператор со СВОИМ кэшем должен работать (gridFine)"
        }
        assertBitwise(opCoarse.apply(probe, u), opCoarse.apply(probe, opCoarse.integrandCache(u))) {
            "Оператор со СВОИМ кэшем должен работать (gridCoarse)"
        }

        // А вот ЧУЖОЙ кэш обязан быть отвергнут внятным исключением — в ОБЕ стороны.
        for ((name, pair) in listOf(
            "грубый кэш в тонкий оператор" to (opFine to opCoarse),
            "тонкий кэш в грубый оператор" to (opCoarse to opFine),
        )) {
            val (target, foreign) = pair
            val foreignCache = foreign.integrandCache(u)
            val error = try {
                target.apply(probe, foreignCache)
                null
            } catch (e: IllegalArgumentException) {
                e
            }
            if (error == null) {
                fail(
                    "Передача ЧУЖОГО кэша не была отвергнута ($name): вызов завершился без " +
                        "IllegalArgumentException. Это и есть латентный дефект: значения u берутся " +
                        "в узлах сетки владельца кэша, а ядро — в узлах сетки вызываемого оператора, " +
                        "то есть результат молча неверен",
                )
            }
            val message = error.message ?: ""
            assertTrue(
                message.contains("VolterraOperator") && message.contains("u(s)"),
                "Сообщение об ошибке невнятное ($name): оно обязано называть класс оператора " +
                    "и объяснять причину (узлы разных сеток). Получено: \"$message\"",
            )
        }
    }

    /**
     * Согласованность путей при `t = NaN`.
     *
     * Некэшированный путь даёт NaN: в `GaussLegendre.integrate` условие `hi <= lo`
     * для NaN ложно, поэтому счёт продолжается и NaN распространяется в сумму.
     * Кэшированный путь использует ТОЖЕ отрицание (`!(t <= lo)`), а не `t > lo`,
     * именно для совпадения ветвления: вариант `t > lo` вернул бы ровно `0.0`,
     * то есть ПРАВДОПОДОБНОЕ число вместо сигнала о плохом входе. Проверяется
     * именно согласованность (оба NaN), а не биты: полезная нагрузка NaN не
     * специфицирована арифметикой и требовать её совпадения было бы чрезмерно.
     */
    @Test fun nanArgument_bothPathsAgree() {
        var checks = 0
        for (g in gridCases()) {
            for (k in kernelCases) {
                val op = VolterraOperator(k.kernel, g.grid, quad)
                for (integrand in integrands(g.grid)) {
                    val cache = op.integrandCache(integrand.u)
                    val uncached = op.apply(Double.NaN, integrand.u)
                    val cached = op.apply(Double.NaN, cache)
                    checks++
                    assertTrue(
                        uncached.isNaN(),
                        "Ожидалось, что некэшированный путь при t=NaN даст NaN " +
                            "(сетка ${g.name}, ядро ${k.name}, ${integrand.name}), получено ${fmt(uncached)}",
                    )
                    assertTrue(
                        cached.isNaN(),
                        "Кэшированный путь разошёлся с некэшированным при t=NaN " +
                            "(сетка ${g.name}, ядро ${k.name}, ${integrand.name}): " +
                            "без кэша ${fmt(uncached)}, с кэшем ${fmt(cached)}. " +
                            "Вероятная причина: условие усечённой ячейки записано как `t > lo` " +
                            "вместо отрицания `!(t <= lo)`, и при NaN ветка молча пропускается",
                    )
                }
            }
        }
        assertTrue(checks > 0, "Вырожденный тест: ни одной проверки NaN не выполнено")
    }

    // --- Механика сверки --------------------------------------------------------

    private class Stats {
        var comparisons = 0
        var withFullCells = 0
        var nonZero = 0
    }

    /**
     * Прогоняет все сочетания (сетка × ядро × подынтегральная функция × t) и сверяет
     * побитово. Кэш создаётся ОДИН на сочетание (сетка, ядро, функция) и обслуживает все
     * `t` — ровно так он живёт в `SecondKindSolver.applyL`.
     */
    private fun compareAll(scenario: String, tsOf: (Grid) -> List<TCase>): Stats {
        val stats = Stats()
        for (g in gridCases()) {
            val ts = tsOf(g.grid)
            for (k in kernelCases) {
                val op = VolterraOperator(k.kernel, g.grid, quad)
                for (integrand in integrands(g.grid)) {
                    val cache = op.integrandCache(integrand.u)
                    for (tc in ts) {
                        val expected = op.apply(tc.t, integrand.u)
                        val actual = op.apply(tc.t, cache)
                        stats.comparisons++
                        if (fullCellCount(g.grid, tc.t) > 0) stats.withFullCells++
                        if (expected != 0.0) stats.nonZero++
                        assertBitwise(expected, actual) {
                            report(scenario, g, k, integrand, tc, expected, actual, op, cache)
                        }
                    }
                }
            }
        }
        return stats
    }

    /**
     * Защита от вырождения самой проверки: если бы все сверяемые значения оказались нулями
     * или ни одна конфигурация не задействовала кэш, тест был бы зелёным ни о чём.
     */
    private fun assertNonDegenerate(stats: Stats) {
        assertTrue(stats.comparisons > 0, "Вырожденный тест: ни одной сверки не выполнено")
        assertTrue(
            stats.withFullCells > 0,
            "Вырожденный тест: ни в одной сверке не было полных ячеек, то есть кэш не задействован",
        )
        assertTrue(
            stats.nonZero > 0,
            "Вырожденный тест: все сверяемые значения оказались нулевыми (сверок ${stats.comparisons})",
        )
    }

    /**
     * Побитовое сравнение.
     *
     * Сравниваются RAW-биты, а не значения через `==`. Причина выбора: `==` для double
     * тоже различает числа побитово, НО объявляет `0.0 == -0.0` (разные знаки нуля —
     * это уже разный результат накопления, его нельзя пропускать) и `NaN != NaN`
     * (совпавший «плохой» результат обоих путей — не расхождение путей). Сравнение
     * `doubleToRawLongBits` свободно от обеих оговорок. Допуск не используется
     * принципиально: смысл проверки — отсутствие ЛЮБОГО, даже однобитового, расхождения.
     */
    private fun assertBitwise(expected: Double, actual: Double, message: () -> String) {
        if (!sameBits(expected, actual)) fail(message())
    }

    private fun sameBits(x: Double, y: Double): Boolean =
        java.lang.Double.doubleToRawLongBits(x) == java.lang.Double.doubleToRawLongBits(y)

    private fun bits(x: Double): String =
        "0x" + java.lang.Long.toHexString(java.lang.Double.doubleToRawLongBits(x))

    private fun fmt(x: Double): String = "$x [${bits(x)}]"

    /**
     * Число ПОЛНЫХ ячеек сетки в разбиении [a,t] (повторяет отбор из обоих путей).
     *
     * Порог берётся из САМОГО оператора (он относительный и зависит от масштаба
     * отрезка), а не зашивается константой: иначе диагностика и проверки
     * невырожденности врали бы на сетках большого масштаба.
     */
    private fun fullCellCount(grid: Grid, t: Double): Int {
        if (t <= grid.a) return 0
        val bp = grid.breakpoints
        val eps = VolterraOperator(kernelCases[0].kernel, grid, quad).breakpointInclusionEps
        var included = 0
        while (included < bp.size && bp[included] < t - eps) included++
        return maxOf(0, included - 1)
    }

    private fun tsDescription(tc: TCase): String = "${tc.label}, t=${fmt(tc.t)}"

    // --- Диагностика ------------------------------------------------------------

    /**
     * Диагностическое сообщение о расхождении. Обязано отвечать на вопросы: на каком `t`,
     * на какой сетке, с каким ядром и какой функцией, в какой ячейке и каком узле, и
     * каковы ОБА значения. Строится только при падении, поэтому может быть подробным.
     */
    private fun report(
        scenario: String,
        g: GridCase,
        k: KernelCase,
        integrand: IntegrandCase,
        tc: TCase,
        expected: Double,
        actual: Double,
        op: VolterraOperator,
        cache: VolterraOperator.IntegrandCache,
    ): String {
        val ulp = Math.ulp(expected)
        val deltaUlps = if (ulp > 0.0) (actual - expected) / ulp else Double.NaN
        val sb = StringBuilder()
        sb.append("Кэшированный путь VolterraOperator.apply(t, cache) разошёлся с некэшированным ")
            .append("apply(t, u) ПОБИТОВО.\n")
        sb.append("  сценарий:      ").append(scenario).append('\n')
        sb.append("  сетка:         ").append(g.name)
            .append(" [a=").append(g.grid.a).append(", b=").append(g.grid.b).append("]\n")
        sb.append("  ядро:          ").append(k.name).append('\n')
        sb.append("  функция u:     ").append(integrand.name).append('\n')
        sb.append("  точка:         ").append(tsDescription(tc)).append('\n')
        sb.append("  полных ячеек:  ").append(fullCellCount(g.grid, tc.t)).append('\n')
        sb.append("  ожидалось (без кэша): ").append(fmt(expected)).append('\n')
        sb.append("  получено (с кэшем):   ").append(fmt(actual)).append('\n')
        sb.append("  разница:       ").append(actual - expected)
            .append(" = ").append(deltaUlps).append(" ulp\n")
        val contents = cacheContentsReport(op, g.grid, integrand.u, cache)
        if (contents.isNotEmpty()) {
            sb.append("  содержимое кэша:\n").append(contents)
        } else {
            sb.append("  содержимое кэша совпадает с прямым u(узел) во всех ячейках и узлах,\n")
            sb.append("  значит расхождение возникло в АРИФМЕТИКЕ накопления суммы ")
                .append("(порядок множителей/слагаемых).\n")
            sb.append(cellLocalization(op, g.grid, integrand.u, cache))
        }
        return sb.toString()
    }

    /**
     * Сверяет СОДЕРЖИМОЕ кэша с прямым вычислением `u` в узлах полных ячеек. Узлы
     * воспроизводятся по той же формуле, что и в квадратуре (`mid + half * refNodes[q]`),
     * поэтому расхождение здесь означает либо иные узлы, либо иные значения.
     *
     * @return пустая строка, если всё совпадает; иначе описание первых расхождений.
     */
    private fun cacheContentsReport(
        op: VolterraOperator,
        grid: Grid,
        u: (Double) -> Double,
        cache: VolterraOperator.IntegrandCache,
    ): String {
        val bp = grid.breakpoints
        val sb = StringBuilder()
        var reported = 0
        for (c in 0 until bp.size - 1) {
            val lo = bp[c]
            val hi = bp[c + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            val cached = cache.values(c)
            for (q in refNodes.indices) {
                val s = mid + half * refNodes[q]
                val direct = u(s)
                if (!sameBits(direct, cached[q])) {
                    if (reported < 5) {
                        sb.append("    ячейка c=").append(c)
                            .append(" [").append(lo).append(", ").append(hi).append("], узел q=").append(q)
                            .append(", s=").append(fmt(s)).append('\n')
                            .append("      прямое u(s) = ").append(fmt(direct)).append('\n')
                            .append("      из кэша     = ").append(fmt(cached[q])).append('\n')
                    }
                    reported++
                }
            }
        }
        if (reported > 5) sb.append("    ... и ещё ").append(reported - 5).append(" расхождений\n")
        return sb.toString()
    }

    /**
     * Локализация ячейки, начиная с которой пути расходятся, когда содержимое кэша верно
     * (случай перестановки множителей/слагаемых). Приём: берём `t` в середине ячейки `m`,
     * тогда полными оказываются ровно ячейки `0..m-1`. Наименьшее `m`, при котором пути
     * разошлись, указывает на ячейку `m-1` как на первую испорченную.
     */
    private fun cellLocalization(
        op: VolterraOperator,
        grid: Grid,
        u: (Double) -> Double,
        cache: VolterraOperator.IntegrandCache,
    ): String {
        val bp = grid.breakpoints
        for (m in 1 until bp.size - 1) {
            val probe = 0.5 * (bp[m] + bp[m + 1])
            val expected = op.apply(probe, u)
            val actual = op.apply(probe, cache)
            if (!sameBits(expected, actual)) {
                return "  первая испорченная ПОЛНАЯ ячейка: c=" + (m - 1) +
                    " [" + bp[m - 1] + ", " + bp[m] + "]\n" +
                    "  (проба t=" + fmt(probe) + " с ровно " + m + " полными ячейками: " +
                    "без кэша " + fmt(expected) + ", с кэшем " + fmt(actual) + ")\n"
            }
        }
        return "  локализовать ячейку пробами не удалось: на пробных t (середины ячеек) оба пути\n" +
            "  дали одинаковые биты — расхождение либо приходится на усечённую ячейку [x_last, t],\n" +
            "  либо на пробах случайно сократилось при округлении\n"
    }
}
