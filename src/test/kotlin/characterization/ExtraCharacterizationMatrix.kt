package characterization

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.SolutionFunc
import numerics.functionals.AveragingFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import java.util.Locale

/**
 * ЕДИНЫЙ ИСТОЧНИК дополнительной характеризационной матрицы (снимок `baseline-extra.tsv`).
 *
 * Зачем отдельная матрица, если уже есть `baseline-eh.tsv`. Действующий эталон покрывает
 * только схемы `base`/`sloan`/`kulkarni`/`iteratedKulkarni`/`nystrom`/`iteratedNystrom`
 * на сетке [Grid.uniform] и отрезке `[0,1]`. Вне покрытия остаются ровно те места,
 * которые сильнее всего рискуют пострадать при выносе общего кода решателей
 * Фредгольма и Вольтерры в общий базовый класс:
 *
 *  1. [solvers.fredholm.FredholmSecondKindSolver.combinedNystrom] и
 *     [solvers.volterra.VolterraSecondKindSolver.combinedNystrom] — у них РАЗНЫЕ критерии
 *     останова простой итерации: Фредгольм меряет расхождение в гауссовых узлах
 *     `op.gNode` (`FredholmSolver.kt`), Вольтерра — в `4n+1` равномерных контрольных
 *     точках (`VolterraSolver.kt`). Слияние тел в общее ядро изменит число итераций,
 *     а значит и результат;
 *  2. неравномерные сетки [Grid.quasiUniform], [Grid.geometric], [Grid.graded];
 *  3. отрезки, отличные от `[0,1]` — на них включается масштабирование порога
 *     [Grid.breakpointInclusionEps], введённое на этапе 3.
 *
 * Почему матрица описана ОДИН раз и используется и инструментом снятия
 * ([ExtraBaselineSnapshotTool]), и проверкой ([ExtraCharacterizationTest]): пара
 * «BaselineSnapshotTool + EhCharacterizationTest» дублирует перечисление сочетаний,
 * и любая правка одного файла без второго молча выводит эталон из-под проверки.
 * Здесь такой рассинхронизации не может быть по построению.
 *
 * СОСТАВ (см. [collect]):
 *  - уравнения: Фредгольм (`F2`) и Вольтерра (`V2`) — рациональные задачи, решение
 *    которых не лежит ни в одной порождающей системе (то есть числа не вырождаются
 *    в машинный ноль и пригодны для относительного сравнения);
 *  - схемы: `combinedNystrom`, `iteratedCombinedNystrom` (целевые) плюс `base` и
 *    `kulkarni` (контрольные: ловят порчу сборки матриц `M`/`M2`);
 *  - порождающие системы: B, H, T;
 *  - семейства функционалов: `theta`, `mu`, `lambda` — ТОЛЬКО без производной.
 *    Ограничение не выбрано, а закреплено кодом: `nystromSupport()` начинается с
 *    `require(!funcs.usesDerivative)` (`FredholmSolver.kt:265`, симметрично в
 *    `VolterraSolver.kt`), поэтому семейства `xi0`/`xi1`/`xi2` для Nyström-схем
 *    невалидны и в матрицу не входят;
 *  - сетки: `uniform`, `quasiUniform`, `geometric`, `graded`;
 *  - n: 8 и 16;
 *  - отрезки: `[0,1]` (полная матрица) и `[0,2]` (сокращённый набор, см. [collect]).
 *
 * ПОЧЕМУ `[0,2]`, А НЕ `[-1,1]`: ядро задач `F2`/`V2` равно `1/(1+t+s)` и на `[-1,1]`
 * имеет полюс при `t+s = -1`, то есть прямо внутри области интегрирования. Снимок
 * фиксировал бы не поведение решателя, а деление на ноль. На `[0,2]` знаменатель
 * лежит в `[1,5]` — особенности нет, а масштаб отрезка удвоен, чего достаточно для
 * срабатывания относительного порога [Grid.breakpointInclusionEps].
 *
 * РЕЖИМ ДИАГНОСТИКИ: решатели создаются с `throwOnDivergence = false`. Это ОСОЗНАННО.
 * НА ТЕКУЩЕЙ МАТРИЦЕ ВСЁ СХОДИТСЯ: все 336 значений `*.iters` лежат в множестве
 * {13, 15, 47, 107}, предела 200 не достигает ни одно сочетание, включая отрезок `[0,2]`.
 * Режим выбран не потому, что расходимость наблюдается, а потому, что она НЕ ДОЛЖНА
 * превращать снимок в бесполезный: при поведении по умолчанию (исключение) первое же
 * несошедшееся сочетание зафиксировало бы факт исключения — ОДИН бит информации вместо
 * числа — и обрушило бы снятие эталона целиком. В режиме диагностики фиксируются и
 * достигнутое значение, и число итераций, поэтому сеть останется чувствительной, если
 * будущая правка сдвинет какое-то сочетание за предел сходимости.
 *
 * ЧИСЛО ИТЕРАЦИЙ СНИМАЕТСЯ ЯВНО (ключи `*.combNystrom.iters`) и это ключевой элемент
 * сети: если при слиянии решателей критерий останова одного из них подменить критерием
 * другого, но итог случайно совпадёт по числу итераций, значение E_h не изменится —
 * именно счётчик итераций делает такую подмену видимой.
 */
object ExtraCharacterizationMatrix {

    /** Путь к ресурсу с эталонным снимком дополнительной матрицы. */
    const val RESOURCE_PATH = "/characterization/baseline-extra.tsv"

    /** Относительный допуск сравнения с эталоном (тот же, что и у основного гейта). */
    const val RELATIVE_TOLERANCE = 1e-9

    /**
     * Абсолютный «пол» сравнения: значения ниже него считаются нулевыми.
     * Нужен по той же причине, что и в [EhCharacterizationTest]: у величин, лежащих
     * на уровне машинного нуля, относительное сравнение бессмысленно.
     */
    const val ABSOLUTE_FLOOR = 1e-11

    /**
     * Относительный допуск для ключей E_h, ОБА значения которых лежат ниже [ABSOLUTE_FLOOR].
     *
     * Зачем понадобился отдельный режим. Раньше такая пара просто ПРОПУСКАЛАСЬ
     * (`continue`), и это было дырой ровно того же вида, что уже закрыта для ключей
     * невязки: под пол `1e-11` попадают 53 из 672 ключей E_h, и среди них — 3 ключа
     * `combNystrom` и 21 `iterCombNystrom`, то есть ЦЕЛЕВЫЕ схемы этапа. Изменение
     * `2.7e-12` → `9e-12` (втрое) проходило молча.
     *
     * Почему допуск НЕ `1e-9`, как у обычных ключей. Значения этого диапазона
     * (наблюдаемый минимум `3.3e-14` при самих решениях порядка единицы) — это разность
     * почти совпавших чисел, то есть катастрофическое сокращение: сдвиг операндов на один
     * ULP (`~2.2e-16`) даёт здесь относительное изменение до нескольких процентов.
     * Допуск `1e-9` был бы строже машинной точности и давал бы ложные падения.
     *
     * Почему `1e-3`. Тот же порог, что и у ключей невязки, и по той же причине: он лежит
     * на порядки ниже любого содержательного эффекта ошибки переноса (мутационная проверка
     * сдвигала значение ВТРОЕ, то есть на 200%) и на порядки выше шума хранения.
     */
    const val SMALL_VALUE_RELATIVE_TOLERANCE = 1e-3

    /**
     * Абсолютный «пол» для ключей E_h, попавших в режим [SMALL_VALUE_RELATIVE_TOLERANCE].
     *
     * Лежит на два порядка ниже наблюдаемого минимума E_h (`3.3e-14`) и служит только
     * защитой от деления на ноль при точном нуле ошибки.
     */
    const val SMALL_VALUE_ABSOLUTE_FLOOR = 1e-16

    /** Порядок квадратуры Гаусса — тот же, что и в основном эталоне. */
    private const val QUADRATURE_ORDER = 8

    /**
     * Суффикс ключей, хранящих ДОСТИГНУТУЮ НЕВЯЗКУ итерации (`SolutionFunc.residual`).
     *
     * Ради чего эти ключи существуют. Невязка — ровно та величина, которую меряет критерий
     * останова: равномерная норма разности соседних итерантов НА КОНТРОЛЬНОМ МНОЖЕСТВЕ.
     * У решателей это множество РАЗНОЕ (Фредгольм — гауссовы узлы `op.gNode`, Вольтерра —
     * `4n+1` равномерных точек), и подмена одного другим при слиянии решателей меняет
     * именно её.
     *
     * Это установлено ЗАМЕРОМ, а не предположением. Мутация «контрольные точки Вольтерры
     * заменены гауссовыми узлами» была внесена и проверена: E_h и число итераций
     * НЕ ИЗМЕНИЛИСЬ НИ НА БИТ (итерация сходится за то же число шагов и приходит в ту же
     * точку — оба множества достаточно плотные, а сходимость линейная). Без ключей
     * невязки сеть эту мутацию пропускала бы молча — то есть была бы бесполезна ровно
     * в том месте, ради которого создавалась.
     */
    const val RESIDUAL_SUFFIX = ".residual"

    /**
     * Допуск сравнения невязок — 1e-3 (относительный), а не общий 1e-9.
     *
     * Почему НЕ 1e-9. Невязка — разность почти совпавших итерантов (порядка 3.6e-14
     * при самих значениях порядка 1), то есть катастрофическое сокращение. Возмущение
     * операндов всего на один ULP (~2.2e-16) даёт в ней относительное изменение около
     * 2.2e-16/3.6e-14 ~ 0.6%, то есть допуск 1e-9 здесь был бы строже машинной точности.
     *
     * Почему ИМЕННО 1e-3 — по замеру, а не по вкусу. Мутация «контрольные точки Вольтерры
     * `4n+1` → гауссовы узлы» изменила 168 из 336 ключей невязки (все вольтерровы;
     * фредгольмовы не тронуты, как и должно быть), причём ВСЕ 168 — больше чем на 1e-3
     * (медиана изменённых ~1.5%, максимум 4.0%). Порог 1e-3 лежит ниже всего
     * эффекта мутации и на 13 порядков выше сдвига САМОГО ЗНАЧЕНИЯ невязки на 1 ULP
     * (~1e-16), то есть разделяет сигнал и шум хранения.
     *
     * ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ СОПРОВОЖДАЮЩЕМУ. Ключи `*.residual` НАМЕРЕННО
     * ГИПЕРЧУВСТВИТЕЛЬНЫ: из-за сокращения они усиливают различие в последних битах
     * примерно в 1e13 раз. Если рефакторинг математически нейтрален, но НЕ побитово
     * тождествен (например, сменился порядок сложения), эти ключи могут упасть ПРИ
     * ЧИСТЫХ ключах E_h. Правильная реакция в таком случае — убедиться, что (1) все
     * ключи E_h и `*.iters` совпали и (2) контрольные множества обоих решателей остались
     * разными, и после этого ОСОЗНАННО переснять снимок, а НЕ ослаблять этот порог.
     */
    const val RESIDUAL_RELATIVE_TOLERANCE = 1e-3

    /**
     * Абсолютный «пол» СПЕЦИАЛЬНО для ключей невязки: 1e-18.
     *
     * Общий [ABSOLUTE_FLOOR] = 1e-11 здесь НЕПРИМЕНИМ и был бы СКРЫТОЙ ДЫРОЙ в сети.
     * Критерий останова итерации равен 1e-13, поэтому ЛЮБАЯ сошедшаяся невязка по
     * построению меньше 1e-13, то есть всегда ниже 1e-11 — с общим полом ВСЕ ключи
     * невязки признавались бы «нулёвыми» и не сравнивались вовсе. Это не догадка:
     * первая редакция теста именно так и ПРОПУСТИЛА мутацию (а) на зелёном прогоне,
     * хотя числа в снимке уже расходились.
     *
     * 1e-18 лежит на четыре порядка ниже наблюдаемых невязок (~3.6e-14) и служит
     * только защитой от деления на ноль при точном нуле невязки.
     */
    const val RESIDUAL_ABSOLUTE_FLOOR = 1e-18

    /** Описание отрезка интегрирования: короткий тег для ключа и сами границы. */
    private data class Segment(val tag: String, val a: Double, val b: Double)

    /** Описание фабрики сетки: короткий тег для ключа и построитель. */
    private data class GridKind(val tag: String, val build: (Int, Double, Double) -> Grid)

    private val gridKinds = listOf(
        GridKind("uniform") { n, a, b -> Grid.uniform(n, a, b) },
        GridKind("quasi") { n, a, b -> Grid.quasiUniform(n, a, b) },
        GridKind("geom") { n, a, b -> Grid.geometric(n, a, b) },
        GridKind("graded") { n, a, b -> Grid.graded(n, a, b) },
    )

    private val systems = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)

    /**
     * Семейства функционалов БЕЗ производной — единственные, применимые к Nyström-схемам.
     * Инвариант проверяется в [collect] явной проверкой `usesDerivative`.
     */
    private val familyNames = listOf("theta", "mu", "lambda")

    private val sizes = listOf(8, 16)

    private val unitSegment = Segment("s01", 0.0, 1.0)
    private val scaledSegment = Segment("s02", 0.0, 2.0)

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "mu" -> AveragingFunctionals(basis)
        "lambda" -> ThreePointFunctionals(basis)
        else -> error("ExtraCharacterizationMatrix: неизвестное семейство '$name'")
    }

    /**
     * Печатное представление значения снимка.
     *
     * 12 значащих цифр — как в [BaselineSnapshotTool]: с запасом относительно допуска
     * 1e-9 и без хранения незначащего шума последних битов. Локаль [Locale.ROOT]
     * задана явно: `"%.12g".format(x)` использует локаль по умолчанию и на машине с
     * русской локалью пишет запятую вместо точки, после чего снимок перестаёт читаться.
     */
    fun formatValue(value: Double): String = when {
        value.isNaN() -> "NaN"
        value.isInfinite() -> if (value > 0) "Infinity" else "-Infinity"
        else -> String.format(Locale.ROOT, "%.12g", value)
    }

    /**
     * Вычисляет одно значение снимка, ПЕРЕХВАТЫВАЯ отказ.
     *
     * Снимок характеризационный: если сочетание сегодня падает, фиксируется сам факт
     * отказа (`ERROR:<класс исключения>`), а не «исправляется» задача. Это сохраняет
     * сочетание в сети: превращение отказа в число (или в другое исключение) после
     * рефакторинга будет замечено так же надёжно, как изменение числа.
     */
    private fun evaluate(compute: () -> Double): String = try {
        formatValue(compute())
    } catch (e: Throwable) {
        "ERROR:" + (e::class.simpleName ?: e::class.java.name)
    }

    /**
     * Собирает всю матрицу в виде отсортированного по ключу списка пар «ключ - значение».
     *
     * Сортировка обязательна: она делает и файл снимка, и его дифф осмысленными
     * независимо от порядка обхода циклов.
     */
    fun collect(): List<Pair<String, String>> {
        val rows = mutableListOf<Pair<String, String>>()

        // Полная матрица на единичном отрезке.
        for (segment in listOf(unitSegment)) {
            for (kind in gridKinds) {
                for (system in systems) {
                    for (familyName in familyNames) {
                        for (n in sizes) {
                            collectFredholm(rows, segment, kind, system, familyName, n)
                            collectVolterra(rows, segment, kind, system, familyName, n)
                        }
                    }
                }
            }
        }

        // Сокращённый набор на отрезке [0,2]: цель — поймать масштабно-зависимые ошибки
        // (порог включения точки разбиения, нормировки шага), а не повторить всю матрицу.
        // Достаточно двух сеток (равномерной как контрольной и геометрической как
        // существенно неравномерной) и одного семейства.
        for (kind in gridKinds.filter { it.tag == "uniform" || it.tag == "geom" }) {
            for (system in systems) {
                for (n in sizes) {
                    collectFredholm(rows, scaledSegment, kind, system, "theta", n)
                    collectVolterra(rows, scaledSegment, kind, system, "theta", n)
                }
            }
        }

        return rows.sortedBy { it.first }
    }

    private fun keyPrefix(
        equation: String,
        problemName: String,
        system: GeneratingSystem,
        familyName: String,
        kind: GridKind,
        segment: Segment,
        n: Int,
    ): String = "$equation.$problemName.${system.name}.$familyName.${kind.tag}.${segment.tag}.n$n"

    /** Схемы решателя Фредгольма для одного сочетания параметров. */
    private fun collectFredholm(
        rows: MutableList<Pair<String, String>>,
        segment: Segment,
        kind: GridKind,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ) {
        val problem = problems.fredholm.FredholmProblem.F2
        val grid = kind.build(n, segment.a, segment.b)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        check(!funcs.usesDerivative) {
            "Семейство '${funcs.name}' использует производную и несовместимо с Nyström-схемами"
        }
        val op = solvers.fredholm.FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
        val solver = solvers.fredholm.FredholmSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            throwOnDivergence = false,
        )
        val exact = { t: Double -> problem.exact(t) }
        val prefix = keyPrefix("F", problem.name, system, familyName, kind, segment, n)
        emitSchemes(rows, prefix, grid, exact) { scheme ->
            when (scheme) {
                Scheme.BASE -> solver.base()
                Scheme.KULKARNI -> solver.kulkarni()
                Scheme.COMBINED_NYSTROM -> solver.combinedNystrom()
                Scheme.ITERATED_COMBINED_NYSTROM -> solver.iteratedCombinedNystrom()
            }
        }
    }

    /** Схемы решателя Вольтерры для одного сочетания параметров. */
    private fun collectVolterra(
        rows: MutableList<Pair<String, String>>,
        segment: Segment,
        kind: GridKind,
        system: GeneratingSystem,
        familyName: String,
        n: Int,
    ) {
        val problem = problems.volterra.VolterraProblem.V2
        val grid = kind.build(n, segment.a, segment.b)
        val basis = MinimalSplineBasis(system, grid)
        val funcs = family(familyName, basis)
        check(!funcs.usesDerivative) {
            "Семейство '${funcs.name}' использует производную и несовместимо с Nyström-схемами"
        }
        val op = solvers.volterra.VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
        val solver = solvers.volterra.VolterraSecondKindSolver(
            basis, funcs, op, 1.0,
            solvers.core.RhsWithDerivatives(
                { t -> problem.rhsExact(t, op) },
                { t -> problem.rhsExactDeriv(t, op) },
                { t -> problem.rhsExactDeriv2(t, op) },
            ),
            throwOnDivergence = false,
        )
        val exact = { t: Double -> problem.exact(t) }
        val prefix = keyPrefix("V", problem.name, system, familyName, kind, segment, n)
        emitSchemes(rows, prefix, grid, exact) { scheme ->
            when (scheme) {
                Scheme.BASE -> solver.base()
                Scheme.KULKARNI -> solver.kulkarni()
                Scheme.COMBINED_NYSTROM -> solver.combinedNystrom()
                Scheme.ITERATED_COMBINED_NYSTROM -> solver.iteratedCombinedNystrom()
            }
        }
    }

    /** Схемы, снимаемые для каждого сочетания параметров. */
    private enum class Scheme(val tag: String) {
        BASE("base"),
        KULKARNI("kulkarni"),
        COMBINED_NYSTROM("combNystrom"),
        ITERATED_COMBINED_NYSTROM("iterCombNystrom"),
    }

    /**
     * Снимает E_h всех схем и дополнительно число итераций у итерационных схем Nyström.
     *
     * Решение задаётся функцией [solve], а не готовым списком. Общая база у решателей
     * теперь есть (`SecondKindSolverCore`), но вызываемые здесь схемы `combinedNystrom`
     * и `iteratedCombinedNystrom` в неё СОЗНАТЕЛЬНО НЕ перенесены (разные веса Nyström
     * и разные критерии останова), то есть объявлены в наследниках независимо.
     * Поэтому замыкание остаётся единственным способом объединить их в одной матрице —
     * и это ПРАВИЛЬНО: сеть должна звать именно те реализации, которые сторожит.
     */
    private fun emitSchemes(
        rows: MutableList<Pair<String, String>>,
        prefix: String,
        grid: Grid,
        exact: (Double) -> Double,
        solve: (Scheme) -> SolutionFunc,
    ) {
        for (scheme in Scheme.entries) {
            var iterations: Int? = null
            var residual: Double? = null
            val value = evaluate {
                val solution = solve(scheme)
                iterations = solution.iterations
                residual = solution.residual
                errorEh(exact, solution.eval, grid)
            }
            rows += "$prefix.${scheme.tag}" to value
            // Счётчик итераций и невязка осмыслены только для итерационных схем; у прямых
            // они тождественно равны 0 и только зашумляют дифф.
            //
            // ОГОВОРКА О ЗАВИСИМОСТИ КЛЮЧЕЙ. Для ITERATED_COMBINED_NYSTROM пары
            // `.iters`/`.residual` ПОБИТОВО РАВНЫ соответствующим ключам COMBINED_NYSTROM:
            // `iteratedCombinedNystrom` наследует эти поля от исходного решения без
            // пересчёта (одно интегрирование, собственных итераций нет). То есть 168 пар
            // из 1344 значений НЕ дают независимого покрытия — независимых значений в сети
            // 1176, и заявлять 1344 как меру независимого покрытия неверно.
            // Ключи всё же снимаются: они фиксируют САМ ФАКТ наследования. Если правка
            // заставит `iteratedCombinedNystrom` пересчитывать критерий останова заново
            // (например, при попытке слить его с `combinedNystrom`), равенство нарушится
            // и тест это покажет — а по одному лишь E_h такая правка может пройти молча.
            val iterationsMatter =
                scheme == Scheme.COMBINED_NYSTROM || scheme == Scheme.ITERATED_COMBINED_NYSTROM
            if (iterationsMatter) {
                val recordedIterations = iterations
                rows += "$prefix.${scheme.tag}.iters" to
                    (recordedIterations?.toString() ?: "ERROR:NoIterations")
                val recordedResidual = residual
                rows += "$prefix.${scheme.tag}$RESIDUAL_SUFFIX" to
                    (recordedResidual?.let { formatValue(it) } ?: "ERROR:NoResidual")
            }
        }
    }
}
