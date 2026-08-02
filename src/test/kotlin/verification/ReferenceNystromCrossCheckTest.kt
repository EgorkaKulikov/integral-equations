package verification

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.functionals.ProjFunctionals
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * СВЕРКА СХЕМ ПРОЕКТА С НЕЗАВИСИМЫМ ЭТАЛОНОМ [ReferenceNystromSolver].
 *
 * Зачем. Проверки, сравнивающие схемы проекта между собой, не обнаруживают ошибку,
 * общую для всех схем: она сократится. Здесь эталоном служит реализация, не
 * использующая НИ ОДНОГО элемента численного ядра проекта (см. KDoc
 * [ReferenceNystromSolver]): ни сплайнов, ни функционалов, ни квадратуры, ни
 * решателя СЛАУ. Поэтому расхождение указывает на дефект, а согласие — свидетельство,
 * не замкнутое на проверяемый код.
 *
 * Что именно проверяется, в двух шагах — иначе результат нельзя истолковать:
 *
 *  1. ПРИГОДНОСТЬ ЭТАЛОНА. Эталон обязан воспроизводить точное решение модельной
 *     задачи на уровне, на порядки лучшем схем проекта. Если он этого не делает,
 *     сверять им нельзя, и шаг 2 бессмыслен ([referenceReproducesExactSolution]).
 *  2. СВЕРКА СХЕМ. Решения проекта отклоняются от эталона не больше, чем от точного
 *     решения (с запасом на то, что эталон сам не идеален) —
 *     [projectSchemesAgreeWithReferenceNystrom].
 *
 * Тег `fast`: эталон на 64 узлах — одно обращение матрицы 64x64, схемы проекта берутся
 * на n = 8..32, весь класс укладывается в доли секунды.
 */
@Tag("fast")
class ReferenceNystromCrossCheckTest {

    private companion object {
        /** Отрезок задачи; совпадает с областью определения модельных задач. */
        const val A = 0.0
        const val B = 1.0

        /** Сетки, на которых берутся схемы проекта. */
        val GRID_SIZES = listOf(8, 16, 32)

        /** Точки сравнения; 41 точка покрывает отрезок мельче самой грубой сетки. */
        const val SAMPLE_SIZE = 41
        val SAMPLE: List<Double> = (0 until SAMPLE_SIZE).map { A + (B - A) * it / (SAMPLE_SIZE - 1.0) }

        /**
         * Допуск на пригодность эталона.
         *
         * Обоснование величины: квадратура Гаусса–Лежандра на гладком ядре сходится
         * экспоненциально, поэтому 64 узла дают точность уровня машинной. Порог 1e-12
         * оставляет запас порядка 100 ulp на накопление ошибки решения СЛАУ 64x64 —
         * и при этом на шесть порядков строже погрешности схем проекта (1e-5..1e-8),
         * так что эталон заведомо точнее измеряемой величины.
         */
        const val TOL_REFERENCE_FITNESS = 1e-12

        /**
         * АБСОЛЮТНЫЕ пороги отклонения от эталона: ключ `задача/система/схема/n`.
         *
         * ПОЧЕМУ АБСОЛЮТНЫЕ, а не «с запасом к отклонению от точного решения».
         * Предыдущий вариант сравнивал `versusReference <= 1.5 * versusExact + 1e-12`
         * и был ТОЖДЕСТВЕННО ИСТИНЕН. Доказательство: по неравенству треугольника
         * `versusReference <= versusExact + |эталон - точное|`, а второе слагаемое уже
         * ограничено [TOL_REFERENCE_FITNESS] тестом [referenceReproducesExactSolution].
         * Значит `versusReference <= versusExact + 1e-12 <= 1.5 * versusExact + 1e-12`
         * выполнялось ПРИ ЛЮБОМ `versusExact`, включая деградировавшее в тысячи раз:
         * тест не мог упасть в принципе и проверял лишь арифметику самих формул.
         *
         * Абсолютный порог от этого свободен: он не зависит от проверяемого решения.
         * Второй рассматривавшийся вариант (`|versusReference - versusExact| <= 2 * TOL`)
         * отвергнут: он проверяет близость ДВУХ ИЗМЕРЕНИЙ одной и той же ошибки
         * (то есть всё то же неравенство треугольника, только в две стороны), и при
         * деградации схемы ОБА измерения растут согласованно, так что разность
         * осталась бы малой — деградация снова прошла бы. Значимо только требование,
         * внешнее к решению.
         *
         * Откуда взяты числа: удвоение фактически измеренного `versusReference`
         * (сводка печатается самим тестом) — тот же принцип, что у таблицы
         * `EH_LIMITS` в `tools/verify_with_scipy.py`. Запас x2 ловит падение порядка
         * сходимости на любой из трёх сеток (переход O(h^3) -> O(h^2) при n = 8 даёт
         * рост в 8 раз), а разброса между прогонами здесь нет: величины
         * детерминированы бит-в-бит.
         *
         * Особый случай `F2exp/H`: решение `exp(t)` лежит в span системы `H`, поэтому
         * остаётся только округление (1e-13...1e-11), растущее с размером СЛАУ. Пороги
         * там такие же по правилу (x2 факта) — и именно они проверяют, что свойство
         * «решение в span» не утрачено.
         */
        val REFERENCE_LIMITS: Map<String, Double> = mapOf(
            "F2/B/base/n=8" to 2.03e-04, "F2/B/base/n=16" to 2.49e-05, "F2/B/base/n=32" to 3.05e-06,
            "F2/B/sloan/n=8" to 9.62e-06, "F2/B/sloan/n=16" to 5.00e-07, "F2/B/sloan/n=32" to 2.51e-08,
            "F2/H/base/n=8" to 1.70e-04, "F2/H/base/n=16" to 2.09e-05, "F2/H/base/n=32" to 2.55e-06,
            "F2/H/sloan/n=8" to 8.62e-06, "F2/H/sloan/n=16" to 4.54e-07, "F2/H/sloan/n=32" to 2.29e-08,
            "F2/T/base/n=8" to 2.37e-04, "F2/T/base/n=16" to 2.90e-05, "F2/T/base/n=32" to 3.55e-06,
            "F2/T/sloan/n=8" to 1.07e-05, "F2/T/sloan/n=16" to 5.45e-07, "F2/T/sloan/n=32" to 2.72e-08,
            "F2exp/B/base/n=8" to 9.82e-05, "F2exp/B/base/n=16" to 1.13e-05, "F2exp/B/base/n=32" to 1.37e-06,
            "F2exp/B/sloan/n=8" to 1.29e-05, "F2exp/B/sloan/n=16" to 6.39e-07, "F2exp/B/sloan/n=32" to 3.50e-08,
            // Решение в span системы H: остаётся только округление (см. выше).
            "F2exp/H/base/n=8" to 1.00e-12, "F2exp/H/base/n=16" to 1.00e-11, "F2exp/H/base/n=32" to 2.40e-11,
            "F2exp/H/sloan/n=8" to 2.20e-13, "F2exp/H/sloan/n=16" to 2.20e-12, "F2exp/H/sloan/n=32" to 3.10e-12,
            "F2exp/T/base/n=8" to 1.97e-04, "F2exp/T/base/n=16" to 2.27e-05, "F2exp/T/base/n=32" to 2.74e-06,
            "F2exp/T/sloan/n=8" to 2.57e-05, "F2exp/T/sloan/n=16" to 1.28e-06, "F2exp/T/sloan/n=32" to 7.00e-08,
        )
    }

    /** Модельные задачи; для каждой — независимо выписанные ядро и точное решение. */
    private fun problems() = listOf(FredholmProblem.F2, FredholmProblem.F2exp)

    /**
     * Эталон строится ТОЛЬКО из постановки задачи: ядро, правая часть, отрезок.
     * Правая часть берётся точной (`f = u - Ku`, интеграл считается самим эталоном
     * на его же квадратуре), чтобы не втянуть в эталон оператор проекта.
     */
    private fun reference(problem: FredholmProblem): ReferenceNystromSolver {
        val kernel = { t: Double, s: Double -> problem.kernel.k(t, s) }
        // f(t) = u(t) - ∫ K(t,s) u(s) ds. Интеграл берётся НЕЗАВИСИМОЙ квадратурой
        // эталона, а НЕ оператором проекта (`problem.rhsExact` требует
        // `FredholmOperator` и втянул бы проверяемую квадратуру в эталон).
        val rhs = { t: Double ->
            problem.exact(t) - ReferenceNystromSolver.integrate(A, B) { s -> kernel(t, s) * problem.exact(s) }
        }
        return ReferenceNystromSolver(kernel, rhs, lambda = 1.0, a = A, b = B, nodeCount = 64)
    }

    /**
     * Шаг 1: эталон воспроизводит точное решение модельных задач.
     *
     * Без этой проверки шаг 2 не имеет смысла: сверка с негодным эталоном либо
     * пропускает дефекты, либо сообщает о ложных.
     */
    @Test
    fun referenceReproducesExactSolution() {
        for (problem in problems()) {
            val solver = reference(problem)
            val deviation = SAMPLE.maxOf { t -> abs(solver.eval(t) - problem.exact(t)) }
            assertTrue(
                deviation <= TOL_REFERENCE_FITNESS,
                "Эталонный Nystrom не воспроизводит точное решение задачи ${problem.name}: " +
                    "отклонение $deviation > допуска $TOL_REFERENCE_FITNESS. Сверять таким " +
                    "эталоном нельзя — сначала причина, а не ослабление допуска.",
            )
        }
    }

    /**
     * Шаг 2: решения схем проекта согласуются с независимым эталоном.
     *
     * Проверяются обе схемы (базовая и итерация Слоана) на всех трёх порождающих
     * системах и трёх сетках — то есть тот же охват, что и у выгружаемого
     * `solution-errors.tsv`, но сравнение идёт ПОТОЧЕЧНО с независимым решением,
     * а не только по агрегату `E_h`.
     */
    @Test
    fun projectSchemesAgreeWithReferenceNystrom() {
        // Сводка худших отклонений: печатается всегда, а не только при падении.
        // Причина: зелёный тест без чисел не позволяет отличить «схемы точны» от
        // «допуск слишком широк», а именно это различие и есть содержание сверки.
        val summary = linkedMapOf<String, Double>()
        for (problem in problems()) {
            val referenceSolver = reference(problem)
            for (system in listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)) {
                for (n in GRID_SIZES) {
                    val grid = Grid.uniform(n, a = A, b = B)
                    val basis = MinimalSplineBasis(system, grid)
                    val op = FredholmOperator(problem.kernel, grid, GaussLegendre(8))
                    val solver = FredholmSecondKindSolver(
                        basis, ProjFunctionals(basis), op, 1.0,
                        RhsWithDerivatives(
                            { t -> problem.rhsExact(t, op) },
                            { t -> problem.rhsExactDeriv(t, op) },
                            { t -> problem.rhsExactDeriv2(t, op) },
                        ),
                    )
                    for ((scheme, evaluate) in listOf(
                        "base" to solver.base().eval,
                        "sloan" to solver.sloan().eval,
                    )) {
                        val versusReference = SAMPLE.maxOf { t -> abs(evaluate(t) - referenceSolver.eval(t)) }
                        val tag = "${problem.name}/${system.name}/$scheme/n=$n"
                        summary[tag] = versusReference
                        // Нефинитное отклонение — отдельный отказ: сравнение `NaN <= limit`
                        // ложно, поэтому без проверки сообщение было бы о превышении порога,
                        // что уводит диагностику в сторону от настоящей причины.
                        assertTrue(
                            versusReference.isFinite(),
                            "$tag: отклонение от эталона НЕФИНИТНО ($versusReference): схема " +
                                "вернула NaN или бесконечность",
                        )
                        val limit = REFERENCE_LIMITS[tag]
                        assertTrue(
                            limit != null,
                            "$tag: для этого сочетания нет порога в REFERENCE_LIMITS. Новое сочетание " +
                                "обязано либо сверяться, либо ронять тест — но не проходить молча",
                        )
                        assertTrue(
                            versusReference <= limit!!,
                            "$tag: отклонение от НЕЗАВИСИМОГО эталона $versusReference превышает " +
                                "абсолютный порог $limit. Эталон не использует код проекта (см. KDoc " +
                                "ReferenceNystromSolver), поэтому расхождение указывает на дефект схемы, " +
                                "а не эталона. Порог НЕ ослаблять — сначала причина.",
                        )
                    }
                }
            }
        }
        // Полнота в ОБРАТНУЮ сторону: каждый порог обязан быть использован. Иначе
        // исчезновение сочетания из цикла (например, сужение GRID_SIZES) сократило бы
        // объём сверки незаметно — тест оставался бы зелёным.
        val unused = REFERENCE_LIMITS.keys - summary.keys
        assertTrue(
            unused.isEmpty(),
            "Пороги REFERENCE_LIMITS остались НЕИСПОЛЬЗОВАННЫМи: ${unused.sorted()}. Значит, " +
                "сверка перестала охватывать часть сочетаний, а тест этого не заметил бы",
        )
        println("Отклонение схем проекта от НЕЗАВИСИМОГО эталона Nystrom (max по $SAMPLE_SIZE точкам):")
        for ((tag, deviation) in summary) {
            println("  %-26s %.3e  (порог %.3e)".format(tag, deviation, REFERENCE_LIMITS[tag]))
        }
    }
}
