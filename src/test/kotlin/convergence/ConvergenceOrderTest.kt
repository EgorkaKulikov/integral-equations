package convergence

import numerics.GaussLegendre
import numerics.GeneratingSystem
import numerics.Grid
import numerics.MinimalSplineBasis
import numerics.SolutionFunc
import numerics.functionals.AveragingFunctionals
import numerics.functionals.DeBoorFixFunctionals
import numerics.functionals.FunctionalFamily
import numerics.functionals.ProjFunctionals
import numerics.functionals.ThreePointFunctionals
import numerics.functionals.errorEh
import numerics.functionals.orders
import org.junit.jupiter.api.Tag
import problems.fredholm.FredholmProblem
import problems.volterra.VolterraProblem
import solvers.fredholm.FredholmOperator
import solvers.volterra.VolterraOperator
import java.util.Locale
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * ПОРЯДОК СХОДИМОСТИ как проверяемое свойство (задача 1 `TASK.md`).
 *
 * ЗАЧЕМ ЭТОТ ТЕСТ СУЩЕСТВУЕТ, если уже есть `baseline-eh.tsv` на 1316 значений.
 * Характеризационный эталон фиксирует ЧИСЛА на сетках n = 8 и 16. Регрессия, которая
 * срезает порядок с 4 до 3, но почти не меняет значения на грубых сетках, проходит
 * его молча — и именно так дефект `kulkarniQuasi` (кусочно-линейная реконструкция
 * итеранта, срезавшая порядок с ~3.8 до ~1.9) прожил незамеченным. Здесь проверяется
 * не значение, а СКОРОСТЬ его убывания: `p = log2(E_n / E_2n)`.
 *
 * ЧТО ИЗМЕРЯЕТСЯ. Для каждого сочетания «уравнение x схема x порождающая система x
 * семейство функционалов» строится последовательность погрешностей `E_h` на сетках
 * [GRID_SIZES] и по ней — эмпирические порядки. Ожидаемое значение задано ЯВНОЙ
 * ТАБЛИЦЕЙ [expected] с допуском [ORDER_TOLERANCE]; таблица снята фактическим
 * прогоном и сверена с `docs/REFERENCES.md` (см. ниже).
 *
 * НАБОРЫ И ТЕГИ. Полная матрица (168 сочетаний до n = 64) — тег `slow` и отдельная
 * задача `./gradlew convergenceOrderTest` (замеры двух прогонов: 91 с и 105 с wall-clock,
 * то есть 1.5-2 мин — вилка, а не точное число). Сокращённый поднабор
 * (базис B, семейства `theta`/`mu`, сетки до n = 32) и проверка точности на span — тег
 * `fast` (замер: 3.0-3.5 с и 1.8-2.2 с), они гоняются на каждой правке.
 *
 * СВЕРКА С `docs/REFERENCES.md` (раздел 3, «Замечание о порядках сходимости»).
 * Расхождений более 0.5 между фактом и документацией НЕ ОБНАРУЖЕНО:
 *  * классический Nyström «порядок не повышает», в документации указано ~4.2 для
 *    Фредгольма с базисом B и семейством theta — измерено 4.21;
 *  * комбинированный Nyström (Фредгольм) — опубликованная оценка `O(h^7)`,
 *    в документации ~7.0 — измерено 7.04; итерированный вариант `O(h^8)` — измерено 8.13;
 *  * комбинированный Nyström (Вольтерра) — «выигрыша не наблюдается, около 3.8» —
 *    измерено 3.90 (расхождение 0.10, в пределах допуска);
 *  * итерация Слоана (Фредгольм) — в документации «наблюдаемые 4.2...4.3» — измерено 4.26;
 *  * базовая коллокация — теоретические 3 — измерено 3.0...3.1.
 * Для схемы Кулкарни на квазиинтерполянтах (`mu`, `lambda`) опубликованного порядка
 * НЕТ (в `REFERENCES.md` она помечена «без источника, численное наблюдение»), поэтому
 * таблица здесь фиксирует именно наблюдение, а не подтверждает теорию.
 *
 * ### СВЕРКА С ОПУБЛИКОВАННЫМИ ТАБЛИЦАМИ (внешний источник, а не собственный прогон)
 *
 * Сверка выполнена по `src/test/resources/verification/published-values.tsv` (ключи `*.ph`,
 * извлечённые скриптом из .tex-таблиц статьи). Итог: из 56 строк таблицы [expected]
 * внешним источником ПОДТВЕРЖДЕНЫ 24; расхождений более 0.5 НЕТ (худшее — 0.44).
 *
 * КАК СРАВНИВАЛОСЬ. Опубликованное `p_h` с ключом `nN` — это порядок на КОНКРЕТНОЙ
 * паре `N -> 2N`, а не асимптотика, поэтому бралась самая мелкая пара, ОБЕ погрешности
 * которой лежат выше порога доверия СТРОКИ. Без этого условия сравнение бессмысленно,
 * и это НЕ подгонка: в самой публикации `F.F2.B.theta.n32.iterKulkarni.ph = 0` при
 * `Eh(n32) = Eh(n64) = 4.441e-16` — то есть авторы тоже упёрлись в машинный нуль, и их
 * число 0 описывает арифметику, а не схему. Тот же фильтр по `NOISE_FLOOR` применяет
 * и `verification.PublishedValuesTest.checkOrders`.
 *
 * СТРОКИ, ПОДТВЕРЖДЁННЫЕ ПУБЛИКАЦИЕЙ (ожидание теста → опубликовано, отклонение):
 *  * `F.theta.base` 3.00→3.02 (0.02); `F.mu.base` 2.95→2.94 (0.01);
 *    `F.lambda.base` 3.10→3.07 (0.03); `F.xi1.base` 3.00→2.96 (0.04);
 *  * `F.theta.sloan` 4.30→4.26 (0.04); `F.xi1.sloan` 3.00→2.99 (0.01);
 *  * `F.theta.kulkarni` 7.35→7.35 (0.00); `F.xi1.kulkarni` 5.90→5.88 (0.02);
 *  * `F.xi1.iterKulkarni` 5.95→5.94 (0.01);
 *    **`F.theta.iterKulkarni` 8.60→8.55 (0.05)** — именно этот ключ доказывает, что строка
 *    измерима и раньше была ошибочно помечена `Saturates`;
 *  * `F.theta.nystrom` 4.20→4.21 (0.01); `F.theta.iterNystrom` 4.20→4.20 (0.00);
 *  * `V.theta.base` 3.00→3.01; `V.mu.base` 2.95→2.94; `V.lambda.base` 3.00→3.03;
 *    `V.xi1.base` 3.00→2.96;
 *  * `V.theta.sloan` 3.90→3.94; `V.xi1.sloan` 3.00→2.99;
 *  * `V.theta.kulkarni` 3.90→3.92; `V.xi1.kulkarni` 4.00→4.00;
 *  * `V.theta.iterKulkarni` 4.15→4.13; `V.xi1.iterKulkarni` 3.75→3.75;
 *  * `V.theta.nystrom` 3.90→3.84 (0.06);
 *    `V.theta.iterNystrom` 4.30→4.74 (0.44) — ХУДШЕЕ расхождение. Причина измерена,
 *    а не предположена: у этой схемы порядок заметно колеблется от уровня к уровню
 *    (наш замер по трём парам: 4.27, 4.74, 4.26 — точно так же, как в публикации),
 *    и опубликованное 4.74 — тот самый выброс на паре 16->32. Схема воспроизводится
 *    точно, расхождение — свойство выбора одного числа на строку, а не дефект.
 *
 * СТРОКИ БЕЗ ВНЕШНЕГО ПОДТВЕРЖДЕНИЯ (32 из 56) — взяты ТОЛЬКО из фактического
 * прогона и закрепляют текущее поведение, а не подтверждают его правильность:
 *  * все 8 строк `combNystrom`/`iterCombNystrom` (в таблицах статьи этих схем нет);
 *  * все схемы семейств `mu`/`lambda`, кроме `base`: публикация даёт для них только
 *    базовую коллокацию (`table-families.tex`);
 *  * `F.mu.iterKulkarni` — вторая бывшая `Saturates`-строка: прямого ключа нет,
 *    ожидание 8.6 взято из замера (8.23...8.84) и согласуется с подтверждённым 8.55 у `theta`.
 *
 * ГРАНИЦЫ ОХВАТА (осознанные, а не по недосмотру):
 *  * задачи — по одной на уравнение: `F2` (`K = 1/(1+t+s)`, `u* = 1/(t+1)`) и `V2`.
 *    Обе выбраны потому, что их решение не лежит НИ В ОДНОЙ порождающей системе, то
 *    есть порядок измерим для всех трёх базисов одним и тем же способом. Задача
 *    `V2win` (наблюдаемые порядки ~3/4/5/6) в матрицу не входит: она содержательна
 *    сама по себе, но её ядро `K = t - s` вырождено на диагонали, и порядки на ней
 *    другие — это отдельная таблица, а не строка этой;
 *  * `F2span`/`V2span` исключены из измерения порядка ПО ПОСТРОЕНИЮ: их решение
 *    `u* = t^2` лежит в span системы B, и погрешность там — чистый шум округления
 *    (1e-14...1.6e-13), не зависящий от шага сетки. Посчитанный по нему «порядок» — это
 *    log2 отношения двух случайных величин: он НЕ NaN, а произвольное число любого знака
 *    (наблюдалось −2.1 при E = 1.9e-14 и 8.2e-14) — и именно поэтому бессмыслен. NaN
 *    `orders` вернёт только при точном нуле. Для этих задач действует отдельная проверка
 *    [spanProblemsAreReproducedExactly] — «погрешность ниже [SPAN_EXACTNESS_TOLERANCE]»;
 *  * семейство `xi1` ([DeBoorFixFunctionals]) НЕ сочетается с четырьмя Nyström-схемами:
 *    `nystromSupport()` начинается с `require(!funcs.usesDerivative)`, а `xi1` —
 *    единственное семейство матрицы с `usesDerivative = true`. Эти 12 сочетаний на
 *    уравнение исключены не выбором, а контрактом кода.
 */
class ConvergenceOrderTest {

    private companion object {

        /** Допуск сравнения наблюдаемого порядка с табличным (по `TASK.md`). */
        const val ORDER_TOLERANCE = 0.4

        /**
         * Допуск для НЕПОСЛЕДНИХ пар сеток (преасимптотический режим).
         *
         * Проверяются ВСЕ пригодные пары, а не только последняя, иначе деградация на грубых
         * сетках проходит молча. Но на грубых сетках схема ещё НЕ вышла на асимптотику,
         * и тот же допуск 0.4 давал бы ложные срабатывания на исправном коде.
         *
         * ЗНАЧЕНИЕ ВЫБРАНО ПО ЗАМЕРУ, а не назначено. По 303 ранним парам исправного
         * кода худший недобор порядка — −0.578 (`V/H/mu/nystrom`, пара 8->16: 3.17 при
         * асимптотических 3.75); хуже −0.5 — три случая, хуже −0.6 — ни одного. Порог 0.8
         * даёт запас 0.22 от худшего факта и при этом вчетверо меньше того недобора (4...7),
         * который даёт целевая регрессия `kulkarniQuasi`, то есть её ловит с большим запасом.
         */
        const val PREASYMPTOTIC_TOLERANCE = 0.8

        /**
         * Порог машинной точности: как только погрешность опускается ниже, измельчение
         * ПРЕКРАЩАЕТСЯ. Дальше убывает не ошибка метода, а шум округления, и порядок,
         * посчитанный по такой паре, характеризует арифметику, а не схему.
         */
        const val MACHINE_PRECISION_FLOOR = 1e-13

        /**
         * Нижняя граница ДОВЕРИЯ к погрешности при вычислении порядка.
         *
         * Порог выше [MACHINE_PRECISION_FLOOR] намеренно. «Плато» округления на задачах
         * матрицы лежит в диапазоне 9.5e-14...2.2e-13: попав в него, погрешность перестаёт
         * убывать при измельчении ВОВСЕ (наблюдались строки вида `1.5510e-13 1.5521e-13
         * 1.5521e-13`). Значение с этого плато формально выше порога остановки, но состоит
         * из шума округления, и порядок, посчитанный ПРОТИВ него, занижен на единицы
         * (наблюдалось 6.34 -> 0.82 у одной и той же схемы). Поэтому пара `(E_n, E_2n)`
         * считается пригодной, только если ОБЕ погрешности не ниже этого порога.
         *
         * ПОЧЕМУ ИМЕННО 2.7e-13, а не круглое число. Порог выбран по ЗАМЕРУ, в самом
         * широком зазоре гистограммы: наибольшее значение с плато шума — 2.2171e-13,
         * наименьшее содержательное значение — 3.3329e-13, и 2.7e-13 отстоит от обоих
         * примерно на 20 %. Любое «круглое» 2e-13 или 3e-13 прижималось бы к одной из
         * границ и делало классификацию хрупкой.
         */
        const val TRUSTED_ERROR_FLOOR = 2.7e-13

        /**
         * Верхняя граница погрешности на самой грубой сетке для сочетаний, помеченных
         * [Expected.Saturates]: у них уже второй уровень измельчения уходит под
         * [TRUSTED_ERROR_FLOOR], и порядок измерить нечем. Наблюдаемый максимум по таким
         * сочетаниям — 1.1e-11, порог даёт запас на порядок. Ради чего он нужен: при
         * деградации схемы (например, возврате кусочно-линейной реконструкции в
         * `kulkarniQuasi`) погрешность на n = 8 подскакивает на 6-8 порядков, и проверка
         * срабатывает даже там, где порядок не определён.
         */
        const val SATURATED_MAX_COARSE_ERROR = 1e-10

        /**
         * Порог для задач, решение которых лежит в span порождающей системы (`TASK.md`, п. 2).
         *
         * Задание требует «ниже 1e-10», но буквальное 1e-10 здесь — ПОЧТИ ПУСТАЯ ПРОВЕРКА:
         * фактический максимум по всем 64 сочетаниям равен 1.5521e-13
         * (`F/lambda/kulkarni/n=16`), то есть запас был 645-кратным. Деградация точности на span
         * в сотни раз прошла бы молча.
         *
         * Порог 5e-13 — три фактических максимума (3.2x). Запас нужен не «на всякий случай»:
         * сама величина есть накопленная ошибка округления в решении СЛАУ, и она закономерно
         * плавает между бэкендами и версиями JDK. При этом 5e-13 в 200 раз строже прежнего
         * порога и срабатывает задолго до того, как погрешность станет содержательной.
         */
        const val SPAN_EXACTNESS_TOLERANCE = 5e-13

        /**
         * Предел длины списка падений в сообщении об ошибке.
         *
         * Само усечение неизбежно: при полной деградации сообщение вырастает до десятков
         * килобайт и теряет читаемость. Важно другое — чтобы факт усечения был ОТМЕЧЕН:
         * см. [reportIfAny].
         */
        const val FAILURE_REPORT_LIMIT = 8000

        /** Порядок квадратуры Гаусса — тот же, что в характеризационных эталонах. */
        const val QUADRATURE_ORDER = 8

        /** Последовательность сеток полной матрицы. */
        val GRID_SIZES = listOf(8, 16, 32, 64)

        /**
         * Последовательность сеток быстрого поднабора: n = 64 стоит около 6 с на одно
         * сочетание для Вольтерры и в бюджет `fastTest` не укладывается.
         *
         * СЕТКА n = 32 ЗДЕСЬ ОБЯЗАТЕЛЬНА, хотя и вдвое дороже пары 8/16. Замер: на паре
         * 8->16 схема `combNystrom` с семейством `mu` даёт порядок 6.24 при асимптотических
         * 6.65 — она ещё не вышла на асимптотику, и даже одностороннее сравнение с допуском
         * 0.4 падало бы на исправном коде. На паре 16->32 та же схема даёт 6.66.
         */
        val FAST_GRID_SIZES = listOf(8, 16, 32)

        /** Сетки проверки точности на span: двух достаточно, погрешность там от n не зависит. */
        val SPAN_GRID_SIZES = listOf(8, 16)

        const val FREDHOLM = "F"
        const val VOLTERRA = "V"

        val SYSTEMS = listOf(GeneratingSystem.B, GeneratingSystem.H, GeneratingSystem.T)
        val FAMILIES = listOf("theta", "xi1", "mu", "lambda")

        /** Схемы, строящие приближение ВНУТРИ сплайнового пространства. */
        val SPLINE_SPACE_SCHEMES = listOf("base", "sloan", "kulkarni", "iterKulkarni")

        /**
         * Быстрый поднабор: базис B и два семейства.
         *
         * Почему не одно `theta`, как предлагалось в задании. `theta` — ПРОЕКТОР, и
         * `kulkarni()` уходит для него в ветвь `kulkarniProjector` (прямое решение СЛАУ).
         * Ветвь `kulkarniQuasi` — та самая, ради регрессии в которой сформулирован
         * критерий приёмки, — вызывается ТОЛЬКО для квазиинтерполянтов. Поднабор из
         * одного `theta` не отреагировал бы на неё вовсе. Добавление `mu` стоит ~2 с и
         * закрывает эту дыру.
         */
        val FAST_FAMILIES = listOf("theta", "mu")
    }

    /** Ожидание для одного сочетания: либо числовой порядок, либо выход на машинную точность. */
    private sealed interface Expected {

        /**
         * Порог доверия НА ЭТУ СТРОКУ, а не общий на всю таблицу.
         *
         * Почему порог НЕ может быть единым. Схемы матрицы различаются по абсолютному
         * уровню погрешности на девять порядков: `base` даёт 1e-4, а `iterKulkarni`
         * на Фредгольме — уже 3e-12 на самой ГРУБОЙ сетке. Общий порог 2.7e-13,
         * подобранный по плато шума большинства схем, ЛЕЖАЛ ВЫШЕ всего диапазона
         * сходимости двух самых точных строк — и их измеримый порядка 8.6 просто
         * отбрасывался как «шум». Порог на строку убирает эту слепую зону.
         */
        val trustFloor: Double

        /** Ожидаемый эмпирический порядок; сравнение — с допуском [ORDER_TOLERANCE]. */
        data class Order(
            val p: Double,
            override val trustFloor: Double = TRUSTED_ERROR_FLOOR,
        ) : Expected

        /**
         * Сочетание выходит на машинную точность раньше, чем накопится пригодная пара
         * погрешностей: порядок НЕ ИЗМЕРИМ. Проверяется величина погрешности на грубой сетке.
         *
         * ПОСЛЕ ФИКСА ПОРОГА НА СТРОКУ этот вариант НЕ ИСПОЛЬЗУЕТСЯ ни одной строкой
         * таблицы: обе бывшие `Saturates`-строки оказались ИЗМЕРИМЫМИ (см. комментарий
         * к `F.theta.iterKulkarni`). Тип сохранён намеренно: схема, дошедшая до машинного
         * нуля на первой же сетке, физически возможна, и тогда проверять порядок будет
         * действительно нечем.
         */
        data class Saturates(
            override val trustFloor: Double = TRUSTED_ERROR_FLOOR,
        ) : Expected
    }

    /**
     * ЯВНАЯ ТАБЛИЦА ОЖИДАЕМЫХ ПОРЯДКОВ. Ключ — `уравнение.семейство.схема`.
     *
     * ПОЧЕМУ КЛЮЧ НЕ СОДЕРЖИТ ПОРОЖДАЮЩУЮ СИСТЕМУ. Это не упрощение, а ИЗМЕРЕННЫЙ ФАКТ:
     * по всем 56 строкам разброс наблюдаемого порядка между системами B, H и T не
     * превышает 0.31 (типично 0.02...0.10), то есть порядок определяется схемой и
     * семейством функционалов, а не выбором `{1,t,t^2}` / `{1,sinh,cosh}` / `{1,sin,cos}`.
     * Единая строка на три системы делает это утверждение ПРОВЕРЯЕМЫМ: если какая-то
     * правка сделает одну из систем хуже других, строка упадёт.
     *
     * Значения — округлённая середина наблюдённого по трём системам разброса
     * (фактический прогон, бэкенд multik, сетки 8/16/32/64). Худший запас до границы
     * допуска — 0.31 у `V.theta.iterNystrom` (наблюдалось 3.99...4.55: у этой схемы на
     * Вольтерре порядок заметно колеблется от уровня к уровню).
     */
    private val expected: Map<String, Expected> = mapOf(
        // --- Фредгольм, задача F2 -------------------------------------------------
        "F.theta.base" to Expected.Order(3.0),
        "F.theta.sloan" to Expected.Order(4.3),
        "F.theta.kulkarni" to Expected.Order(7.35),
        // СТРОКА С ПОНИЖЕННЫМ ПОРОГОМ ДОВЕРИЯ. Раньше здесь стоял `Saturates`, и это была
        // ОШИБКА: порядок здесь ИЗМЕРИМ (E_8 = 2.99e-12, E_16 = 7.77e-15 → p = 8.59), просто обе
        // погрешности лежали ниже общего порога 2.7e-13 и отбрасывались как «шум».
        // Подмена на «E_8 < 1e-10» была слабее факта в 33 раза: деградация в 30 раз прошла бы молча.
        // Порог 1e-15: выше абсолютного машинного нуля задачи (~1e-16), но ниже наименьшей
        // содержательной погрешности строки (7.55e-15 по трём системам) с запасом в 7 раз.
        // ОЖИДАНИЕ 8.6 ПОДТВЕРЖДЕНО ПУБЛИКАЦИЕЙ: `F.F2.B.theta.n8.iterKulkarni.ph` = 8.55
        // (`published-values.tsv`, `table-t2-fredholm.tex`); измерено 8.44...8.79 по трём системам.
        "F.theta.iterKulkarni" to Expected.Order(8.6, trustFloor = 1e-15),
        "F.theta.nystrom" to Expected.Order(4.2),
        "F.theta.iterNystrom" to Expected.Order(4.2),
        "F.theta.combNystrom" to Expected.Order(7.0),
        "F.theta.iterCombNystrom" to Expected.Order(8.1),
        "F.xi1.base" to Expected.Order(3.0),
        "F.xi1.sloan" to Expected.Order(3.0),
        "F.xi1.kulkarni" to Expected.Order(5.9),
        "F.xi1.iterKulkarni" to Expected.Order(5.95),
        "F.mu.base" to Expected.Order(2.95),
        "F.mu.sloan" to Expected.Order(3.9),
        "F.mu.kulkarni" to Expected.Order(6.3),
        // То же самое, что и у `F.theta.iterKulkarni`: было `Saturates`, фактически порядок измерим
        // (E_8 = 8.21e-12, E_16 = 1.99e-14 → p = 8.69; по трём системам 8.23...8.84). Публикацией не
        // закреплено НАПРЯМУЮ (в таблицах нет `mu` с `iterKulkarni`), но согласуется с 8.55
        // у `theta`: у квазиинтерполянта тот же механизм суперсходимости.
        "F.mu.iterKulkarni" to Expected.Order(8.6, trustFloor = 1e-15),
        "F.mu.nystrom" to Expected.Order(3.9),
        "F.mu.iterNystrom" to Expected.Order(3.9),
        "F.mu.combNystrom" to Expected.Order(6.65),
        "F.mu.iterCombNystrom" to Expected.Order(7.0),
        "F.lambda.base" to Expected.Order(3.1),
        "F.lambda.sloan" to Expected.Order(3.9),
        "F.lambda.kulkarni" to Expected.Order(6.6),
        "F.lambda.iterKulkarni" to Expected.Order(7.1),
        "F.lambda.nystrom" to Expected.Order(3.9),
        "F.lambda.iterNystrom" to Expected.Order(3.9),
        "F.lambda.combNystrom" to Expected.Order(6.85),
        "F.lambda.iterCombNystrom" to Expected.Order(7.25),
        // --- Вольтерра, задача V2 -------------------------------------------------
        "V.theta.base" to Expected.Order(3.0),
        "V.theta.sloan" to Expected.Order(3.9),
        "V.theta.kulkarni" to Expected.Order(3.9),
        "V.theta.iterKulkarni" to Expected.Order(4.15),
        "V.theta.nystrom" to Expected.Order(3.9),
        "V.theta.iterNystrom" to Expected.Order(4.3),
        "V.theta.combNystrom" to Expected.Order(3.9),
        "V.theta.iterCombNystrom" to Expected.Order(4.15),
        "V.xi1.base" to Expected.Order(3.0),
        "V.xi1.sloan" to Expected.Order(3.0),
        "V.xi1.kulkarni" to Expected.Order(4.0),
        "V.xi1.iterKulkarni" to Expected.Order(3.75),
        "V.mu.base" to Expected.Order(2.95),
        "V.mu.sloan" to Expected.Order(3.8),
        "V.mu.kulkarni" to Expected.Order(3.8),
        "V.mu.iterKulkarni" to Expected.Order(3.9),
        "V.mu.nystrom" to Expected.Order(3.75),
        "V.mu.iterNystrom" to Expected.Order(3.9),
        "V.mu.combNystrom" to Expected.Order(3.75),
        "V.mu.iterCombNystrom" to Expected.Order(3.9),
        "V.lambda.base" to Expected.Order(3.0),
        "V.lambda.sloan" to Expected.Order(3.9),
        "V.lambda.kulkarni" to Expected.Order(3.9),
        "V.lambda.iterKulkarni" to Expected.Order(3.9),
        "V.lambda.nystrom" to Expected.Order(3.9),
        "V.lambda.iterNystrom" to Expected.Order(3.9),
        "V.lambda.combNystrom" to Expected.Order(3.9),
        "V.lambda.iterCombNystrom" to Expected.Order(3.9),
    )

    // ------------------------------------------------------------------------
    // Тесты
    // ------------------------------------------------------------------------

    /**
     * ПОЛНАЯ МАТРИЦА: 2 уравнения x 3 системы x 4 семейства x (4 + 4) схем за вычетом
     * запрещённых сочетаний `xi1` x Nyström — 168 сочетаний, сетки 8/16/32/64.
     *
     * Сравнение ДВУСТОРОННЕЕ: порядок обязан не только не упасть, но и не «улучшиться»
     * необъяснимо — рост порядка на 0.5 означает, что схема считает не то, что раньше,
     * и это столь же весомый повод разобраться.
     *
     * Фактическое время прогона САМОГО ЭТОГО метода — 79...99 с по двум замерам
     * (остальное время задачи — два `fast`-метода и старт JVM), поэтому тег `slow`;
     * отдельная задача `./gradlew convergenceOrderTest` делает набор исполнимым, не
     * дожидаясь починки всего `slowTest`.
     */
    @Test
    @Tag("slow")
    fun convergenceOrdersMatchExpectedTable() {
        val failures = mutableListOf<String>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (system in SYSTEMS) {
                for (familyName in FAMILIES) {
                    val errorsByScheme = collectErrors(equation, system, familyName, GRID_SIZES)
                    for ((scheme, errs) in errorsByScheme) {
                        checked++
                        checkCombination(
                            key = "$equation.$familyName.$scheme",
                            label = "$equation/${system.name}/$familyName/$scheme",
                            gridSizes = GRID_SIZES,
                            errs = errs,
                            oneSided = false,
                            failures = failures,
                        )
                    }
                }
            }
        }
        assertTrue(
            checked == 168,
            "Матрица должна содержать 168 сочетаний (2 уравнения x 3 системы x " +
                "(4 семейства x 4 схемы + 3 семейства без производной x 4 схемы Nyström)), " +
                "проверено $checked",
        )
        reportIfAny(failures, checked, "Порядок сходимости не соответствует таблице")
    }

    /**
     * БЫСТРЫЙ ПОДНАБОР: базис B, семейства `theta` и `mu`, сетки 8/16/32 — 32 сочетания
     * (2 уравнения x 2 семейства x 8 схем; оба семейства без производной, поэтому все восемь
     * схем доступны), замер 3.0-3.5 с. Запускается на каждой правке в составе `fastTest`.
     *
     * Сравнение ОДНОСТОРОННЕЕ — `p >= ожидаемый - допуск`. Это не послабление, а
     * следствие того, что на сетках до n = 32 схема ещё не всегда вышла на асимптотику:
     * например, у `V/B/theta/iterNystrom` порядок на паре 16->32 равен 4.74 при
     * асимптотических 4.3, и двустороннее сравнение падало бы на исправном коде.
     * Ловить нужно ДЕГРАДАЦИЮ, а любая деградация порядок только СНИЖАЕТ. Точное
     * соответствие таблице проверяет [convergenceOrdersMatchExpectedTable].
     */
    @Test
    @Tag("fast")
    fun convergenceOrdersDoNotDegradeOnFastSubset() {
        val failures = mutableListOf<String>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (familyName in FAST_FAMILIES) {
                val errorsByScheme =
                    collectErrors(equation, GeneratingSystem.B, familyName, FAST_GRID_SIZES)
                for ((scheme, errs) in errorsByScheme) {
                    checked++
                    checkCombination(
                        key = "$equation.$familyName.$scheme",
                        label = "$equation/B/$familyName/$scheme",
                        gridSizes = FAST_GRID_SIZES,
                        errs = errs,
                        oneSided = true,
                        failures = failures,
                    )
                }
            }
        }
        assertTrue(
            checked == 32,
            "Быстрый поднабор должен содержать 32 сочетания (2 уравнения x 2 семейства x 8 схем), " +
                "проверено $checked",
        )
        reportIfAny(failures, checked, "Порядок сходимости деградировал относительно таблицы")
    }

    /**
     * Задачи `F2span`/`V2span`: решение `u* = t^2` лежит в span полиномиальной системы B,
     * поэтому схемы, строящие приближение ВНУТРИ сплайнового пространства, обязаны
     * воспроизводить его с погрешностью ниже [SPAN_EXACTNESS_TOLERANCE].
     *
     * ПОЧЕМУ НЕ ИЗМЕРЯЕТСЯ ПОРЯДОК. Погрешность здесь — шум округления решения СЛАУ
     * (1e-14...1.6e-13), от шага сетки она не зависит. `orders` при этом вернёт НЕ NaN,
     * а log2 отношения двух шумов — любое число (например, −2.1); NaN был бы только при
     * точном нуле. Сравнивать такое число с таблицей ожиданий невозможно, и проверяется
     * ВЕЛИЧИНА погрешности.
     *
     * ОГРАНИЧЕНИЯ ПРОВЕРКИ, обе — свойства метода, а не дефекты:
     *  * только система B: `u* = t^2` не лежит в span `{1, sinh, cosh}` и `{1, sin, cos}`,
     *    и на системах H/T погрешность закономерно равна обычным ~4e-5;
     *  * только схемы [SPLINE_SPACE_SCHEMES]: приближение Nyström лежит ВНЕ сплайнового
     *    пространства (`u^N_h = f + L^N_h u`), точность на span оно не наследует — фактически
     *    наблюдается ~5e-5, что согласуется с его собственным порядком 4.
     */
    @Test
    @Tag("fast")
    fun spanProblemsAreReproducedExactly() {
        val failures = mutableListOf<String>()
        val spanErrors = LinkedHashMap<String, MutableList<Double>>()
        var checked = 0
        for (equation in listOf(FREDHOLM, VOLTERRA)) {
            for (familyName in FAMILIES) {
                for (n in SPAN_GRID_SIZES) {
                    val grid = Grid.uniform(n)
                    val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
                    val funcs = family(familyName, basis)
                    val schemes: List<Pair<String, () -> SolutionFunc>>
                    val exact: (Double) -> Double
                    if (equation == FREDHOLM) {
                        val problem = FredholmProblem.F2span
                        val op = FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                        val solver = problems.fredholm.secondKindSolver(problem, basis, funcs, op)
                        schemes = fredholmSchemes(solver, funcs)
                        exact = { t -> problem.exact(t) }
                    } else {
                        val problem = VolterraProblem.V2span
                        val op = VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                        val solver = problems.volterra.secondKindSolver(problem, basis, funcs, op)
                        schemes = volterraSchemes(solver, funcs)
                        exact = { t -> problem.exact(t) }
                    }
                    for ((scheme, build) in schemes) {
                        if (scheme !in SPLINE_SPACE_SCHEMES) continue
                        checked++
                        val error = errorEh(exact, build().eval, grid)
                        spanErrors.getOrPut("$equation/$familyName/$scheme") { mutableListOf() } += error
                    }
                }
                // Сообщение собирается ПОСЛЕ обхода всех сеток семейства: `TASK.md` (п. 4)
                // требует полную последовательность погрешностей во ВСЕХ ветвях падения,
                // а не только там, где считается порядок.
                for (scheme in SPLINE_SPACE_SCHEMES) {
                    val series = spanErrors["$equation/$familyName/$scheme"] ?: continue
                    val worst = series.max()
                    if (!(worst < SPAN_EXACTNESS_TOLERANCE)) {
                        val seq = SPAN_GRID_SIZES.indices.joinToString(", ") {
                            "n=${SPAN_GRID_SIZES[it]}: ${fmtError(series[it])}"
                        }
                        failures += "$equation/span/B/$familyName/$scheme: " +
                            "наибольшая E_h=${fmtError(worst)} должна быть ниже " +
                            "${fmtError(SPAN_EXACTNESS_TOLERANCE)} " +
                            "(u* = t^2 лежит в span порождающей системы B). E_h [$seq]"
                    }
                }
            }
        }
        assertTrue(
            checked == 64,
            "Проверка на span должна охватывать 64 сочетания " +
                "(2 уравнения x 4 семейства x 2 сетки x 4 схемы), проверено $checked",
        )
        reportIfAny(failures, checked, "Точность на span порождающей системы не достигнута")
    }

    // ------------------------------------------------------------------------
    // Измерение
    // ------------------------------------------------------------------------

    private fun family(name: String, basis: MinimalSplineBasis): FunctionalFamily = when (name) {
        "theta" -> ProjFunctionals(basis)
        "xi1" -> DeBoorFixFunctionals(basis, 1)
        "mu" -> AveragingFunctionals(basis, 0.5)
        "lambda" -> ThreePointFunctionals(basis, 0.5)
        else -> error("Неизвестное семейство функционалов: $name")
    }

    /**
     * Схемы решателя Фредгольма. Nyström-схемы отбрасываются для семейств с производной:
     * это контракт кода (`require(!funcs.usesDerivative)`), а не выбор теста.
     */
    private fun fredholmSchemes(
        solver: solvers.fredholm.FredholmSecondKindSolver,
        funcs: FunctionalFamily,
    ): List<Pair<String, () -> SolutionFunc>> {
        val core = listOf<Pair<String, () -> SolutionFunc>>(
            "base" to { solver.base() },
            "sloan" to { solver.sloan() },
            "kulkarni" to { solver.kulkarni() },
            "iterKulkarni" to { solver.iteratedKulkarni() },
        )
        if (funcs.usesDerivative) return core
        return core + listOf<Pair<String, () -> SolutionFunc>>(
            "nystrom" to { solver.nystrom() },
            "iterNystrom" to { solver.iteratedNystrom() },
            "combNystrom" to { solver.combinedNystrom() },
            "iterCombNystrom" to { solver.iteratedCombinedNystrom() },
        )
    }

    /** Схемы решателя Вольтерры; то же ограничение на семейства с производной. */
    private fun volterraSchemes(
        solver: solvers.volterra.VolterraSecondKindSolver,
        funcs: FunctionalFamily,
    ): List<Pair<String, () -> SolutionFunc>> {
        val core = listOf<Pair<String, () -> SolutionFunc>>(
            "base" to { solver.base() },
            "sloan" to { solver.sloan() },
            "kulkarni" to { solver.kulkarni() },
            "iterKulkarni" to { solver.iteratedKulkarni() },
        )
        if (funcs.usesDerivative) return core
        return core + listOf<Pair<String, () -> SolutionFunc>>(
            "nystrom" to { solver.nystrom() },
            "iterNystrom" to { solver.iteratedNystrom() },
            "combNystrom" to { solver.combinedNystrom() },
            "iterCombNystrom" to { solver.iteratedCombinedNystrom() },
        )
    }

    /**
     * Погрешности `E_h` всех схем на последовательности сеток.
     *
     * Решатель строится ОДИН РАЗ на сетку и переиспользуется всеми схемами: сборка
     * матриц `M`/`M2` — самая дорогая часть, и её повторение на каждую схему умножило
     * бы время прогона примерно на восемь.
     *
     * Измельчение для конкретной схемы прекращается, как только её погрешность
     * опустилась ниже [MACHINE_PRECISION_FLOOR] (`TASK.md`, п. 3).
     */
    private fun collectErrors(
        equation: String,
        system: GeneratingSystem,
        familyName: String,
        gridSizes: List<Int>,
    ): Map<String, List<Double>> {
        val errors = LinkedHashMap<String, MutableList<Double>>()
        for (n in gridSizes) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(system, grid)
            val funcs = family(familyName, basis)
            val schemes: List<Pair<String, () -> SolutionFunc>>
            val exact: (Double) -> Double
            if (equation == FREDHOLM) {
                val problem = FredholmProblem.F2
                val op = FredholmOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                schemes = fredholmSchemes(
                    problems.fredholm.secondKindSolver(problem, basis, funcs, op),
                    funcs,
                )
                exact = { t -> problem.exact(t) }
            } else {
                val problem = VolterraProblem.V2
                val op = VolterraOperator(problem.kernel, grid, GaussLegendre(QUADRATURE_ORDER))
                schemes = volterraSchemes(
                    problems.volterra.secondKindSolver(problem, basis, funcs, op),
                    funcs,
                )
                exact = { t -> problem.exact(t) }
            }
            for ((scheme, build) in schemes) {
                val sequence = errors.getOrPut(scheme) { mutableListOf() }
                if (sequence.isNotEmpty() && sequence.last() < MACHINE_PRECISION_FLOOR) continue
                sequence += errorEh(exact, build().eval, grid)
            }
        }
        return errors
    }

    // ------------------------------------------------------------------------
    // Проверка и диагностика
    // ------------------------------------------------------------------------

    /**
     * ВСЕ пригодные пары `(индекс, порядок)` в порядке измельчения: те, у которых обе
     * погрешности не ниже [floor] и порядок не `NaN`.
     *
     * ВОЗВРАЩАЮТСЯ ВСЕ ПАРЫ, а не последняя. Раньше бралась только последняя пара
     * (самая близкая к асимптотике), и это оставляло дыру: преасимптотическая деградация
     * — та, что портит точность на грубых сетках, но выходит на тот же наклон к n = 64,
     * — проходила молча, хотя именно на грубых сетках библиотекой и пользуются.
     */
    private fun trustedOrders(errs: List<Double>, floor: Double): List<Pair<Int, Double>> {
        val ps = orders(errs)
        return (0 until maxOf(errs.size - 1, 0)).mapNotNull { i ->
            if (errs[i] >= floor && errs[i + 1] >= floor && !ps[i].isNaN()) i to ps[i] else null
        }
    }

    /**
     * Сверяет одно сочетание с таблицей и, при расхождении, добавляет в [failures] строку
     * с ПОЛНОЙ последовательностью погрешностей и всеми порядками (`TASK.md`, п. 4):
     * без них по сообщению невозможно понять, схема сломалась или измерение вырождено.
     */
    private fun checkCombination(
        key: String,
        label: String,
        gridSizes: List<Int>,
        errs: List<Double>,
        oneSided: Boolean,
        failures: MutableList<String>,
    ) {
        val expectation = expected[key] ?: run {
            failures += "$label: ключ '$key' отсутствует в таблице ожидаемых порядков. " +
                diagnostics(gridSizes, errs, TRUSTED_ERROR_FLOOR)
            return
        }
        val floor = expectation.trustFloor
        val diagnostics = diagnostics(gridSizes, errs, floor)
        val trustedPairs = trustedOrders(errs, floor)
        val trusted = trustedPairs.lastOrNull()
        when (expectation) {
            is Expected.Saturates -> {
                // ПОРЯДОК ПРОВЕРОК ВАЖЕН. Сначала — величина погрешности, потом — наличие
                // измеримого порядка. При деградации схемы верны ОБА условия сразу (ошибка
                // выросла на порядки И появился измеримый порядок), и сообщить надо про
                // деградацию, а не предлагать обновить таблицу под сломанный код.
                val coarse = errs.first()
                if (!(coarse < SATURATED_MAX_COARSE_ERROR)) {
                    val orderPart = if (trusted == null) {
                        ""
                    } else {
                        " Погрешность перестала упираться в машинную точность: появился измеримый " +
                            "порядок ${fmtOrder(trusted.second)} — признак ДЕГРАДАЦИИ схемы."
                    }
                    failures += "$label: таблица объявляет выход на машинную точность, поэтому " +
                        "проверяется погрешность на грубой сетке: ${fmtError(coarse)} должна быть ниже " +
                        "${fmtError(SATURATED_MAX_COARSE_ERROR)}.$orderPart $diagnostics"
                    return
                }
                if (trusted != null) {
                    failures += "$label: таблица объявляет выход на машинную точность без измеримого " +
                        "порядка, но пригодная пара погрешностей нашлась (p=${fmtOrder(trusted.second)}) при " +
                        "прежнем уровне погрешности. Это изменение поведения — обновите таблицу " +
                        "осознанно. $diagnostics"
                }
            }

            is Expected.Order -> {
                if (trusted == null) {
                    failures += "$label: таблица ожидает порядок ${fmtOrder(expectation.p)}, но ни одной " +
                        "пары погрешностей выше порога доверия ${fmtError(floor)} нет — " +
                        "измерить порядок нечем. $diagnostics"
                    return
                }
                // ПРОВЕРЯЮТСЯ ВСЕ ПРИГОДНЫЕ ПАРЫ, а не только последняя: иначе деградация
                // на грубых сетках проходит молча, если последняя пара в допуске.
                val (lastIndex, _) = trusted
                for ((index, observed) in trustedPairs) {
                    val isLast = index == lastIndex
                    // РАННИЕ ПАРЫ — ПО ОСЛАБЛЕННОМУ ПРАВИЛУ, и это не послабление ради зелёного
                    // теста, а свойство метода: на паре 8->16 схема ещё не вышла на асимптотику.
                    // Замер по 303 ранним парам ИСПРАВНОГО кода: худший недобор — −0.578
                    // (`V/H/mu/nystrom`, пара 8->16), всего три случая хуже −0.5 и ни одного хуже −0.6.
                    // Отсюда [PREASYMPTOTIC_TOLERANCE] = 0.8: вдвое от обычного допуска, с запасом 0.22
                    // до худшего факта. Мутация `kulkarniQuasi` даёт недобор 4...7, то есть ловится.
                    // Рост порядка на ранних парах НЕ проверяется вовсе: у `V/*/xi1/iterKulkarni`
                    // он штатно достигает 4.94 при асимптотических 3.75 — выброс на грубой сетке,
                    // а не улучшение схемы.
                    val tolerance = if (isLast) ORDER_TOLERANCE else PREASYMPTOTIC_TOLERANCE
                    val degraded = observed < expectation.p - tolerance
                    val improved = isLast && observed > expectation.p + ORDER_TOLERANCE
                    if (!degraded && !(improved && !oneSided)) continue
                    val pair = "${gridSizes[index]}->${gridSizes[index + 1]}"
                    val direction = if (degraded) "НИЖЕ" else "ВЫШЕ"
                    val rule = when {
                        !isLast -> "для преасимптотической пары требуется " +
                            "p >= ${fmtOrder(expectation.p - PREASYMPTOTIC_TOLERANCE)}"
                        oneSided -> "требуется p >= ${fmtOrder(expectation.p - ORDER_TOLERANCE)}"
                        else -> "требуется ${fmtOrder(expectation.p)} +- $ORDER_TOLERANCE"
                    }
                    failures += "$label: наблюдаемый порядок ${fmtOrder(observed)} (пара n=$pair) $direction " +
                        "ожидаемого ${fmtOrder(expectation.p)}, $rule. $diagnostics"
                }
            }
        }
    }

    /**
     * Полная диагностика сочетания: погрешности по сеткам и все порядки между ними.
     *
     * @param floor порог доверия ЭТОЙ строки: пометка `[шум]` обязана совпадать с тем,
     *        что фактически отброшено при проверке, иначе диагностика вводит в заблуждение.
     */
    private fun diagnostics(gridSizes: List<Int>, errs: List<Double>, floor: Double): String {
        val ps = orders(errs)
        val errorPart = errs.indices.joinToString(", ") { "n=${gridSizes[it]}: ${fmtError(errs[it])}" }
        val orderPart = if (errs.size < 2) {
            "порядков нет (измерен один уровень)"
        } else {
            (0 until errs.size - 1).joinToString(", ") { i ->
                val trusted = errs[i] >= floor && errs[i + 1] >= floor
                val mark = if (trusted) "" else " [шум]"
                "${gridSizes[i]}->${gridSizes[i + 1]}: ${fmtOrder(ps[i])}$mark"
            }
        }
        return "E_h [$errorPart]; порядки [$orderPart]"
    }

    /**
     * Единый отчёт о падениях.
     *
     * @param subject что именно не сошлось. Параметр, а не константа: span-проверка НЕ
     *        измеряет порядок, и заголовок «порядок не соответствует таблице» был там просто
     *        неверен — читатель искал бы строку в таблице `expected`, которой там нет.
     */
    private fun reportIfAny(failures: List<String>, checked: Int, subject: String) {
        val body = failures.joinToString("\n")
        // Усечение ОТМЕЧАЕТСЯ ЯВНО: молчаливо обрезанный отчёт выглядит как полный,
        // и последние сочетания можно просто не заметить.
        val shown = if (body.length <= FAILURE_REPORT_LIMIT) {
            body
        } else {
            body.take(FAILURE_REPORT_LIMIT) +
                "\n[ВЫВОД УСЕЧЁН: показаны первые $FAILURE_REPORT_LIMIT из ${body.length} символов; " +
                "полный список — в XML-отчёте build/test-results]"
        }
        assertTrue(failures.isEmpty(), "$subject (${failures.size} из $checked сочетаний):\n$shown")
    }

    private fun fmtError(x: Double): String = String.format(Locale.ROOT, "%.4e", x)

    private fun fmtOrder(x: Double): String =
        if (x.isNaN()) "---" else String.format(Locale.ROOT, "%.2f", x)
}
