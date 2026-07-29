package problems.analytic

import numerics.MinimalSplineBasis
import numerics.functionals.FunctionalFamily
import solvers.fredholm.FredholmOperator
import solvers.fredholm.KernelF
import solvers.volterra.KernelV
import solvers.volterra.VolterraOperator

// ============================================================================
// АНАЛИТИЧЕСКИ ТОЧНЫЕ ЗАДАЧИ (сильнейший эталон верификации)
//
// Принципиальное отличие от `problems.fredholm.FredholmProblem` и
// `problems.volterra.VolterraProblem`: там правая часть `f` строится ЧИСЛЕННО,
// квадратурой самого проекта (`rhsExact` вызывает `op.apply`). Из-за этого ошибка
// квадратуры входит и в эталон, и в решение, частично взаимно компенсируясь, —
// проверка получается замкнутой на собственную реализацию.
//
// Здесь и точное решение `u*`, и правая часть `f` выписаны АНАЛИТИЧЕСКИ: интегралы
// взяты вручную, коэффициенты получены решением конечных систем на бумаге. Ни одна
// величина не вычисляется квадратурой проекта, поэтому связь «ошибка эталона —
// ошибка решения» разорвана полностью.
//
// Три независимых источника точных решений:
//  1. ВЫРОЖДЕННЫЕ (сепарабельные) ядра Фредгольма: `K = sum a_i(t) b_i(s)` сводит
//     уравнение к КОНЕЧНОЙ системе линейных уравнений, решаемой в замкнутой форме
//     без какой бы то ни было дискретизации.
//  2. ЯДРА СВЁРТКИ Вольтерры: `K(t,s) = k(t-s)` позволяет применить преобразование
//     Лапласа и получить решение в замкнутой форме.
//  3. МЕТОД ПРОИЗВОДНЫХ РЕШЕНИЙ (method of manufactured solutions): решение
//     задаётся произвольно, правая часть вычисляется подстановкой аналитически.
// ============================================================================

/**
 * Задача Фредгольма II рода `u(t) - \int_a^b K(t,s) u(s) ds = f(t)` с
 * АНАЛИТИЧЕСКИ известными точным решением и правой частью.
 *
 * @param name краткое имя задачи для сообщений тестов.
 * @param kernel ядро `K(t,s)` вместе с частными производными.
 * @param exact точное решение `u*(t)`, выписанное в замкнутой форме.
 * @param exactDeriv первая производная `u*'(t)`.
 * @param exactDeriv2 вторая производная `u*''(t)`.
 * @param rhs правая часть `f(t)` — вычислена АНАЛИТИЧЕСКИ, а не квадратурой.
 * @param rhsDeriv первая производная `f'(t)`, также аналитическая.
 * @param rhsDeriv2 вторая производная `f''(t)`, также аналитическая.
 * @param derivation словесное описание вывода — попадает в сообщения тестов,
 *        чтобы при расхождении сразу было видно, какую выкладку проверять.
 * @param spectralRadius спектральный радиус `rho(K)` интегрального оператора.
 *
 *        Зачем нужен: само уравнение однозначно разрешимо при любом `rho != 1`
 *        (что и подтверждается замкнутой формой решения), но схемы, основанные
 *        на ПРОСТОЙ ИТЕРАЦИИ (`kulkarniQuasi` для квазиинтерполянтов `mu`, `lambda`;
 *        `combinedNystrom`), требуют `rho < 1` и при `rho > 1` расходятся. Это
 *        ограничение МЕТОДА РЕШЕНИЯ, а не задачи, и тесты должны его учитывать
 *        явно, а не прятать за ослабленным допуском.
 *
 *        Значения получены АНАЛИТИЧЕСКИ: для вырожденного ядра ранга `m`
 *        оператор конечномерен, и его ненулевые собственные числа совпадают с
 *        собственными числами матрицы моментов `G_{ij} = \int_0^1 b_i(s) a_j(s) ds`.
 */
class AnalyticFredholmProblem(
    val name: String,
    val kernel: KernelF,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val exactDeriv2: (Double) -> Double,
    val rhs: (Double) -> Double,
    val rhsDeriv: (Double) -> Double,
    val rhsDeriv2: (Double) -> Double,
    val derivation: String,
    val spectralRadius: Double,
) {
    /**
     * Пригодна ли задача для схем, решаемых простой итерацией.
     *
     * Порог взят с запасом (0.9, а не 1.0): вблизи единицы итерация формально
     * сходится, но так медленно, что упирается в предел числа шагов.
     */
    val supportsFixedPointSchemes: Boolean get() = spectralRadius < 0.9

    companion object {
        /**
         * РАНГ 1, пример из постановки задания: `K(t,s) = t*s`, `f(t) = t`.
         *
         * Вывод (полностью на бумаге). Ядро вырождено с единственным слагаемым
         * `a(t) = t`, `b(s) = s`, поэтому
         *
         *     u(t) = f(t) + t * c,   c = \int_0^1 s u(s) ds.
         *
         * Подставляя `u(s) = s + c s = s(1 + c)` в определение `c`:
         *
         *     c = \int_0^1 s * s(1+c) ds = (1+c) \int_0^1 s^2 ds = (1+c)/3,
         *     3c = 1 + c  =>  c = 1/2.
         *
         * Итог: `u*(t) = t (1 + 1/2) = (3/2) t`.
         *
         * Проверка подстановкой: `u* - K u* = (3/2)t - t \int_0^1 s (3/2)s ds
         * = (3/2)t - t (3/2)(1/3) = (3/2)t - (1/2)t = t = f`.
         *
         * ОСОБЕННОСТЬ: решение линейно, то есть лежит в `span{1, t, t^2}` —
         * полиномиальной порождающей системе `phi^B`. Поэтому на базисе `B` метод
         * обязан воспроизводить его с МАШИННОЙ точностью; порядок сходимости здесь
         * не определён. Задача используется как проверка точности, а не порядка.
         */
        val SEPARABLE_RANK1_LINEAR = AnalyticFredholmProblem(
            name = "A-sep1-lin",
            kernel = KernelF(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> 1.5 * t },
            exactDeriv = { 1.5 },
            exactDeriv2 = { 0.0 },
            rhs = { t -> t },
            rhsDeriv = { 1.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=t*s, f=t; c=int_0^1 s u ds=1/2; u*=(3/2)t",
            // Ядро ранга 1: единственное собственное число = int_0^1 s*s ds = 1/3.
            spectralRadius = 1.0 / 3.0,
        )

        /**
         * РАНГ 1 с трансцендентным решением: `K(t,s) = t*s`, `f(t) = e^t`.
         *
         * Вывод. Как и выше, `u(t) = e^t + c t`, где `c = \int_0^1 s u(s) ds`.
         * Подстановка даёт
         *
         *     c = \int_0^1 s e^s ds + c \int_0^1 s^2 ds = 1 + c/3,
         *
         * поскольку `\int_0^1 s e^s ds = [s e^s - e^s]_0^1 = (e - e) - (0 - 1) = 1`.
         * Отсюда `(2/3) c = 1`, то есть `c = 3/2` и
         *
         *     u*(t) = e^t + (3/2) t.
         *
         * Проверка: `K u* = t (\int_0^1 s e^s ds + (3/2)\int_0^1 s^2 ds)
         * = t (1 + 1/2) = (3/2) t`, значит `u* - K u* = e^t = f`.
         *
         * В отличие от [SEPARABLE_RANK1_LINEAR], решение НЕ лежит в `span{1,t,t^2}`,
         * поэтому задача пригодна для измерения порядка сходимости.
         */
        val SEPARABLE_RANK1 = AnalyticFredholmProblem(
            name = "A-sep1",
            kernel = KernelF(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) + 1.5 * t },
            exactDeriv = { t -> Math.exp(t) + 1.5 },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { t -> Math.exp(t) },
            rhsDeriv = { t -> Math.exp(t) },
            rhsDeriv2 = { t -> Math.exp(t) },
            derivation = "K=t*s, f=e^t; c=1+c/3 => c=3/2; u*=e^t+(3/2)t",
            // То же ядро, что и в SEPARABLE_RANK1_LINEAR: rho = 1/3.
            spectralRadius = 1.0 / 3.0,
        )

        /**
         * РАНГ 2: `K(t,s) = 1 + t*s`, `f(t) = e^t`.
         *
         * Вывод. Ядро вырождено с двумя слагаемыми (`a_1 = 1, b_1 = 1` и
         * `a_2 = t, b_2 = s`), поэтому
         *
         *     u(t) = e^t + c_0 + c_1 t,
         *     c_0 = \int_0^1 u(s) ds,   c_1 = \int_0^1 s u(s) ds.
         *
         * Подстановка `u` в определения даёт систему
         *
         *     c_0 = (e-1) + c_0 + c_1/2,          (1)
         *     c_1 = 1     + c_0/2 + c_1/3.        (2)
         *
         * Из (1) сразу `0 = (e-1) + c_1/2`, то есть `c_1 = -2(e-1)`.
         * Подставляя в (2): `-2(e-1) = 1 + c_0/2 - (2/3)(e-1)`, откуда
         * `c_0/2 = -(4/3)(e-1) - 1` и `c_0 = -(8/3)(e-1) - 2`.
         *
         * Итог: `u*(t) = e^t - (8/3)(e-1) - 2 - 2(e-1) t`.
         *
         * Проверка: `K u* = c_0 + c_1 t` по построению коэффициентов, значит
         * `u* - K u* = e^t = f` тождественно.
         *
         * Использованные табличные интегралы: `\int_0^1 e^s ds = e - 1`,
         * `\int_0^1 s e^s ds = 1`.
         */
        val SEPARABLE_RANK2 = AnalyticFredholmProblem(
            name = "A-sep2",
            kernel = KernelF(
                k = { t, s -> 1.0 + t * s },
                kT = { _, s -> s },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) + RANK2_C0 + RANK2_C1 * t },
            exactDeriv = { t -> Math.exp(t) + RANK2_C1 },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { t -> Math.exp(t) },
            rhsDeriv = { t -> Math.exp(t) },
            rhsDeriv2 = { t -> Math.exp(t) },
            derivation = "K=1+t*s, f=e^t; c_1=-2(e-1), c_0=-(8/3)(e-1)-2",
            // Матрица моментов [[1, 1/2], [1/2, 1/3]]; собственные числа — корни
            // lambda^2 - (4/3)lambda + 1/12 = 0, то есть lambda = (2/3) +- sqrt(1/3 - 1/12 + 1/9).
            // Больший корень rho = 2/3 + sqrt(13)/6 ~ 1.2676 > 1.
            spectralRadius = 2.0 / 3.0 + Math.sqrt(13.0) / 6.0,
        )

        /**
         * РАНГ 3: `K(t,s) = 1 + t*s + t^2 s^2`, `f(t) = t^3`.
         *
         * Вывод. Ядро вырождено с тремя слагаемыми, поэтому
         *
         *     u(t) = t^3 + c_0 + c_1 t + c_2 t^2,
         *     c_k = \int_0^1 s^k u(s) ds,  k = 0,1,2.
         *
         * Обозначив моменты правой части `m_k = \int_0^1 s^k * s^3 ds`, то есть
         * `m_0 = 1/4`, `m_1 = 1/5`, `m_2 = 1/6`, и подставив `u` в определения `c_k`,
         * получаем систему (слагаемые `\int_0^1 s^k * s^j ds = 1/(k+j+1)`):
         *
         *     c_0 = m_0 + c_0     + c_1/2 + c_2/3,
         *     c_1 = m_1 + c_0/2   + c_1/3 + c_2/4,
         *     c_2 = m_2 + c_0/3   + c_1/4 + c_2/5.
         *
         * Первое уравнение вырождается в `0 = 1/4 + c_1/2 + c_2/3`, откуда
         * `c_1 = -1/2 - (2/3) c_2`. Подстановка во второе даёт
         * `-c_0/2 - (25/36) c_2 = 8/15`, то есть `c_0 = -16/15 - (25/18) c_2`.
         * Подстановка обоих в третье приводит к `(193/135) c_2 = -113/360`, откуда
         *
         *     c_2 = -339/1544,
         *     c_1 = -273/772,
         *     c_0 = -11761/15440.
         *
         * Итог: `u*(t) = t^3 - (339/1544) t^2 - (273/772) t - 11761/15440`.
         *
         * Контрольные подстановки (выполнены вручную):
         *  - при `t = 0`: `u*(0) = c_0`, `K u*(0) = c_0`, значит `f(0) = 0 = 0^3`;
         *  - при `t = 1`: `u*(1) - K u*(1) = 1 = 1^3`.
         *
         * Решение — многочлен ТРЕТЬЕЙ степени, поэтому оно не лежит в
         * `span{1, t, t^2}` и задача пригодна для измерения порядка сходимости.
         */
        val SEPARABLE_RANK3 = AnalyticFredholmProblem(
            name = "A-sep3",
            kernel = KernelF(
                k = { t, s -> 1.0 + t * s + t * t * s * s },
                kT = { t, s -> s + 2.0 * t * s * s },
                kTT = { _, s -> 2.0 * s * s },
            ),
            exact = { t -> t * t * t + RANK3_C2 * t * t + RANK3_C1 * t + RANK3_C0 },
            exactDeriv = { t -> 3.0 * t * t + 2.0 * RANK3_C2 * t + RANK3_C1 },
            exactDeriv2 = { t -> 6.0 * t + 2.0 * RANK3_C2 },
            rhs = { t -> t * t * t },
            rhsDeriv = { t -> 3.0 * t * t },
            rhsDeriv2 = { t -> 6.0 * t },
            derivation = "K=1+ts+t^2s^2, f=t^3; c_2=-339/1544, c_1=-273/772, c_0=-11761/15440",
            // Матрица моментов — матрица Гильберта 3x3; её наибольшее собственное
            // число — известная табличная величина ~1.40832 > 1.
            spectralRadius = 1.4083189271236535,
        )

        /**
         * МЕТОД ПРОИЗВОДНЫХ РЕШЕНИЙ (MMS) для уравнения Фредгольма:
         * ядро `K(t,s) = 1/(1+t+s)`, ЗАДАННОЕ решение `u*(t) = t^3`.
         *
         * Здесь порядок обратный: решение выбирается произвольно, а правая часть
         * вычисляется подстановкой. Ключевое требование задания — взять интеграл
         * АНАЛИТИЧЕСКИ, а не квадратурой.
         *
         * Вычисление `\int_0^1 s^3/(a+s) ds` при `a = 1+t`. Деление многочленов
         * (тождество `s^3 = (s+a)(s^2 - a s + a^2) - a^3`) даёт
         *
         *     s^3/(a+s) = s^2 - a s + a^2 - a^3/(a+s),
         *
         * откуда
         *
         *     \int_0^1 s^3/(a+s) ds = 1/3 - a/2 + a^2 - a^3 ln((a+1)/a).
         *
         * Следовательно
         *
         *     f(t) = t^3 - 1/3 + a/2 - a^2 + a^3 ln((a+1)/a),   a = 1 + t.
         *
         * Производные (получены дифференцированием по `t`, при `da/dt = 1` и
         * `d/da ln((a+1)/a) = -1/(a(a+1))`):
         *
         *     f'(t)  = 3t^2 + 1/2 - 2a + 3a^2 ln((a+1)/a) - a^2/(a+1),
         *     f''(t) = 6t - 2 + 6a ln((a+1)/a) - 3a/(a+1) - (a^2+2a)/(a+1)^2.
         *
         * Контроль при `t = 0` (`a = 1`): `f(0) = -1/3 + 1/2 - 1 + ln 2 ~ -0.14019`,
         * что совпадает с `-\int_0^1 s^3/(1+s) ds = -(1/3 - 1/2 + 1 - ln 2)`.
         *
         * В отличие от сепарабельных задач, ядро здесь НЕ вырождено, поэтому проверка
         * задействует иной путь в сборке матрицы.
         */
        val MANUFACTURED = AnalyticFredholmProblem(
            name = "A-mms-F",
            kernel = KernelF(
                k = { t, s -> 1.0 / (1.0 + t + s) },
                kT = { t, s -> -1.0 / ((1.0 + t + s) * (1.0 + t + s)) },
                kTT = { t, s -> 2.0 / ((1.0 + t + s) * (1.0 + t + s) * (1.0 + t + s)) },
            ),
            exact = { t -> t * t * t },
            exactDeriv = { t -> 3.0 * t * t },
            exactDeriv2 = { t -> 6.0 * t },
            rhs = { t ->
                val a = 1.0 + t
                t * t * t - 1.0 / 3.0 + a / 2.0 - a * a + a * a * a * Math.log((a + 1.0) / a)
            },
            rhsDeriv = { t ->
                val a = 1.0 + t
                3.0 * t * t + 0.5 - 2.0 * a + 3.0 * a * a * Math.log((a + 1.0) / a) - a * a / (a + 1.0)
            },
            rhsDeriv2 = { t ->
                val a = 1.0 + t
                6.0 * t - 2.0 + 6.0 * a * Math.log((a + 1.0) / a) -
                    3.0 * a / (a + 1.0) - (a * a + 2.0 * a) / ((a + 1.0) * (a + 1.0))
            },
            derivation = "MMS: K=1/(1+t+s), u*=t^3; f=t^3-1/3+a/2-a^2+a^3 ln((a+1)/a), a=1+t",
            // Оценка сверху нормой в C[0,1]: rho <= max_t int_0^1 ds/(1+t+s) = ln 2 ~ 0.693 < 1.
            spectralRadius = Math.log(2.0),
        )

        /** Все аналитические задачи Фредгольма. */
        val ALL = listOf(
            SEPARABLE_RANK1_LINEAR,
            SEPARABLE_RANK1,
            SEPARABLE_RANK2,
            SEPARABLE_RANK3,
            MANUFACTURED,
        )

        /**
         * Задачи, пригодные для измерения ПОРЯДКА сходимости: их решения не лежат
         * в `span{1, t, t^2}` полиномиальной порождающей системы.
         */
        val CONVERGENT = listOf(SEPARABLE_RANK1, SEPARABLE_RANK2, SEPARABLE_RANK3, MANUFACTURED)

        /**
         * Задачи с `rho(K) < 1` — единственные, на которых применимы схемы,
         * реализованные ПРОСТОЙ ИТЕРАЦИЕЙ (схема Кулкарни для квазиинтерполянтов
         * `mu` и `lambda`, комбинированный Nyström).
         *
         * Задачи [SEPARABLE_RANK2] и [SEPARABLE_RANK3] сюда НЕ входят: у них `rho > 1`.
         * Подчёркнем — сами задачи корректны и однозначно разрешимы (что и доказывает
         * замкнутая форма решения), а схемы на основе ПРЯМОГО решения СЛАУ (`base`,
         * `sloan`, `kulkarni` для проекторов) решают их без каких-либо проблем.
         */
        val FIXED_POINT_SAFE = ALL.filter { it.supportsFixedPointSchemes }

        // --- Точные константы, посчитанные вручную (см. KDoc соответствующих задач) ---

        /** `c_1 = -2(e-1)` для задачи ранга 2. */
        private val RANK2_C1 = -2.0 * (Math.E - 1.0)

        /** `c_0 = -(8/3)(e-1) - 2` для задачи ранга 2. */
        private val RANK2_C0 = -(8.0 / 3.0) * (Math.E - 1.0) - 2.0

        /** `c_2 = -339/1544` для задачи ранга 3. */
        private const val RANK3_C2 = -339.0 / 1544.0

        /** `c_1 = -273/772` для задачи ранга 3. */
        private const val RANK3_C1 = -273.0 / 772.0

        /** `c_0 = -11761/15440` для задачи ранга 3. */
        private const val RANK3_C0 = -11761.0 / 15440.0
    }
}

/**
 * Задача Вольтерры II рода `u(t) - \int_a^t K(t,s) u(s) ds = f(t)` с АНАЛИТИЧЕСКИ
 * известными точным решением и правой частью.
 *
 * Смысл полей совпадает с [AnalyticFredholmProblem]; отличается только оператор
 * (переменный верхний предел интегрирования).
 *
 * ПОЧЕМУ ЗДЕСЬ НЕТ ПОЛЯ `spectralRadius` (в отличие от [AnalyticFredholmProblem]).
 * Оператор Вольтерры с ограниченным ядром КВАЗИНИЛЬПОТЕНТЕН: его спектральный
 * радиус равен нулю при ЛЮБОМ ядре (оценка `|V^m u| <= (M(b-a))^m/m! * |u|`).
 * Поэтому простая итерация сходится всегда, и ограничение, существенное для
 * задач Фредгольма, здесь отсутствует по построению.
 */
class AnalyticVolterraProblem(
    val name: String,
    val kernel: KernelV,
    val exact: (Double) -> Double,
    val exactDeriv: (Double) -> Double,
    val exactDeriv2: (Double) -> Double,
    val rhs: (Double) -> Double,
    val rhsDeriv: (Double) -> Double,
    val rhsDeriv2: (Double) -> Double,
    val derivation: String,
) {
    companion object {
        /**
         * ЯДРО СВЁРТКИ, пример из постановки задания: `K(t,s) = 1`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Ядро зависит только от разности,
         * `k(tau) = 1`, поэтому уравнение `u - k * u = f` (звёздочка — свёртка)
         * переходит в алгебраическое:
         *
         *     U(p) - \hat k(p) U(p) = F(p),   U(p) = F(p) / (1 - \hat k(p)).
         *
         * Здесь `\hat k(p) = 1/p` и `F(p) = 1/p`, значит
         *
         *     U(p) = (1/p) / (1 - 1/p) = (1/p) * p/(p-1) = 1/(p-1),
         *
         * и обратное преобразование даёт `u*(t) = e^t`.
         *
         * Прямая проверка: `\int_0^t e^s ds = e^t - 1`, поэтому
         * `u* - V u* = e^t - (e^t - 1) = 1 = f`.
         */
        val CONVOLUTION_CONST = AnalyticVolterraProblem(
            name = "A-conv-1",
            kernel = KernelV(
                k = { _, _ -> 1.0 },
                kT = { _, _ -> 0.0 },
                kS = { _, _ -> 0.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.exp(t) },
            exactDeriv = { t -> Math.exp(t) },
            exactDeriv2 = { t -> Math.exp(t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=1, f=1; U=(1/p)/(1-1/p)=1/(p-1); u*=e^t",
        )

        /**
         * ЯДРО СВЁРТКИ `K(t,s) = t - s`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Здесь `k(tau) = tau`, поэтому
         * `\hat k(p) = 1/p^2`, а `F(p) = 1/p`. Тогда
         *
         *     U(p) = (1/p) / (1 - 1/p^2) = (1/p) * p^2/(p^2 - 1) = p/(p^2 - 1),
         *
         * что является образом гиперболического косинуса: `u*(t) = cosh t`.
         *
         * Прямая проверка (интегрирование по частям):
         *
         *     \int_0^t (t-s) cosh s ds = t sinh t - [s sinh s - cosh s]_0^t
         *                              = t sinh t - t sinh t + cosh t - 1
         *                              = cosh t - 1,
         *
         * поэтому `u* - V u* = cosh t - (cosh t - 1) = 1 = f`.
         */
        val CONVOLUTION_LINEAR = AnalyticVolterraProblem(
            name = "A-conv-lin",
            kernel = KernelV(
                k = { t, s -> t - s },
                kT = { _, _ -> 1.0 },
                kS = { _, _ -> -1.0 },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cosh(t) },
            exactDeriv = { t -> Math.sinh(t) },
            exactDeriv2 = { t -> Math.cosh(t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=t-s, f=1; U=(1/p)/(1-1/p^2)=p/(p^2-1); u*=cosh t",
        )

        /**
         * ЯДРО СВЁРТКИ `K(t,s) = e^{t-s}`, `f(t) = 1`.
         *
         * Вывод преобразованием Лапласа. Здесь `k(tau) = e^{tau}`, поэтому
         * `\hat k(p) = 1/(p-1)`, а `F(p) = 1/p`. Тогда
         *
         *     U(p) = (1/p) / (1 - 1/(p-1)) = (1/p) * (p-1)/(p-2) = (p-1)/(p(p-2)).
         *
         * Разложение на простейшие дроби: `(p-1)/(p(p-2)) = A/p + B/(p-2)` с
         * `A(p-2) + Bp = p - 1`. При `p = 0` получаем `-2A = -1`, то есть `A = 1/2`;
         * при `p = 2` получаем `2B = 1`, то есть `B = 1/2`. Значит
         *
         *     U(p) = (1/2)/p + (1/2)/(p-2)   =>   u*(t) = (1 + e^{2t}) / 2.
         *
         * Прямая проверка:
         *
         *     \int_0^t e^{t-s} (1+e^{2s})/2 ds = (e^t/2) \int_0^t (e^{-s} + e^{s}) ds
         *                                      = (e^t/2)(e^t - e^{-t}) = (e^{2t} - 1)/2,
         *
         * поэтому `u* - V u* = (1 + e^{2t})/2 - (e^{2t} - 1)/2 = 1 = f`.
         */
        val CONVOLUTION_EXP = AnalyticVolterraProblem(
            name = "A-conv-exp",
            kernel = KernelV(
                k = { t, s -> Math.exp(t - s) },
                kT = { t, s -> Math.exp(t - s) },
                kS = { t, s -> -Math.exp(t - s) },
                kTT = { t, s -> Math.exp(t - s) },
            ),
            exact = { t -> 0.5 * (1.0 + Math.exp(2.0 * t)) },
            exactDeriv = { t -> Math.exp(2.0 * t) },
            exactDeriv2 = { t -> 2.0 * Math.exp(2.0 * t) },
            rhs = { 1.0 },
            rhsDeriv = { 0.0 },
            rhsDeriv2 = { 0.0 },
            derivation = "K=e^{t-s}, f=1; U=(p-1)/(p(p-2)); u*=(1+e^{2t})/2",
        )

        /**
         * МЕТОД ПРОИЗВОДНЫХ РЕШЕНИЙ (MMS) для уравнения Вольтерры:
         * ядро `K(t,s) = t*s`, ЗАДАННОЕ решение `u*(t) = cos t`.
         *
         * Ядро НЕ является ядром свёртки, поэтому путь вывода полностью независим
         * от лапласовых задач выше.
         *
         * Вычисление образа (интеграл взят по частям):
         *
         *     (V u*)(t) = t \int_0^t s cos s ds = t [s sin s + cos s]_0^t
         *               = t (t sin t + cos t - 1)
         *               = t^2 sin t + t cos t - t.
         *
         * Отсюда правая часть
         *
         *     f(t) = cos t - t^2 sin t - t cos t + t.
         *
         * Производные (прямое дифференцирование):
         *
         *     f'(t)  = -sin t - t sin t - t^2 cos t - cos t + 1,
         *     f''(t) = -cos t - 3 t cos t + t^2 sin t.
         *
         * Решение `cos t` не лежит в `span{1, t, t^2}`, поэтому задача пригодна
         * для измерения порядка сходимости.
         */
        val MANUFACTURED = AnalyticVolterraProblem(
            name = "A-mms-V",
            kernel = KernelV(
                k = { t, s -> t * s },
                kT = { _, s -> s },
                kS = { t, _ -> t },
                kTT = { _, _ -> 0.0 },
            ),
            exact = { t -> Math.cos(t) },
            exactDeriv = { t -> -Math.sin(t) },
            exactDeriv2 = { t -> -Math.cos(t) },
            rhs = { t -> Math.cos(t) - t * t * Math.sin(t) - t * Math.cos(t) + t },
            rhsDeriv = { t ->
                -Math.sin(t) - t * Math.sin(t) - t * t * Math.cos(t) - Math.cos(t) + 1.0
            },
            rhsDeriv2 = { t -> -Math.cos(t) - 3.0 * t * Math.cos(t) + t * t * Math.sin(t) },
            derivation = "MMS: K=t*s, u*=cos t; f=cos t - t^2 sin t - t cos t + t",
        )

        /** Все аналитические задачи Вольтерры. */
        val ALL = listOf(CONVOLUTION_CONST, CONVOLUTION_LINEAR, CONVOLUTION_EXP, MANUFACTURED)
    }
}

/**
 * Создаёт решатель уравнения Фредгольма II рода для аналитической задачи.
 *
 * В отличие от `problems.fredholm.secondKindSolver`, правая часть и её производные
 * берутся из АНАЛИТИЧЕСКИХ формул задачи и не проходят через квадратуру проекта.
 *
 * @param throwOnDivergence политика обработки расходимости итерационных схем;
 *        значение `false` нужно тестам, изучающим САМУ расходимость на задачах
 *        со спектральным радиусом больше единицы (см. [AnalyticFredholmProblem.supportsFixedPointSchemes]).
 */
fun analyticFredholmSolver(
    problem: AnalyticFredholmProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: FredholmOperator,
    throwOnDivergence: Boolean = true,
): solvers.fredholm.SecondKindSolver = solvers.fredholm.SecondKindSolver(
    basis, funcs, op, cL = 1.0,
    fEff = problem.rhs,
    fEffDeriv = problem.rhsDeriv,
    fEffDeriv2 = problem.rhsDeriv2,
    throwOnDivergence = throwOnDivergence,
)

/**
 * Создаёт решатель уравнения Вольтерры II рода для аналитической задачи.
 *
 * Правая часть и её производные — аналитические, без обращения к квадратуре проекта.
 */
fun analyticVolterraSolver(
    problem: AnalyticVolterraProblem,
    basis: MinimalSplineBasis,
    funcs: FunctionalFamily,
    op: VolterraOperator,
): solvers.volterra.SecondKindSolver = solvers.volterra.SecondKindSolver(
    basis, funcs, op, cL = 1.0,
    fEff = problem.rhs,
    fEffDeriv = problem.rhsDeriv,
    fEffDeriv2 = problem.rhsDeriv2,
)
