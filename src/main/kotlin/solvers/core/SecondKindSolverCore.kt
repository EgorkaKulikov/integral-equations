package solvers.core

import kotlin.math.abs
import numerics.*
import numerics.functionals.*
import solvers.core.SecondKindDefaults.KULKARNI_QUASI_MAX_ITERATIONS
import solvers.core.SecondKindDefaults.KULKARNI_QUASI_TOLERANCE

/**
 * Образ `L u` вместе с двумя его производными — ровно то, что функционал `chi_j`
 * получает на вход при сборке матриц `M` и `M2`.
 *
 * Тройка возвращается ОДНИМ объектом, а не тремя независимыми вызовами: у Вольтерры
 * все три замыкания строятся из одного подготовленного операнда, и разделение заставило бы
 * создавать операнд (а вместе с ним — кэш подынтегральной функции) заново,
 * изменив число обращений к ядру.
 */
class ImageTriple(
    val value: (Double) -> Double,
    val deriv: (Double) -> Double,
    val deriv2: (Double) -> Double,
)

/**
 * ОБЩЕЕ ЯДРО ЛИНЕЙНЫХ РЕШАТЕЛЕЙ УРАВНЕНИЯ II РОДА `u - L u = f`.
 *
 * Здесь собрано всё, что у решателей Фредгольма и Вольтерры совпадало ДОСЛОВНО либо
 * различалось только способом применения оператора `L`: сборка матриц `M`, `M2`,
 * векторов `g`, `d`, базовая коллокация, итерация Слоана и всё семейство схем Кулкарни.
 *
 * Матрицы дискретной задачи:
 *   M_{j,i}  = chi_j(L omega_i),  M2_{j,i} = chi_j(L(L omega_i)),
 *   g_j      = chi_j(f),          d_j      = chi_j(L f).
 *
 * ### Что СОЗНАТЕЛЬНО не переехало сюда
 *
 * Всё семейство Nyström (`nystromSupport`, `nystromEval`, `nystrom`,
 * `iteratedNystrom`, `combinedNystrom`, `iteratedCombinedNystrom`) осталось
 * в наследниках. Причина не в объёме, а в структуре: у Фредгольма веса `b_r`
 * НЕ зависят от `t` и вычисляются один раз, у Вольтерры они пересчитываются
 * на каждом `t` из-за причинного усечения носителя `[a, t]`. Из-за этого
 * различаются типы результата (`Pair` против выделенного класса), сигнатуры
 * `nystromEval`, разделение «сборка матрицы / решение системы», а у
 * `combinedNystrom` — ещё и множество точек, на котором меряется критерий
 * останова. Слияние потребовало бы менять порядок операций, то есть числа.
 *
 * ### Абстракция «подготовленный операнд» ([Operand])
 *
 * Единственное содержательное различие общей части — КАК представляется функция,
 * к которой многократно применяется `L`:
 *
 * * Фредгольм: пределы интегрирования постоянны, поэтому существует фиксированный
 *   набор глобальных гауссовых узлов, и операнд — это массив значений функции
 *   в этих узлах (`DoubleArray`); повторное применение `L` — свёртка с ядром
 *   по уже готовым значениям.
 * * Вольтерра: верхний предел `t` переменный, фиксированного набора узлов НЕ
 *   существует, и операнд — сама функция (замыкание); применение `L` идёт через
 *   квадратуру по усечённому отрезку.
 *
 * Различие выражено параметром типа [Operand] и парой [prepare] / [image], а НЕ
 * булевым флагом «это Вольтерра»: флаг пришлось бы проверять в каждом методе,
 * и добавление третьего уравнения потребовало бы править все ветвления.
 * Параметр типа выбран вместо интерфейса-обёртки сознательно: `DoubleArray`
 * и `(Double) -> Double` — оба ссылочные типы, поэтому упаковки нет, а обёртка
 * добавила бы по объекту на каждый подготовленный операнд в горячем пути сборки
 * матриц.
 *
 * ### Порядок инициализации
 *
 * Инициализаторы БАЗОВОГО класса выполняются РАНЬШЕ инициализаторов производного,
 * поэтому здесь нет ни одного поля, значение которого читается через
 * `abstract`/`open` член: [grid], [n] и [dim] вычисляются исключительно из
 * параметров первичного конструктора. Все точки расширения ([checkPoints],
 * [prepare], [image], …) вызываются ТОЛЬКО из методов, то есть уже после того,
 * как конструктор наследника отработал. Нарушение этого правила даёт молчаливый
 * дефект: например, `dim` увидел бы `n = 0` и стал бы равен `2`, все матрицы
 * стали бы `2x2`, а система — разрешимой и бессмысленной.
 *
 * @param basis базис минимальных сплайнов.
 * @param funcs семейство функционалов chi_j.
 * @param cL множитель оператора: `c_L = 1` для уравнения II рода;
 *        `c_L = -1/alpha` при регуляризации уравнения I рода по Вазвазу.
 * @param rhs правая часть `f` вместе с двумя производными — задаётся явно, чтобы
 *        решатель переиспользовался решателями уравнений I рода с ДРУГОЙ (эффективной)
 *        правой частью. Одним объектом, а не тремя параметрами: см. [RhsWithDerivatives].
 * @param throwOnDivergence поведение ИТЕРАЦИОННЫХ схем ([kulkarni] для
 *        квазиинтерполянтов, `combinedNystrom` в наследниках) при недостижении
 *        сходимости: `true` (по умолчанию) — исключение, `false` — результат с
 *        `converged = false` и достигнутой невязкой в [SolutionFunc.residual].
 *        На прямые схемы не влияет. Параметр задан на уровне решателя, а не
 *        каждого метода: это политика обработки ошибок, а не свойство схемы.
 */
abstract class SecondKindSolverCore<Operand>(
    val basis: MinimalSplineBasis,
    val funcs: FunctionalFamily,
    val cL: Double,
    rhs: RhsWithDerivatives,
    val throwOnDivergence: Boolean,
    val ctx: NumericsContext = NumericsContext.default(),
) {
    init {
        // Семейство функционалов решает СЛАУ при построении — тем же бэкендом, что и решатель.
        NumericsContext.requireSame("SecondKindSolverCore", ctx, "funcs", funcs.ctx)
    }

    // Тройка РАСПАКОВЫВАЕТСЯ в собственные поля один раз, в конструкторе.
    // Так все чтения в горячем пути (сборка M/M2, итерации Слоана и Кулкарни)
    // остаются ровно тем же одним разыменованием поля, что и до объединения
    // параметров: `rhs.value(t)` добавлял бы второе разыменование на КАЖДЫЙ вызов.
    //
    // Сам `rhs` — НЕ свойство, а просто параметр конструктора: иначе на три функции
    // приходилось бы шесть публичных членов (`rhs.value` и `fEff` — одно и то же),
    // а два способа получить одно и то же приглашают к ошибке. Как параметр он ещё и
    // не занимает поля в объекте.

    /** Правая часть `f(t)`. */
    val fEff: (Double) -> Double = rhs.value

    /** Первая производная правой части `f'(t)`. */
    val fEffDeriv: (Double) -> Double = rhs.deriv

    /** Вторая производная правой части `f''(t)`. */
    val fEffDeriv2: (Double) -> Double = rhs.deriv2

    val grid = basis.grid
    val n = grid.n
    val dim = n + 2

    // ==== Точки расширения ==================================================

    /** Название уравнения для диагностических сообщений («Фредгольм» / «Вольтерра»). */
    protected abstract val equationName: String

    /**
     * Пояснение к сообщению о расходимости схемы Кулкарни для квазиинтерполянтов.
     *
     * Вынесено в точку расширения, а не собрано из общего шаблона, потому что
     * тексты у двух решателей исторически различаются: у Фредгольма он длиннее
     * (называет условие сжатия явно). Диагностика — часть наблюдаемого поведения,
     * поэтому объединение текстов здесь было бы изменением поведения.
     */
    protected abstract val kulkarniQuasiHint: String

    /**
     * Точки, на которых меряется критерий останова итерационных схем.
     *
     * У Фредгольма это глобальные гауссовы узлы оператора: итерант там уже вычислен
     * как побочный результат подготовки операнда, поэтому критерий бесплатен.
     * У Вольтерры фиксированных узлов нет, и берётся отдельная равномерная выборка
     * `4n+1` точек. Множества РАЗНЫЕ, и это существенно: один и тот же допуск
     * `1e-13`, применённый к другому множеству, даёт другое число итераций.
     */
    protected abstract val checkPoints: DoubleArray

    /**
     * Готовит функцию `u` к многократному применению оператора `L`.
     *
     * Фредгольм: значения `u` в глобальных гауссовых узлах. Вольтерра: сама `u`
     * (подготовка невозможна — узлы зависят от `t`).
     */
    protected abstract fun prepare(u: (Double) -> Double): Operand

    /** `(L u)(t)` по подготовленному операнду. */
    protected abstract fun image(o: Operand): (Double) -> Double

    /** `(L u)'(t)` по подготовленному операнду. */
    protected abstract fun imageDeriv(o: Operand): (Double) -> Double

    /**
     * `(L u)''(t)` по подготовленному операнду.
     *
     * @param uD производная САМОГО операнда. У Вольтерры она обязательна: из-за
     *        переменного верхнего предела вторая производная содержит член Лейбница
     *        `K(t,t) u'(t)`. У Фредгольма пределы постоянны, такого члена нет,
     *        и аргумент не читается — см. переопределение в решателе Фредгольма.
     */
    protected abstract fun imageDeriv2(o: Operand, uD: (Double) -> Double): (Double) -> Double

    /**
     * Значения итеранта в [checkPoints] — то, по чему мерится критерий останова.
     *
     * Точка расширения, а не просто вычисление по формуле: у Фредгольма контрольные
     * точки СОВПАДАЮТ с узлами подготовки операнда, поэтому значения уже посчитаны
     * и достаточно вернуть сам операнд. Общая формула удвоила бы там стоимость
     * критерия останова на КАЖДОЙ итерации: `u` там не константа, а результат
     * применения оператора.
     */
    protected open fun checkValues(u: (Double) -> Double, o: Operand): DoubleArray =
        DoubleArray(checkPoints.size) { u(checkPoints[it]) }

    /** `(\mathcal K u)(t)` — применение оператора к НЕподготовленной функции. */
    protected abstract fun applyOperator(t: Double, u: (Double) -> Double): Double

    /** `d/dt (\mathcal K u)(t)`. */
    protected abstract fun applyOperatorDeriv(t: Double, u: (Double) -> Double): Double

    /**
     * `d^2/dt^2 (\mathcal K u)(t)`.
     *
     * @param uD производная `u`; нужна только Вольтерре (член Лейбница), у Фредгольма
     *        игнорируется. Арность выбрана по «широкому» варианту: сузить её нельзя,
     *        а лишний аргумент у Фредгольма ничего не стоит.
     */
    protected abstract fun applyOperatorDeriv2(
        t: Double,
        u: (Double) -> Double,
        uD: (Double) -> Double,
    ): Double

    /** `L omega_i` и её производные для столбца `i` матрицы `M`. */
    protected abstract fun omegaImages(i: Int): ImageTriple

    /** `L(L omega_i)` и её производные для столбца `i` матрицы `M2`. */
    protected abstract fun doubleOmegaImages(i: Int): ImageTriple

    // ==== Общая часть =======================================================

    /** chi_j(g) по значениям g, g' и g'' (обёртка). */
    private fun chiOf(
        g: (Double) -> Double,
        gD: (Double) -> Double,
        gDD: (Double) -> Double = { 0.0 },
    ): DoubleArray = DoubleArray(dim) { funcs.chi(it - 2).apply(g, gD, gDD) }

    /**
     * Сборка матрицы `chi_j(<образ>_i)` по столбцам с последующим транспонированием.
     *
     * Транспонирование вынесено из параллельной части сознательно: столбцы
     * независимы по `i`, а построчная запись в общую матрицу из потоков потребовала
     * бы синхронизации. Порядок обхода при копировании сохранён дословно.
     */
    private fun assembleChiMatrix(images: (Int) -> ImageTriple): Array<DoubleArray> {
        // Столбцы M независимы по i; cols[i] = столбец i.
        val cols = ParallelAssembly.assembleRows(dim, dim, ctx.parallel) { i ->
            val im = images(i)
            DoubleArray(dim) { j -> funcs.chi(j - 2).apply(im.value, im.deriv, im.deriv2) }
        }
        val m = LinearAlgebra.zeros(dim, dim)
        for (i in 0 until dim) for (j in 0 until dim) m[j][i] = cols[i][j]
        return m
    }

    /** Матрица M_{j,i} = chi_j(L omega_i). Для xi учитывается (L omega_i)', для xi^<0> и (L omega_i)''. */
    fun matrixM(): Array<DoubleArray> = assembleChiMatrix { i -> omegaImages(i) }

    /** Матрица M2_{j,i} = chi_j(L(L omega_i)) (двойное применение L). */
    fun matrixM2(): Array<DoubleArray> = assembleChiMatrix { i -> doubleOmegaImages(i) }

    /** g_j = chi_j(f). */
    fun vectorG(): DoubleArray = chiOf(fEff, fEffDeriv, fEffDeriv2)

    /**
     * d_j = chi_j(L f).
     *
     * ВАЖНО: образ правой части строится через ПРЯМОЕ применение оператора
     * ([applyOperator]), а не через [prepare] + [image]. У Вольтерры это означает,
     * что кэш подынтегральной функции здесь НЕ создаётся — правая часть применяется
     * ровно по одному разу на каждый функционал, и кэш только добавил бы аллокаций.
     * Факт зафиксирован в KDoc `VolterraOperator.IntegrandCache` и менять его нельзя.
     */
    fun vectorD(): DoubleArray {
        val rhsImage = { t: Double -> cL * applyOperator(t) { s -> fEff(s) } }
        val rhsImageDeriv = { t: Double -> cL * applyOperatorDeriv(t) { s -> fEff(s) } }
        // Вторая производная образа правой части требует и f, и f' (член Лейбница у Вольтерры).
        val rhsImageDeriv2 = { t: Double -> cL * applyOperatorDeriv2(t, fEff, fEffDeriv) }
        return chiOf(rhsImage, rhsImageDeriv, rhsImageDeriv2)
    }

    /** Базовая схема: (I - M) c = g. */
    fun solveBaseCoeffs(): DoubleArray {
        val m = matrixM()
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) { for (c in 0 until dim) a[r][c] = -m[r][c]; a[r][r] += 1.0 }
        return LinearAlgebra.solve(a, vectorG(), ctx.backend)
    }

    fun base(): SolutionFunc {
        val c = solveBaseCoeffs()
        return SolutionFunc(eval = { t -> basis.evalSpline(c, t) })
    }

    /** Слоан: ~u_h(t) = f(t) + (L u_h)(t). u_h — сплайн, L применяется к подготовленному операнду. */
    fun sloan(): SolutionFunc {
        val c = solveBaseCoeffs()
        val splineImage = image(prepare { s -> basis.evalSpline(c, s) })
        return SolutionFunc(eval = { t -> fEff(t) + splineImage(t) })
    }

    /**
     * Кулкарни для проекторов (theta, xi): (I - M - M2 + M^2) c = (I - M) g + d;
     * u_h^K = y_h + (I - P_chi)[f + L y_h]. Для квазиинтерполянтов (mu, lambda) без
     * редукции — прямая итерация конечноранговым U^K_h [численное наблюдение].
     */
    fun kulkarni(): SolutionFunc {
        return if (funcs.isProjector) kulkarniProjector() else kulkarniQuasi()
    }

    private fun kulkarniProjector(): SolutionFunc {
        val m = matrixM(); val m2 = matrixM2(); val g = vectorG(); val d = vectorD()
        val mm = LinearAlgebra.matMat(m, m, ctx.backend)
        // A = I - M - M2 + M^2
        val a = LinearAlgebra.zeros(dim, dim)
        for (r in 0 until dim) {
            for (c in 0 until dim) a[r][c] = -m[r][c] - m2[r][c] + mm[r][c]
            a[r][r] += 1.0
        }
        // rhs = (I - M) g + d
        val mg = LinearAlgebra.matVec(m, g, ctx.backend)
        val rhs = DoubleArray(dim) { g[it] - mg[it] + d[it] }
        val c = LinearAlgebra.solve(a, rhs, ctx.backend) // коэффициенты y_h
        // u_h^K = y_h + (I - P_chi)[f + L y_h]; (I - P_chi)w = w - P_chi w.
        val yh = { s: Double -> basis.evalSpline(c, s) }
        val yhD = { s: Double -> basis.evalSplineDeriv(c, s) }
        val yhOperand = prepare(yh)
        val yhImage = image(yhOperand)
        val yhImageDeriv = imageDeriv(yhOperand)
        val yhImageDeriv2 = imageDeriv2(yhOperand, yhD)
        val wFun = { t: Double -> fEff(t) + yhImage(t) }
        val wDFun = { t: Double -> fEffDeriv(t) + yhImageDeriv(t) }
        val wDDFun = { t: Double -> fEffDeriv2(t) + yhImageDeriv2(t) }
        val pwCoeffs = funcs.projectorCoeffs(wFun, wDFun, wDDFun)
        return SolutionFunc(eval = { t -> basis.evalSpline(c, t) + (wFun(t) - basis.evalSpline(pwCoeffs, t)) })
    }

    /**
     * Кулкарни для mu, lambda: итерация u^{(m+1)} = f + U^K_h u^{(m)},
     * U^K_h u = P_chi(L u) + L(P_chi u) - P_chi(L(P_chi u)).
     * [численное наблюдение]: разрешимость/сходимость не гарантированы (нет P^2=P).
     *
     * Итерант хранится как НЕПРЕРЫВНАЯ функция u^{(m)}(t): реконструкция того же
     * порядка, что и базис (через basis.evalSpline и точную квадратуру оператора),
     * без понижающей кусочно-линейной интерполяции узловых значений. Ранее итерант
     * хранился значениями на равномерной выборке с кусочно-линейным восстановлением,
     * что ограничивало точность величиной O(h_sample^2) НЕЗАВИСИМО от порядка базиса.
     *
     * Контрольные точки ([checkPoints]) участвуют ТОЛЬКО в критерии останова,
     * в самой итерации они не используются.
     */
    private fun kulkarniQuasi(): SolutionFunc {
        var uFun: (Double) -> Double = { t -> fEff(t) }
        var uOperand = prepare(uFun)
        var uAtCheck = checkValues(uFun, uOperand)
        var converged = false
        var performedIterations = 0
        var lastDiff = Double.NaN
        for (iter in 0 until KULKARNI_QUASI_MAX_ITERATIONS) {
            val curFun = uFun
            val curOperand = uOperand
            // P_chi u: коэффициенты chi_j(u) по непрерывной u^{(m)} (сохраняет порядок).
            val pc = funcs.projectorCoeffs(curFun)
            val pcFun = { s: Double -> basis.evalSpline(pc, s) }
            val luFun = image(curOperand)                 // L u
            val pLu = funcs.projectorCoeffs(luFun)        // P_chi(L u)
            val lpu = image(prepare(pcFun))               // L(P_chi u)
            val pLPu = funcs.projectorCoeffs(lpu)         // P_chi(L(P_chi u))
            // Непрерывная реконструкция следующего итеранта u^{(m+1)}(t).
            val nextFun = { t: Double ->
                fEff(t) + basis.evalSpline(pLu, t) + lpu(t) - basis.evalSpline(pLPu, t)
            }
            val nextOperand = prepare(nextFun)
            val nextAtCheck = checkValues(nextFun, nextOperand)
            var diff = 0.0
            for (k in nextAtCheck.indices) diff = maxOf(diff, abs(nextAtCheck[k] - uAtCheck[k]))
            uFun = nextFun
            uOperand = nextOperand
            uAtCheck = nextAtCheck
            performedIterations = iter + 1
            lastDiff = diff
            if (diff < KULKARNI_QUASI_TOLERANCE) { converged = true; break }
        }
        // Ранее несошедшийся итерант возвращался МОЛЧА: отличить его от верного
        // результата было невозможно. Теперь действует единый контракт [reportConvergence].
        reportConvergence(
            converged = converged,
            throwOnDivergence = throwOnDivergence,
            methodName = "Схема Кулкарни для квазиинтерполянта '${funcs.name}' ($equationName)",
            iterations = performedIterations,
            maxIterations = KULKARNI_QUASI_MAX_ITERATIONS,
            residual = lastDiff,
            tolerance = KULKARNI_QUASI_TOLERANCE,
            hint = kulkarniQuasiHint,
        )
        val finalFun = uFun
        return SolutionFunc(
            eval = { t -> finalFun(t) },
            converged = converged,
            iterations = performedIterations,
            residual = lastDiff,
        )
    }

    /**
     * Итерированный Кулкарни: ^u_h^K = f + L u_h^K.
     *
     * Признак сходимости НАСЛЕДУЕТСЯ от [kulkarni]: сама итерация Слоана — одно
     * интегрирование без итераций, но её результат осмыслен лишь тогда, когда
     * осмыслено исходное приближение.
     */
    fun iteratedKulkarni(): SolutionFunc {
        val kulkarniSolution = kulkarni()
        val kulkarniImage = image(prepare { s -> kulkarniSolution.eval(s) })
        return SolutionFunc(
            eval = { t -> fEff(t) + kulkarniImage(t) },
            converged = kulkarniSolution.converged,
            iterations = kulkarniSolution.iterations,
            residual = kulkarniSolution.residual,
        )
    }
}
