# Источники численных методов

Документ связывает каждый реализованный численный метод с первичной публикацией и
фиксирует статус соответствия: подтверждено сверкой формул, адаптация или
собственная конструкция без опубликованного аналога.

Проверка проводилась сопоставлением формул кода с формулами источников. Для методов,
у которых опубликованного первоисточника нет, это указано явно — такие места **не
следует** сопровождать ссылками на литературу.

## Условные обозначения статуса

| Статус | Значение |
|---|---|
| **Подтверждено** | формула кода дословно совпадает с формулой публикации |
| **Адаптация** | метод перенесён на класс задач, для которого он в источнике не доказан |
| **Без источника** | опубликованного аналога не найдено; конструкция собственная |

---

## 1. Минимальные сплайны и порождающие системы

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Аппроксимационные соотношения, построение `a_j`, `M_k^{-1} phi(t)` | `numerics/MinimalSplineBasis.kt` | [Makarov 2012], [Демьянович 1994] | Подтверждено |
| Полиномиальная система `phi^B = (1, t, t^2)` | `numerics/GeneratingSystem.kt` | [Makarov 2012], [Демьянович 1994] | Подтверждено |
| Гиперболическая система `phi^H = (1, sinh t, cosh t)` | `numerics/GeneratingSystem.kt` | [Kulikov, Makarov 2019a] | Подтверждено |
| Тригонометрическая система `phi^T = (1, sin t, cos t)` | `numerics/GeneratingSystem.kt` | [Kosogorov, Makarov 2017] | Подтверждено |
| Явные формулы `ReferenceSplines.omegaB` | `numerics/MinimalSplineBasis.kt` | [Makarov 2012] | Подтверждено |
| Явные формулы `ReferenceSplines.omegaH` | `numerics/MinimalSplineBasis.kt` | [Kulikov, Makarov 2019a] | Подтверждено (выведено из общей формулы) |
| Узлы кратности 3 на концах отрезка | `numerics/Grid.kt` | [Kulikov, Makarov 2020] | Подтверждено |

## 2. Аппроксимационные функционалы

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Проекционные `theta_j` (5-точечная закрытая форма) | `numerics/functionals/Functionals.kt`, `ProjFunctionals` | [Kulikov, Makarov 2025] | Подтверждено |
| Функционалы де Бура–Фикса `xi^<0>`, `xi^<1>`, `xi^<2>` | `DeBoorFixFunctionals` | [Kulikov, Makarov 2019b] | Подтверждено (послагаемо) |
| Биортогональность `xi_i(omega_j) = delta_ij` | `DeBoorFixFunctionals` | [Kulikov, Makarov 2019b] | Подтверждено |
| Краевые `xi` при `j = -2`, `n-1` (чистые значения) | `DeBoorFixFunctionals.buildXi` | — | **Без источника**: допущение реализации, закрытое тестом биортогональности `biorthogonalityAllRAllBasesIncludingBoundary` |
| Усредняющие `mu_j` | `AveragingFunctionals` | [Kulikov, Makarov 2022] | Подтверждено |
| Трёхточечные `lambda_j` | `ThreePointFunctionals` | [Kulikov, Makarov 2022] | Подтверждено |
| Дискретизованные `xitilde^<1>`, `xitilde^<2>` | `DiscreteDeBoorFixFunctionals` | — | **Без источника**: новизна работы, в литературе ранее не рассматривались |

## 3. Схемы решения уравнений второго рода

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Базовая коллокация `(I - M)c = g` | `SecondKindSolverCore.base` (Фредгольм, Вольтерра) | [Dagnino, Remogna, Sablonnière 2014] | Подтверждено |
| Итерация Слоана | `SecondKindSolverCore.sloan` | [Sloan 1976] | Подтверждено |
| Схема Кулкарни (проекторы) `(I - M - M2 + M^2)c = (I - M)g + d` | `kulkarniProjector` | [Kulkarni 2003] | Подтверждено |
| Восстановление `u^K = y_h + (I - P)[f + L y_h]` | `kulkarniProjector` | [Kulkarni 2003] | Подтверждено |
| Итерированный Кулкарни | `iteratedKulkarni` | [Kulkarni 2003] | Подтверждено |
| Схема Кулкарни для квазиинтерполянтов (`mu`, `lambda`) | `kulkarniQuasi` | — | **Без источника**: теория Кулкарни существенно требует `P^2 = P`; статус — численное наблюдение. Реализована простой итерацией и требует сжатия (спектральный радиус меньше единицы) — выявлено на задачах `A-sep2`, `A-sep3` |
| Классический Nyström (голая квадратура) | `nystrom`, `iteratedNystrom` | [Atkinson 1997] | Подтверждено; **порядок не повышает** |
| Комбинированный Nyström `L_n = P_chi L + (I - P_chi)L^N_h` (Фредгольм) | `combinedNystrom` | [Allouch и др. 2021], [Remogna, Sbibih, Tahrichi 2023] | Подтверждено |
| Комбинированный Nyström (Вольтерра) | `combinedNystrom` | — | **Адаптация**: доказательств для переменного верхнего предела нет |

### Замечание о порядках сходимости

Опубликованные оценки `O(h^7)` для комбинированного оператора и `O(h^8)` для его
итерированного варианта доказаны **только** для уравнения Фредгольма с равномерной
сеткой, полиномиальными B-сплайнами и квазиинтерполяционным проектором.

Измеренные в проекте порядки (`CombinedNystromTest`):

* уравнение Фредгольма, задача F2, базис B, семейство `theta`: классический Nyström — около 4.2,
  комбинированный — около 7.0 (согласуется с теорией);
* уравнение Вольтерры: **выигрыша не наблюдается**, порядок обоих вариантов около 3.8
  (см. `CombinedNystromVolterraTest`).

Для уравнения Вольтерры аналогов теорем о суперсходимости в известной литературе нет:
переменный верхний предел приводит к зависящим от точки коллокации весам и усечению
последней ячейки, что требует отдельного анализа.

## 4. Уравнения первого рода

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Регуляризация Фредгольма `(alpha I + K)u = f` | `solvers/fredholm`, `FredholmFirstKindSolver` | [Wazwaz 2011a] | Подтверждено |
| Значение `alpha = 1e-10` | `FredholmFirstKindSolver` | [Kulikov, Makarov 2023] | Экспериментальный выбор авторов, не рекомендация [Wazwaz 2011a] |
| Сведение Вольтерры I → II рода дифференцированием | `solvers/volterra`, `VolterraFirstKindSolver` | [Wazwaz 2011b], [Brunner 2004] | Подтверждено для случая `m = 1`; требует `K(t,t) != 0` во всех точках деления, проверяется `safeDiagonal` |
| Формулы `(Vu)'` и `(Vu)''` (правило Лейбница) | `VolterraOperator.applyDeriv`, `applyDeriv2` | [Makarov, Kulikov 2026] | `(Vu)'` подтверждено; `(Vu)''` — отдельной публикации нет. Обе формулы численно сверены с `scipy.integrate.quad` проверками слоя L4/L5 `V/V2/rhsDeriv`, `V/V2/rhsDeriv2`, `V/V2exp/rhsDeriv`, `V/V2exp/rhsDeriv2`, `V/V2win/rhsDeriv`, `V/V2win/rhsDeriv2` (эталон выведен независимо от кода: `volterra_image_deriv` в `tools/verify_with_scipy.py`, включая ПОЛНУЮ производную диагонали `d/dt K(t,t) = K_t(t,t) + K_s(t,t)`); фактическое отклонение — машинная точность, до 6.7e-16 при допуске 1e-10 |

## 5. Нелинейное уравнение Урысона

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Оператор Урысона и производная Фреше | `UrysohnOperator` | [Krasnoselskii 1964], [Zeidler 1986] | Подтверждено |
| Ньютон с аналитическим якобианом `B(c)_{j,i} = theta_j(U'(x_h) omega_i)` | `CollocationCore.bMatrix` | [Atkinson 1997] | Подтверждено |
| Схема Кулкарни (квази-Ньютон с предобуславливателем) | `UrysonSecondKindSolver.kulkarni` | [Kulkarni 2003], [Dagnino, Dallefrate, Remogna 2019] | Подтверждено |
| Сплайн-Nyström для Урысона | `UrysonSecondKindSolver.nystrom` | [Remogna, Sbibih, Tahrichi 2023] | Подтверждено |
| Регуляризация Тихонова, стабилизатор `R_h` в норме `W^{1,2}` | `SplineSpace.gramR` | [Тихонов, Арсенин 1977], [Engl, Hanke, Neubauer 1996] | Подтверждено |
| Гаусс–Ньютон для регуляризованной задачи | `UrysonFirstKindSolver.solveFixedAlpha` | [Engl, Hanke, Neubauer 1996] | Подтверждено (знак в знак) |
| Принцип невязки Морозова | `UrysonFirstKindSolver.solveMorozov` | [Engl, Hanke, Neubauer 1996], разд. 4.3 | Подтверждено |
| Гомотопия по убывающему `alpha` с тёплым стартом | `solveMorozov` | — | Деталь реализации |
| Модель шума (кусочно-линейный профиль) | `noisyThetaF` | — | Деталь реализации; детерминированность и масштабирование в `L^2` соответствуют постановке |

---

## 6. Независимая верификация: чем проверен каждый метод

Разделы 1–5 связывают методы с публикациями (соответствие ФОРМУЛ). Настоящий
раздел фиксирует другое: какими **независимыми от кода проекта** средствами
проверен РЕЗУЛЬТАТ вычислений.

Причина выделения: характеризационные тесты (`characterization/baseline-eh.tsv`)
сняты ТОЙ ЖЕ реализацией, которую проверяют: они фиксируют неизменность
результата, но не его правильность. Ошибка, внесённая до создания эталона,
была бы зафиксирована вместе с ним.

### Средства верификации

| Обозначение | Средство | Реализация | Степень независимости |
|---|---|---|---|
| **А** | Аналитически точные решения (сепарабельные ядра, свёртка/Лаплас, MMS) | `problems/analytic/AnalyticProblems.kt`, `verification/AnalyticSolutionTest.kt` | Полная: решение и правая часть выведены вручную, без квадратуры проекта |
| **П** | Опубликованные значения `E_h` и `p_h` (18 таблиц статьи) | `verification/published-values.tsv`, `PublishedValuesTest.kt` | Полная: внешний источник, допуск 2 % |
| **К** | Перекрёстная согласованность схем между собой | `verification/CrossSchemeConsistencyTest.kt` | Частичная: общая ошибка всех схем не обнаруживается |
| **И** | Математические инварианты (биортогональность, разбиение единицы, идемпотентность, точность на span) | `healthchecks/SplineCoreHealthCheckTest.kt` | Высокая: свойства выведены из теории, а не из кода |
| **Ф** | Сверка с явными замкнутыми формулами (`ReferenceSplines`, `closedFormInternal`) | `SplineCoreHealthCheckTest.kt` | Высокая: формулы выписаны независимо от общего построения |
| **С** | Внешняя сверка через SciPy/NumPy | `ScipyCrossVerificationTest` + `tools/verify_with_scipy.py`; задача `scipyVerify` и job `scipy` в CI. В `check` НЕ входит (требует venv с Python); в `test` без venv пропускается | Полная: сторонние библиотеки |
| **По** | Порядок сходимости против явной таблицы ожиданий | `convergence/ConvergenceOrderTest.kt` (задача `convergenceOrderTest`, 168 сочетаний; быстрый поднабор — тег `fast`) | Внутренняя: ожидаемые порядки сверены с теорией и публикацией, но сами числа считает код проекта |
| **Н** | Эталонный Nyström на квадратуре Гаусса–Лежандра (без сплайнов и функционалов) | `verification/ReferenceNystromSolver.kt` + `ReferenceNystromCrossCheckTest` (тег `fast`); внешний аналог — слои L6a/L6b `tools/verify_with_scipy.py` | Полная: учебный метод [Atkinson 1997]; собственные узлы Гаусса и гауссово исключение, ни одного импорта из `src/main` |

#### Границы внешней сверки (средство **С**): что НЕ сверяется и почему

Слой L2 выгружает пять блоков (`M`, `M2`, `g`, `d`, `c_base`), а сверяет три: `M`, `g`, `c_base`
(через `scipy.linalg.solve` на той же системе `(I - M) c = g`).

**Блоки `M2` и `d` внешне НЕ СВЕРЯЮТСЯ.** Это зафиксировано явно, а не умолчано:
по коду (`solvers/core/SecondKindSolverCore.kt`) `M2_{j,i} = chi_j(L(L omega_i))` — то есть
функционал от ДВОЙНОГО применения интегрального оператора к базисной функции,
а `d_j = chi_j(L f)`. Оба блока выражены через аппроксимационные функционалы `chi`
(семейства `theta`, `xi`, `mu`, `lambda`) и базис минимальных сплайнов. Аналогов этих
конструкций в SciPy нет; выписать их в скрипте значило бы перенести туда ту же
конструкцию, что проверяется, и получить сверку, замкнутую саму на себя.
Имитация внешней сверки хуже честного признания её отсутствия: зелёная проверка без
независимого источника создаёт ложную уверенность.

Чем они проверены вместо этого — и чего это НЕ даёт.

* Средство **И**: биортогональность `chi_j(omega_i) = delta_{ij}`, идемпотентность
  проектора, точность на `span`. Это проверяет ФУНКЦИОНАЛЫ `chi`, из которых собраны
  `M2` и `d`, но не саму сборку блоков.
* Внутренняя проверка ПО СЛЕДСТВИЯМ: `M2` и `d` входят ровно в две схемы —
  `kulkarniProjector` и `iteratedKulkarni` (`solvers/core/SecondKindSolverCore.kt`,
  `kulkarni()` и `iteratedKulkarni()`). Порядок сходимости обеих проверяется
  `convergence.ConvergenceOrderTest` (полная матрица — задача `./gradlew convergenceOrderTest`,
  быстрый поднабор — тег `fast`) против явной таблицы ожидаемых порядков: ошибка в `M2`
  или `d` сместила бы наблюдаемый порядок `kulkarni`/`iteratedKulkarni` и уронила бы этот
  тест. Дополнительно числа этих схем зафиксированы характеризационными эталонами
  (`baseline-eh.tsv`, `baseline-extra.tsv`) — то есть от изменения они защищены.

**Чего у `M2` и `d` НЕТ: внешней сверки — ни прямой, ни косвенной.** Слой L6b сверяет
только схемы `base` и `sloan`, а `sloan()` использует лишь `matrixM` и `vectorG`
(`SecondKindSolverCore.kt`, `sloan()`): ни `M2`, ни `d` в неё не входят, поэтому L6b
на ошибку в них НЕ отреагировал бы. Раньше здесь утверждалось обратное — утверждение
было ложным и удалено. Итог: `M2` и `d` покрыты внутренней проверкой порядка сходимости
и характеризационными эталонами, но независимого внешнего источника у них нет.

### Соответствие «метод → средства верификации»

| Метод | Средства | Статус |
|---|---|---|
| Базис минимальных сплайнов (`B`) | Ф, И, С | Проверен |
| Базис (`H`) | Ф, И | Проверен |
| Базис (`T`) | И | Только инварианты: явной эталонной формулы в `ReferenceSplines` нет |
| Квадратура Гаусса–Лежандра | И, С | Проверена (точность на многочленах до степени 15) |
| Функционалы `theta` | Ф, И | Проверены (в том числе опубликованные коэффициенты `{1/14, -2/7, 10/7, -2/7, 1/14}`) |
| Функционалы `xi<0>`, `xi<1>`, `xi<2>` | И | Проверены биортогональностью и идемпотентностью |
| Функционалы `mu`, `lambda` | И | Проверены точностью на `span{1, rho, sigma}` |
| Базовая коллокация (Фредгольм) | А, П, К | Проверена тремя независимыми средствами |
| Базовая коллокация (Вольтерра) | А, П, К | Проверена тремя независимыми средствами |
| Итерация Слоана | А, П, К | Проверена |
| Схема Кулкарни (проекторы) | А, П, К | Проверена |
| Схема Кулкарни (квазиинтерполянты) | А, К | Ограничение: расходится при спектральном радиусе больше единицы |
| Классический Nyström | П, К, И | Проверен |
| Комбинированный Nyström (Фредгольм) | К | Порядок 7.0 согласуется с опубликованной оценкой `O(h^7)` |
| Комбинированный Nyström (Вольтерра) | К | Адаптация: суперсходимость не наблюдается (порядок около 3.8) |
| Вольтерра I рода (редукция) | П | Проверена по `table-v1.tex` |
| Фредгольм I рода (регуляризация) | П | **РАСХОЖДЕНИЕ, не закрыто.** См. примечание ниже |
| Урысон (нелинейные схемы) | К | Слабее остальных: аналитические решения не выведены |

#### Открытое расхождение: Фредгольм первого рода против `table-f1.tex`

Строка выше НЕ помечена как «проверено» намеренно: тест
`verification.PublishedValuesTest.fredholmFirstKindMatchesPublishedValues` СЕЙЧАС
КРАСНЫЙ. Фактический прогон `./gradlew slowTest`: 4 ключа `F.F1.H.theta.*`
расходятся с публикацией на 6.78 %, 4.23 %, 3.47 % и 3.14 % при допуске 2 %.

Что установлено измерениями (а не предположено):

* F1 — уравнение ПЕРВОГО рода с регуляризацией Вазваза (`c_L = -1/alpha`,
  `alpha = 1e-10`); измеренное `cond_2(I - M)` порядка `1e10`;
* бэкенд `reference` воспроизводит опубликованные числа до 4-й значащей цифры,
  а `multik` — нет; эталоны проекта сняты на разных бэкендах.

Причина похожа на разную чувствительность двух реализаций решения ПЛОХО
ОБУСЛОВЛЕННОЙ СЛАУ, но окончательный разбор НЕ ЗАВЕРШЁН и отложен на этап 8
(`.tasks/code-review-remediation/SPEC.md`, п. 8.6). Допуск НЕ ослаблялся и тест не
отключался: красный тест — честный сигнал о незакрытом вопросе.

### Результат внешней сверки со SciPy (фактический прогон)

Сверка выполнена на NumPy 2.5.1 / SciPy 1.18.0 и пройдена полностью: 60 проверок
(12 547 сравнённых чисел, 83 пропущены как неприменимые), расхождений нет.
Числа в таблице ниже взяты из `build/verification/scipy-report.json` этого прогона.

**Статус: постоянная проверка, а не разовая.** Сверка оформлена смок-тестами
`verification.ScipyCrossVerificationTest` и выполняется задачей `./gradlew scipyVerify`
— в CI на каждой ветке. Окружение Python готовится автоматически задачей
`setupScipyVerification` (создаёт `.venv-verify` и ставит версии, закреплённые в
`tools/requirements-verify.txt`). Причина выбора: разовая сверка защищает лишь в
момент запуска, и любая последующая правка численного ядра могла бы разойтись
со SciPy незаметно.

Поведение без окружения Python зависит от задачи, и это сделано намеренно:

* `./gradlew scipyVerify` — ПАДАЕТ (задача выставляет `scipy.required=true`). Здесь
  сверка запрошена явно и окружение готовится зависимыми задачами, поэтому тихий
  пропуск означал бы зелёную сборку без единого внешнего доказательства;
* `./gradlew test` (и любой другой прогон) — ПРОПУСКАЕТ: отсутствие venv это
  состояние машины, а не расхождение чисел.

В обоих режимах РАСХОЖДЕНИЯ ЧИСЕЛ всегда дают падение — смягчается только
реакция на неготовое окружение.

Действенность проверена внесением искусственного дефекта: возмущение весов
квадратуры на `1e-9` уронило ровно слои L1 и L4/L5 (с указанием отклонения и
допуска), оставив L2, L3, L6a/L6b зелёными; после отката все тесты вновь проходят.

| Слой | Проверяемое | Эталон | Проверок | Наибольшее отклонение | Допуск |
|---|---|---|---|---|---|
| L1 | Узлы и веса Гаусса–Лежандра, `m = 1..16` | `numpy leggauss` | 2 | `1.1e-16` (узлы) / `1.8e-15` (веса), абс. | `1e-14` |
| L2 | Решение `(I - M) c = g` и невязка | `scipy.linalg.solve` | 2 | `2.2e-16`, абс. | `1e-10` |
| L3 | Базис `B` и две производные, 4 типа сеток | `scipy.interpolate.BSpline` | 12 | `1.1e-13` абс. (`omega'`) / `1.5e-14` отн. (`omega''`) | `1e-12` / `1e-13` |
| L4/L5 | Образы `K u`, `V u` и правые части (5 задач) | `scipy.integrate.quad` | 20 | `8.9e-16`, абс. | `1e-10` |
| L6a | Эталонный Nyström против точного решения (F2, F2exp) | 80 узлов `leggauss` | 2 | `1.3e-15`, абс. | `1e-12` |
| L6b (а) | `E_h` схем `base`/`sloan` против порогов эталона | эталонный Nyström | 12 | `0.50`, отн. | `1.0` |
| L6b (б) | порядок сходимости тех же схем не ниже порога | эталонный Nyström | 10 | `0.00` (дефицит p) | `0.0` |

Пояснение к L6b: его 22 проверки — ДВУХ РАЗНЫХ ВИДОВ, и единого «допуска 1.0» у слоя нет.

* Вид (а), 12 проверок: сама величина `E_h`, допуск 1.0 ОТНОСИТЕЛЬНЫЙ — грубый по самой
  постановке: сравниваются НЕ одни и те же числа, а порядки величин двух разных
  методов. Фактическое отклонение во всех 12 — около 0.50, то есть запас двукратный.
* Вид (б), 10 проверок: НАБЛЮДАЕМЫЙ ПОРЯДОК против порога эталона. Здесь допуск 0.0 —
  не опечатка и не строгость до последнего бита: сравнивается НЕ отклонение, а ДЕФИЦИТ
  порядка `max(0, порог − наблюдаемый p)`, и «допуск 0» означает требование «порядок не
  НИЖЕ порога» (превышение разрешено). Сами пороги заданы с запасом от теории.

Наблюдаемые в этом прогоне порядки: `base` — 3.02...3.05, `sloan` — 4.19...4.28.
Отметим, что это внешняя сверка порядка только двух схем; систематическая проверка
порядков всех схем — средство **По** (`ConvergenceOrderTest`), где 24 из 56 строк таблицы
ожиданий сверены с `published-values.tsv` (худшее расхождение 0.44).

**Обнаруженное расхождение и его разбор.** Первый прогон дал два расхождения на
второй производной базиса (сетки `quasiUniform` и `graded`): абсолютное отклонение
`1.79e-12` при допуске `1e-12`. Разбор показал, что дефекта нет, а неверен был
сам критерий сравнения: `omega''` растёт как `O(1/h^2)` и на этих сетках достигает
`|omega''| ~ 200...400`, так что наблюдавшееся отклонение отвечает ОТНОСИТЕЛЬНОЙ
величине `1.2e-14` — порядка 50 ulp. Это шум округления двух разных способов
вычисления (проект обращает матрицу `M_k` размера 3x3, SciPy использует рекуррентные
соотношения де Бура), а не расхождение методов. Абсолютный порог заменён
относительным (`1e-13`); допуск НЕ ослаблялся — исправлена размерность сравнения.

**Чего средства верификации принципиально НЕ покрывают.**
тригонометрические минимальные сплайны и все четыре семейства аппроксимационных
функционалов — конструкции из работ авторов, аналогов в сторонних библиотеках у них
нет. Для них внешняя сверка (С) неприменима принципиально, а не из-за неполноты
работы; их проверка опирается на математические инварианты (И), которые выведены из
теории независимо от реализации.

---

## Список литературы

1. **[Демьянович 1994]** Демьянович Ю. К. *Локальная аппроксимация на многообразии и
   минимальные сплайны.* — СПб.: Изд-во С.-Петерб. ун-та, 1994. — 356 с.

2. **[Тихонов, Арсенин 1977]** Тихонов А. Н., Арсенин В. Я. *Методы решения
   некорректных задач.* — М.: Наука, 1977.

3. **[Makarov 2012]** Makarov A. A. Construction of Splines of Maximal Smoothness //
   Journal of Mathematical Sciences. — 2012. — Vol. 178, No. 6. — P. 589–604.

4. **[Kosogorov, Makarov 2017]** Kosogorov O., Makarov A. On Some Piecewise Quadratic
   Spline Functions // Numerical Analysis and Its Applications. Lecture Notes in
   Computer Science, Vol. 10187. — 2017. — P. 448–455.

5. **[Kulikov, Makarov 2019a]** Kulikov E. K., Makarov A. A. On Approximation by
   Hyperbolic Splines // Journal of Mathematical Sciences. — 2019. — Vol. 240, No. 6. —
   P. 822–832.

6. **[Kulikov, Makarov 2019b]** Kulikov E. K., Makarov A. A. On de Boor–Fix Type
   Functionals for Minimal Splines // Topics in Classical and Modern Analysis (Applied
   and Numerical Harmonic Analysis). — Springer, 2019. — P. 211–225.

7. **[Kulikov, Makarov 2020]** Kulikov E. K., Makarov A. A. Quadratic minimal splines
   with multiple nodes // Journal of Mathematical Sciences. — 2020. — Vol. 249, No. 2. —
   P. 256–262.

8. **[Kulikov, Makarov 2022]** Kulikov E. K., Makarov A. A. Construction of
   Approximation Functionals for Minimal Splines // Journal of Mathematical Sciences. —
   2022. — Vol. 262, No. 1. — P. 84–98.

9. **[Kulikov, Makarov 2023]** Kulikov E. K., Makarov A. A. A Method for Solving the
   Fredholm Integral Equation of the First Kind // Journal of Mathematical Sciences. —
   2023. — Vol. 272, No. 4. — P. 558–565. DOI: 10.1007/s10958-023-06449-3.

10. **[Kulikov, Makarov 2025]** Kulikov E. K., Makarov A. A. On Projection-Type
    Approximation Functionals for Minimal Splines // Записки научных семинаров ПОМИ. —
    2025. — Т. 542. — С. 126–143.

11. **[Makarov, Kulikov 2026]** Makarov A., Kulikov E. Spline collocation for Volterra
    integral equations with improved accuracy // Numerical Algorithms. — 2026.
    DOI: 10.1007/s11075-026-02388-7.

12. **[Sloan 1976]** Sloan I. H. Improvement by iteration for compact operator
    equations // Mathematics of Computation. — 1976. — Vol. 30, No. 136. — P. 758–764.

13. **[Kulkarni 2003]** Kulkarni R. P. A superconvergence result for solutions of
    compact operator equations // Bulletin of the Australian Mathematical Society. —
    2003. — Vol. 68, No. 3. — P. 517–528. DOI: 10.1017/S0004972700037916.

14. **[Atkinson 1997]** Atkinson K. E. *The Numerical Solution of Integral Equations of
    the Second Kind.* — Cambridge University Press, 1997.

15. **[Brunner 2004]** Brunner H. *Collocation Methods for Volterra Integral and Related
    Functional Differential Equations.* Cambridge Monographs on Applied and
    Computational Mathematics, Vol. 15. — Cambridge University Press, 2004.

16. **[Wazwaz 2011a]** Wazwaz A. The regularization method for Fredholm integral
    equations of the first kind // Computers & Mathematics with Applications. — 2011. —
    Vol. 61, No. 10. — P. 2981–2986.

17. **[Wazwaz 2011b]** Wazwaz A.-M. *Linear and Nonlinear Integral Equations: Methods
    and Applications.* — Springer, Berlin, Heidelberg, 2011. — XVIII+639 p.
    DOI: 10.1007/978-3-642-21449-3.

18. **[Dagnino, Remogna, Sablonnière 2014]** Dagnino C., Remogna S., Sablonnière P. On
    the solution of Fredholm integral equations based on spline quasi-interpolating
    projectors // BIT Numerical Mathematics. — 2014. — Vol. 54, No. 4. — P. 979–1008.

19. **[Dagnino, Dallefrate, Remogna 2019]** Dagnino C., Dallefrate A., Remogna S. Spline
    quasi-interpolating projectors for the solution of nonlinear integral equations //
    Journal of Computational and Applied Mathematics. — 2019. — Vol. 354. — P. 360–372.
    DOI: 10.1016/j.cam.2018.06.054.

20. **[Allouch и др. 2021]** Allouch C., Remogna S., Sbibih D., Tahrichi M.
    Superconvergent methods based on quasi-interpolating operators for Fredholm integral
    equations of the second kind // Applied Mathematics and Computation. — 2021. —
    Vol. 404. — Art. 126227. DOI: 10.1016/j.amc.2021.126227.

21. **[Remogna, Sbibih, Tahrichi 2023]** Remogna S., Sbibih D., Tahrichi M.
    Superconvergent Nyström Method Based on Spline Quasi-Interpolants for Nonlinear
    Urysohn Integral Equations // Mathematics. — 2023. — Vol. 11, No. 14. — Art. 3236.
    DOI: 10.3390/math11143236.

22. **[Engl, Hanke, Neubauer 1996]** Engl H. W., Hanke M., Neubauer A. *Regularization
    of Inverse Problems.* — Kluwer Academic Publishers, Dordrecht, 1996.

23. **[Krasnoselskii 1964]** Krasnoselskii M. A. *Topological Methods in the Theory of
    Nonlinear Integral Equations.* — Pergamon Press, Oxford, 1964.

24. **[Zeidler 1986]** Zeidler E. *Nonlinear Functional Analysis and its Applications I:
    Fixed-Point Theorems.* — Springer, New York, 1986.
