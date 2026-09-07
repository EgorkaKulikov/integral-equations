# AGENTS.md — правила работы с репозиторием `integral-equations`

Документ адресован агентам (и людям), вносящим изменения в этот репозиторий.
Он обязателен к прочтению перед любой правкой файлов.

## 1. Назначение репозитория

Прикладной исследовательский проект: численное решение интегральных уравнений
**Фредгольма, Вольтерры и Урысона** первого и второго рода методом сплайн-коллокации
на квадратичных минимальных сплайнах. Содержит:

- решатели (`src/main/kotlin/solvers/{core,fredholm,volterra,uryson}`) и их общее
  ядро (`solvers.core`: `SecondKindSolverCore`, `SolutionFunc`, `reportConvergence`,
  `RhsWithDerivatives`);
- модельные задачи и аналитические фикстуры (`src/problems`);
- демонстрации таблиц сходимости и бенчмарк (`src/demo`);
- сквозную численную верификацию: характеризационные эталоны, сверку с публикацией,
  аналитические решения, сверку со SciPy, порядки сходимости (`src/test`).

Это **приложение, а не библиотека**: артефакт не публикуется, публичного API в смысле
внешних потребителей у него нет.

## 2. Архитектурные границы

Направление зависимостей строго одностороннее и ациклично:

```
numerical-core  <--  minimal-splines  <--  integral-equations (этот репозиторий)
       ^                                          |
       +------------------------------------------+
```

| Библиотека | Пакеты | Что там |
|---|---|---|
| `numerical-core` | `numerics.*`, `numerics.backend.*` | `GaussLegendre`, `LinearAlgebra`, бэкенды, `NumericsContext`, `ParallelAssembly`, `Conditioning`, `orders`/`reliableOrders` |
| `minimal-splines` | `splines.*`, `splines.functionals.*`, `splines.metrics.*` | `Grid`, `GeneratingSystem`, `MinimalSplineBasis`, семейства функционалов, `SupportPoints`, `errorEh` |

Правила:

1. **Разрешено** импортировать `numerics.*` и `splines.*` — но ТОЛЬКО через
   опубликованные артефакты (`io.github.egorkakulikov:numerical-core`,
   `io.github.egorkakulikov:minimal-splines`), версии которых закреплены в
   `gradle.properties` (`numericalCoreVersion`, `minimalSplinesVersion`).
2. **Запрещено** добавлять сюда код сплайнов, квадратуры, линейной алгебры,
   бэкендов, порогов достоверности и т.п. Такое изменение делается в соответствующей
   библиотеке, затем там `./gradlew publishToMavenLocal`, затем здесь — обновление
   версии в `gradle.properties` (если версия менялась) и прогон гейтов (раздел 6).
3. **Запрещены** зависимости `project(":...")` и `files("../numerical-core/...")`,
   `files("../minimal-splines/...")`, любые `srcDir` на соседние репозитории.
   Задача `verifyArtifactDependencies` (входит в `check`) проваливает сборку, если
   на classpath обнаружен не-jar или файл из `build/` соседнего репозитория.
4. Типы `solvers.*`/`problems.*` не должны появляться в библиотеках — если для
   решателя нужна новая возможность библиотеки, она формулируется в терминах
   библиотеки (сетка, базис, функционал, СЛАУ), а не уравнения.
5. `SplineSpace` (стабилизатор Тихонова, `gramR`, `omegaReg`) — часть решателя
   Урысона и остаётся здесь, хотя строится из объектов `splines.*`.

## 3. Политика видимости

Приложение, не библиотека: модификатор `public` не является обещанием совместимости.
Символы, помеченные `internal` (`solvers.core.SecondKindDefaults`,
`solvers.core.IterationStopCriterion`, `solvers.uryson.NewtonRun`,
`solvers.uryson.runNewtonIterations`, `VolterraOperator.IntegrandCache` и его члены),
остаются `internal`; расширять видимость ради удобства теста нельзя — тесты живут в том
же модуле и видят `internal`.

## 4. Команды

| Команда | Что делает | Время |
|---|---|---|
| `./gradlew fastTest` | быстрый набор (тег `fast`): решатели, проводка контекста, regression, быстрый поднабор порядков | секунды |
| `./gradlew characterizationTest extraCharacterizationTest` | **ГЛАВНЫЙ гейт численной нейтральности**: 1366 + 1344 значения против `baseline-eh.tsv`/`baseline-extra.tsv`, допуск 1e-9 | ~1 мин |
| `./gradlew check` / `./gradlew build` | `fastTest` + оба гейта + `slowTest` + `verifyArtifactDependencies` + `koverVerify` | ~9 мин |
| `./gradlew slowTest` | сверка с публикацией, перекрёстная согласованность, аналитика, полная матрица порядков (сетки до n = 64) | ~8.5 мин |
| `./gradlew convergenceOrderTest` | полная матрица порядков сходимости, 168 сочетаний | 1.5–2 мин |
| `./gradlew scipyVerify` | внешняя сверка со SciPy/NumPy; требует Python и сети (создаёт `.venv-verify`) | ~1 мин + установка |
| `./gradlew captureBaseline` / `captureExtraBaseline` | снять снимки `E_h` в `build/baseline/` — только по протоколу раздела 6 | ~1 мин |
| `./gradlew sec4Tables` | таблицы §5–6 статьи в `build/sec4/` (`-Dsec4.quad=8|16`) | минуты |
| `./gradlew runFredholm` / `runVolterra` / `runUryson` / `runBenchmark` | демонстрации и бенчмарк | — |
| `./gradlew verifyArtifactDependencies` | библиотеки на classpath только как jar | секунды |

Перед первой сборкой библиотеки должны быть в `mavenLocal()`: в каждой из них
`./gradlew publishToMavenLocal`. Альтернатива — `-PnumericsRepositoryUrl=<url>`.

Все тестовые задачи получают `-Dnumerics.backend=multik` по умолчанию; внешнее
значение уважается. Эталоны привязаны к бэкенду multik.

## 5. Численные инварианты, которые нельзя нарушать

- **Порядки сходимости** из таблицы ожиданий `convergence.ConvergenceOrderTest`
  (168 сочетаний «схема × базис × семейство»); деградацию ловит быстрый поднабор в
  `fastTest`.
- **Контракт сходимости** `solvers.core.reportConvergence`: итерационная схема не может
  молча вернуть несошедшийся результат; при `throwOnDivergence = true` —
  `IllegalStateException`, при `false` — `SolutionFunc.converged == false` и `residual`.
- **Побитовая идентичность** результата при `NumericsContext.parallel = true/false`
  (`solvers.core.NumericsContextWiringTest.parallelFlagDoesNotChangeResultBitwise`).
- **Согласование контекста**: решатель и его зависимости (`funcs`, `space`) обязаны
  нести один `NumericsContext`; расхождение — громкий отказ через
  `NumericsContext.requireSame`.
- **Единый допуск включения точки разбиения** `Grid.breakpointInclusionEps` —
  единственный источник для `VolterraOperator` и `SplineSpace`.
- Характеризационные значения `baseline-eh.tsv`, `baseline-extra.tsv` и
  опубликованные значения `published-values.tsv` (допуск 2 %, узкий класс 8 % для
  шести ключей `table-f1.tex`).

## 6. Правила изменения эталонов (baselines)

Полный протокол — [`docs/baseline-changes.md`](docs/baseline-changes.md). Кратко:

1. До изменения алгоритма: `./gradlew captureBaseline captureExtraBaseline`, сохранить
   снимки вне `build/`.
2. После изменения: снять снимки повторно и сравнить ключи попарно; пересечение
   ключей ОБЯЗАНО совпадать побитово, если изменение не претендует на смену чисел.
3. Если числа меняются намеренно (исправление ошибки) — запись в
   `docs/baseline-changes.md` со старыми и новыми значениями и причиной; коммит с
   эталоном отделён от коммита с рефакторингом.
4. **Подгонять допуски (1e-9, 2 %, 8 %) ради зелёной сборки ЗАПРЕЩЕНО.**
5. Машинно-зависимые гейты (`@Tag("machine")`: оба характеризационных класса и
   `PublishedValuesTest.fredholmFirstKindMatchesPublishedValues`) работают только на
   машине снятия эталона (macOS aarch64, multik). Локально включены всегда; в CI —
   `-PmachineDependentGates=false`. Различие архитектуры CPU не является
   доказательством математической регрессии — см. `docs/TESTING.md`.

Процедура при изменении библиотеки: изменил `numerical-core` или `minimal-splines` →
там `./gradlew test publishToMavenLocal` → здесь
`./gradlew characterizationTest extraCharacterizationTest` (затем `check` перед мержем).
Библиотеки собственных численных эталонов не имеют — их защитой служат гейты этого
репозитория.

## 7. Regression-тесты

Каждое исправление дефекта сопровождается тестом в `regression.DefectRegressionTest`
(или в тестах соответствующего решателя), который **падает на коде до исправления**.
Тест называется по дефекту и содержит описание: в чём была ошибка, почему не ловилась
раньше, что проверяется теперь. Нумерация дефектов общая с исходным монорепозиторием
(дефекты 4, 5 — в `numerical-core`).

## 8. Новые численные алгоритмы

- Ссылка на первичную публикацию в [`docs/REFERENCES.md`](docs/REFERENCES.md) со
  статусом («Подтверждено» / «Адаптация» / «Без источника»). Если источника нет —
  это указывается явно; выдумывать ссылки, DOI и библиографические данные запрещено.
- Новая схема второго рода — в `SecondKindSolverCore` или рядом, с покрытием в
  `ConvergenceOrderTest` (ожидаемый порядок с обоснованием) и, при наличии внешнего
  эталона, в `verification.*`.
- Различать: доказанную оценку, численное наблюдение, адаптацию без доказательства.
  Формулировки в KDoc и документации не должны завышать статус.
- Численные эталоны расширяются по протоколу раздела 6, не переснимаются.

## 9. Язык и стиль

Комментарии, KDoc, сообщения об ошибках, документация и сообщения коммитов — на
русском языке (сложившаяся конвенция проекта). Имена идентификаторов — английские.
KDoc объясняет **почему** принято решение и какие альтернативы отвергнуты, а не
пересказывает код.

## 10. Соседние репозитории

| Репозиторий | Роль | Зависит от |
|---|---|---|
| `numerical-core` | универсальные численные примитивы | — |
| `minimal-splines` | минимальные сплайны и функционалы | `numerical-core` |
| `integral-equations` (этот) | решатели, задачи, верификация | обеих библиотек |
| будущий четвёртый проект | другие численные методы на сплайнах | `numerical-core`, `minimal-splines`; **не** от этого репозитория |

Правила библиотек — в их собственных `AGENTS.md`. План и обоснование разделения —
[`docs/REPOSITORY_SPLIT_PLAN.md`](docs/REPOSITORY_SPLIT_PLAN.md).

## 11. Документы

| Файл | Назначение |
|---|---|
| [`README.md`](README.md) | обзор, быстрый старт, API, верификация |
| [`docs/TESTING.md`](docs/TESTING.md) | задачи Gradle, состав `check`, машинно-зависимые гейты, CI, зависимости-артефакты |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | источники схем, регуляризации, верификации |
| [`docs/ACCURACY.md`](docs/ACCURACY.md) | граница по `alpha` для уравнений первого рода |
| [`docs/HPC.md`](docs/HPC.md) | бенчмарк и параллельная сборка в решателях |
| [`docs/baseline-changes.md`](docs/baseline-changes.md) | протокол и история правки эталонов |
| [`TASK.md`](TASK.md), [`tasks/uryson-task.md`](tasks/uryson-task.md) | исторические задания исходного монорепозитория |
