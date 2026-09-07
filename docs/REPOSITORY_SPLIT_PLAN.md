# План разделения репозитория Numerical-Algorithms на три репозитория

Документ фиксирует архитектурные решения раздела ДО перемещения файлов. Это
структурный рефакторинг: математические алгоритмы не меняются, численные
эталоны (`baseline-eh.tsv`, `baseline-extra.tsv`, `published-values.tsv`)
служат защитой миграции и не переснимаются.

## 1. Имена репозиториев и координаты

| Репозиторий | Артефакт | Назначение |
|---|---|---|
| `numerical-core` | `io.github.egorkakulikov:numerical-core` | универсальные средства numerical computing: квадратура, плотная линейная алгебра, SPI бэкендов, обусловленность, параллельная сборка, контекст вычислений, порог достоверности |
| `minimal-splines` | `io.github.egorkakulikov:minimal-splines` | квадратичные минимальные сплайны: сетка с кратными узлами, порождающие системы, базис, аппроксимационные функционалы, квазиинтерполяция, метрики ошибки |
| `integral-equations` | приложение (не публикуется) | решатели Фредгольма, Вольтерры, Урысона; модельные задачи; демонстрации; характеризационные и верификационные гейты |

Версия: `0.1.0` (semver; библиотеки ещё не имеют внешних пользователей, поэтому
совместимость не гарантируется до `1.0.0`). Переопределение: `-Pversion=…`.

## 2. DAG зависимостей

```
numerical-core  <--  minimal-splines  <--  integral-equations
       ^                                          |
       +------------------------------------------+
```

`integral-equations` подключает ОБЕ библиотеки как Maven-артефакты
(`mavenLocal()` / настраиваемый репозиторий). Обратных и циклических
зависимостей нет. Будущий четвёртый проект подключает `numerical-core` и
`minimal-splines` без `integral-equations` (см. `examples/standalone-consumer`
в `minimal-splines`).

## 3. Владение символами

Критерий: сущность попадает в `numerical-core`, только если имеет смысл без
знания о минимальных сплайнах и интегральных уравнениях. Решение принято по
фактическим usages, а не по имени пакета.

| Текущий символ / путь | Целевой repo | Новый пакет | Причина |
|---|---|---|---|
| `numerics.GaussLegendre` (`Quadrature.kt`) | numerical-core | `numerics` | принимает `DoubleArray` точек разбиения; о сетке сплайнов не знает |
| `numerics.LinearAlgebra`, `ReferenceLinearAlgebra`, `DenseOps` (internal) | numerical-core | `numerics` | плотная LU/матричные операции без предметной семантики |
| `numerics.Conditioning`, `ConditionEstimate`, `ForwardError` | numerical-core | `numerics` | оценка обусловленности произвольной СЛАУ |
| `numerics.backend.*` (`LinAlgBackend`, `Backends`, `MultikCpuBackend`, `ReferenceBackend`) | numerical-core | `numerics.backend` | SPI бэкендов линейной алгебры |
| `numerics.NumericsContext` | numerical-core | `numerics` | бэкенд + флаг параллелизма; используется и сплайнами, и решателями |
| `numerics.ParallelAssembly` | numerical-core | `numerics` | параллельная сборка произвольной матрицы |
| `numerics.functionals.Reliability` (`Measured`, `reliableOrders`, …) | numerical-core | `numerics` | порог достоверности измеренных величин; от сплайнов не зависит |
| `numerics.Grid` | minimal-splines | `splines` | узловой вектор квадратичного минимального сплайна: тройные узлы на концах, `idx(j)=j+2`, `isCoincident`, носители `[x_j,x_{j+3}]`; в core потребителей нет |
| `numerics.GeneratingSystem`, `cross3/dot3/det3/invert3` | minimal-splines | `splines` | порождающая вектор-функция `phi=(1,rho,sigma)` |
| `numerics.Degeneracy` (`DEGENERACY_RELATIVE_EPS`, `isSignificant`, `det3Scale`, …) | minimal-splines | `splines` (internal) | используется только `GeneratingSystem`/`MinimalSplineBasis`/`Functionals` |
| `numerics.MinimalSplineBasis`, `ReferenceSplines`, `nonDegenerate` | minimal-splines | `splines` | сам базис |
| `numerics.functionals.Functionals.kt` (`ApproxFunctional`, `FunctionalFamily`, `ProjFunctionals`, `DeBoorFixFunctionals`, `DiscreteDeBoorFixFunctionals`, `AveragingFunctionals`, `ThreePointFunctionals`, `ValueFunctional`, `DerivFunctional`, `SecondDerivFunctional`) | minimal-splines | `splines.functionals` | аппроксимационные функционалы и квазиинтерполяция |
| `numerics.functionals.SupportPoints` | minimal-splines | `splines.functionals` | объединение опорных точек семейства `ValueFunctional`; зависит от функционалов, не от решателей |
| `numerics.functionals.Metrics` (`errorEh`, `orders`, `constCh`) | minimal-splines | `splines.metrics` | `errorEh` берёт контрольную сетку из `Grid` |
| `numerics.SolutionFunc`, `reportConvergence` | integral-equations | `solvers.core` | «результат решения интегрального уравнения»; потребители — только решатели |
| `solvers.core.*`, `solvers.fredholm.*`, `solvers.volterra.*`, `solvers.uryson.*` (включая `SplineSpace`) | integral-equations | без изменений | решатели |
| `src/problems` (`problems.*`) | integral-equations | без изменений, source set `problems` | модельные задачи и аналитические фикстуры |
| `src/demo` (`demo.*`) | integral-equations | без изменений, source set `demo` | таблицы сходимости, бенчмарк |

`SplineSpace` (Урысон) остаётся в `integral-equations`: он строит матрицу Грама
стабилизатора Тихонова и веса `omegaReg` для регуляризации — это часть решателя,
а не библиотеки сплайнов.

## 4. Тесты

| Тест (текущий путь `src/test/kotlin/…`) | Repo | Примечание |
|---|---|---|
| `numerics/{Conditioning,ForwardError,LinearAlgebra,LinearAlgebraExtra,LinearAlgebraVsReference,ParallelAssembly,ParallelAssemblyToggle,Quadrature}Test`, `numerics/backend/BackendSpiTest` | core | как есть |
| `numerics/functionals/ReliabilityTest` | core | пакет `numerics` |
| `regression/DefectRegressionTest`: `defect4_singularityDetectedRegardlessOfScale`, `defect5_backendsAgreeOnNonFiniteInput` | core | выделяются в `numerics/LinearAlgebraRegressionTest` (только `ReferenceLinearAlgebra`) |
| `numerics/DefensiveCopyTest`: `refNodesWeightsReturnsCopy`, `mutatingRefNodesDoesNotAffectIntegral` | core | выделяются в `numerics/QuadratureDefensiveCopyTest` |
| `numerics/NumericsContextWiringTest`: `defaultContextIsSharedInstance`, `contextEqualityIsByValue` | core | выделяются в `numerics/NumericsContextTest` |
| `numerics/{Grid,GridExtra,GridGeometric,GridGraded,GridCoincidence,GeneratingSystem,DegeneracyScale,MinimalSplineBasis,MinimalSplineBasisExtra}Test` | splines | пакет `splines` |
| `numerics/functionals/{DeBoorFixThree,EdgeBranch,Flags,Functionals,FunctionalsExtra,SupportPoints}Test` | splines | пакет `splines.functionals` |
| `numerics/functionals/{Metrics,MetricsControlGrid}Test` | splines | пакет `splines.metrics` |
| `healthchecks/SplineCoreHealthCheckTest` | splines | биортогональность, разбиение единицы, точность на span |
| всё остальное: `characterization/*`, `convergence/*`, `healthchecks/{Fredholm,Volterra,Uryson}*`, `regression/DefectRegressionTest` (остаток), `solvers/**`, `verification/*`, остаток `DefensiveCopyTest` и `NumericsContextWiringTest` | integral-equations | сквозные и solver-специфичные |

Кросс-репозиторные проверки:
- `integral-equations`: существующие `EhCharacterizationTest`/`ExtraCharacterizationTest` становятся де-факто интеграционными гейтами всех трёх слоёв; `NumericsContextWiringTest` проверяет согласование контекста между библиотеками и решателем.
- `minimal-splines`: `examples/standalone-consumer` — отдельная Gradle-сборка, подключающая опубликованные артефакты и реализующая метод, не связанный с интегральными уравнениями (квазиинтерполяция с оценкой порядка).
- Сверка со SciPy остаётся в `integral-equations` целиком: её выгрузка (`VerificationArtifacts`) строится вокруг `FredholmOperator`, а проверка слоёв (квадратура, базис, СЛАУ) идёт по одному набору артефактов. Разрезать её означало бы дублировать инструментарий выгрузки без выигрыша в доказательности.

## 5. Ресурсы

| Ресурс | Repo |
|---|---|
| `src/test/resources/characterization/baseline-eh.tsv`, `baseline-extra.tsv` | integral-equations (снимки решателей) |
| `src/test/resources/verification/published-values.tsv` | integral-equations |
| `tools/verify_with_scipy.py`, `tools/requirements-verify.txt` | integral-equations |

## 6. Документация

| Документ | Repo | Что меняется |
|---|---|---|
| `README.md` | integral-equations (перерабатывается); новые README в core и splines | у каждого — назначение, что НЕ входит, подключение, пример, тесты, гарантии, источники |
| `docs/REFERENCES.md` §1–2 + источники 1,3–8,10 | minimal-splines | таблицы «элемент → реализация → источник» с новыми путями |
| `docs/REFERENCES.md` §3–6 + источники 2,9,11–24 | integral-equations | ссылка на REFERENCES сплайнов для §1–2 |
| `docs/REFERENCES.md` (новый, краткий) | numerical-core | Гаусса–Лежандра, LU, обусловленность — общие источники; cross-links |
| `docs/ACCURACY.md` §1,2,4,5 (обратная/прямая ошибка, `solveDiagnosed`, `cond`) | numerical-core | без примеров F1 |
| `docs/ACCURACY.md` §3 (граница `alpha` для F1) | integral-equations | ссылается на ACCURACY core |
| `docs/HPC.md` | numerical-core | бэкенды и параллельная сборка; раздел о бенчмарке → integral-equations |
| `docs/TESTING.md`, `docs/baseline-changes.md`, `TASK.md`, `tasks/uryson-task.md` | integral-equations | TESTING сокращается до задач этого repo; в core/splines — свои разделы в README |
| `AGENTS.md`, `CLAUDE.md` | все три, разные | границы импорта, команды тестов, правила baseline |

## 7. Публичный API библиотек

`numerical-core` (public): `GaussLegendre`, `LinearAlgebra`, `ReferenceLinearAlgebra`,
`Conditioning`/`ConditionEstimate`/`ForwardError`, `NumericsContext`,
`ParallelAssembly`, `backend.{LinAlgBackend, Backends, MultikCpuBackend, ReferenceBackend}`,
`Measured`/`measured`/`ratio`/`orderOrNull`/`reliableOrders`/`reliableConstCh`.
Internal: `DenseOps`, `Backends.select`.

`minimal-splines` (public): `Grid`, `GeneratingSystem`, `MinimalSplineBasis`,
`ReferenceSplines`, `nonDegenerate`, `cross3/dot3/det3/invert3`, все семейства
`functionals.*`, `SupportPoints`, `metrics.{errorEh, orders, constCh, DEFAULT_CONTROL_REFINEMENT}`.
Internal: `Degeneracy.kt`, `MinimalSplineBasis.computeA`.

Ни один тип `solvers.*`/`problems.*` не встречается в сигнатурах двух библиотек.

## 8. Устраняемые связи

1. `SolutionFunc` в `numerics` — перенос в `solvers.core`.
2. `internal` символы `Degeneracy.kt`, используемые из `functionals` — оба в одном модуле `minimal-splines`, видимость сохраняется.
3. `MinimalSplineBasis.computeA` (`internal`) вызывается из `Functionals.kt` и теста `GridCoincidenceTest` — тот же модуль.
4. `Backends.select` (`internal`) — тест `BackendSpiTest` в том же модуле core.
5. Смешанные тесты (`DefectRegressionTest`, `DefensiveCopyTest`, `NumericsContextWiringTest`) — разрезаются по слоям.

## 9. Что намеренно НЕ делается

- Не меняются формулы, допуски, эталоны, порядок суммирования.
- Не разрезается сверка со SciPy.
- Не вводится compatibility-слой пакета `numerics.functionals` — внешних пользователей нет.
- Не исправляются известные математические оговорки (например, краевые `xi` без источника) — они остаются в REFERENCES как есть.
- Машинно-зависимые гейты (`@Tag("machine")`) сохраняют существующую логику `-PmachineDependentGates`.

## 10. Порядок коммитов в каждом новом repo

1. извлечение истории (`git filter-repo`, только пути данного repo);
2. архитектурное разделение (сборка, удаление чужого кода, перенос пакетов);
3. очистка пакетов/API;
4. миграция тестов;
5. документация и CI.

## 11. Фактический результат

Разделение выполнено по плану; отличия от разделов 3–4 зафиксированы ниже.

| Что | План | Факт | Причина |
|---|---|---|---|
| `orders`, `constCh` (из `numerics.functionals.Metrics`) | `splines.metrics` | `numerical-core`, файл `numerics.ConvergenceRates` | `Reliability.reliableConstCh` (core) зависит от `constCh`; обе функции работают с готовыми числами погрешностей и о сетке не знают. В `splines.metrics` остались только `errorEh` и `DEFAULT_CONTROL_REFINEMENT` |
| `healthchecks.SplineCoreHealthCheckTest` | minimal-splines, пакет не оговорён | пакет `splines` (`splines.SplineCoreHealthCheckTest`) | в библиотеке нет отдельного слоя `healthchecks`; тест инвариантов живёт рядом с базисом |
| `numerics.NumericsContextWiringTest` (остаток) | integral-equations | `solvers.core.NumericsContextWiringTest`; части `defaultContextIsSharedInstance`/`contextEqualityIsByValue` — в core (`numerics.NumericsContextTest`) | тест проверяет проводку контекста через решатели, а не пакет `numerics` |
| `numerics.DefensiveCopyTest` (остаток) | integral-equations | `solvers.uryson.SplineSpaceDefensiveCopyTest`; квадратурные тесты — в core (`numerics.QuadratureDefensiveCopyTest`) | оставшиеся проверки касаются только `SplineSpace` решателя Урысона |
| диагностика активного бэкенда в CI | не планировалась | добавлен `solvers.core.ActiveBackendDiagnosticTest` | `numerics.backend.BackendSpiTest`, на который опирался шаг CI «Диагностика активного бэкенда», ушёл в `numerical-core`; нужен тест в этом репозитории, утверждающий, что активен `MultikCpuBackend` |
| `regression.DefectRegressionTest` | integral-equations | без дефектов 4 и 5 — они в core (`numerics.LinearAlgebraRegressionTest`) | оба теста используют только `ReferenceLinearAlgebra` |
| `MetricsTest` | minimal-splines | тесты `orders`/`constCh` — в core (`numerics.ConvergenceRatesTest`), тесты `errorEh` — в `splines.metrics.MetricsTest` | вслед за переносом функций |

Численная нейтральность подтверждена: `characterizationTest` 5/5 и
`extraCharacterizationTest` 1/1 зелёные против неизменённых эталонов; снимки
`captureBaseline`/`captureExtraBaseline` до и после разделения совпадают побитово.

## 12. Синхронизация с исходным репозиторием после split

Исходный `Numerical-Algorithms` продолжает жить; изменения переносятся сюда
`git cherry-pick -x` из remote `mono` (`https://github.com/EgorkaKulikov/Numerical-Algorithms.git`)
с последующей перезаписью импортов `numerics.{Grid,…}` → `splines.*`,
`numerics.functionals.*` → `splines.functionals.*`/`splines.metrics.*`,
`numerics.SolutionFunc` → `solvers.core.SolutionFunc` (скрипт-помощник: см. историю коммита
синхронизации). Изменения, затрагивающие `src/main/kotlin/numerics/**` исходного репозитория,
переносятся в `numerical-core`/`minimal-splines` с публикацией новой версии артефакта и
обновлением `gradle.properties` здесь.

| Дата | Диапазон в mono | Перенесено | Затронутые репо |
|---|---|---|---|
| 2026-09-07 | `9a71ec9..edc9eeb` (4 коммита ветки `feat/uryson-unify-new01`) | комбинированный/итерированный Nyström и итерированный Кулкарни для Урысона, `NoiseNorm { L2, SUP }`, `CombinedNystromTest`, демо-таблицы в формате new-01 | только `integral-equations`; библиотеки не затронуты (upstream не менял `numerics/`) |

