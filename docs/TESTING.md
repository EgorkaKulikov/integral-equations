# Тесты, гейты и непрерывная интеграция

Подробный справочник по проверкам проекта. Краткая версия — в разделе
«Проверка качества» [README](../README.md).

Тесты разделены по времени выполнения тегами JUnit (`fast`, `slow`, `scipy`), на
каждый тег есть своя задача. Причина разделения: проверка после правки должна
занимать секунды, иначе её перестают делать.

## Задачи Gradle

```bash
./gradlew fastTest                  # быстрый набор: 289 тестов, единицы секунд
./gradlew slowTest                  # прогон по крупным сеткам: 23 теста, ~8.5 мин
./gradlew scipyVerify               # внешняя сверка со SciPy/NumPy: 9 тестов, 60 проверок
./gradlew characterizationTest      # гейт численной нейтральности: 5 тестов, ~27 с
./gradlew extraCharacterizationTest # гейт combinedNystrom и неравномерных сеток, ~19 с
./gradlew convergenceOrderTest      # полная матрица порядков сходимости: 168 сочетаний, 1.5-2 мин
./gradlew test                      # весь набор сразу — см. предупреждение ниже
./gradlew test --tests 'healthchecks.*'  # произвольная выборка по имени
./gradlew koverHtmlReport           # отчёт о покрытии
```

| Задача | Состав | В `check` |
|---|---|---|
| `fastTest` | всё, кроме перечисленного ниже: ядро, решатели, инварианты, regression, быстрый поднабор порядков сходимости | да |
| `characterizationTest` | `EhCharacterizationTest` против `baseline-eh.tsv` | да |
| `extraCharacterizationTest` | `ExtraCharacterizationTest` против `baseline-extra.tsv` (`combinedNystrom`, неравномерные сетки) | да |
| `slowTest` | `PublishedValuesTest`, `EhCharacterizationTest`, `ExtraCharacterizationTest`, `CrossSchemeConsistencyTest`, `AnalyticSolutionTest`, `ConvergenceOrderTest` — прогон по сеткам до n = 64 | да (явный `dependsOn`) |
| `convergenceOrderTest` | полная матрица `ConvergenceOrderTest`, 168 сочетаний | нет, но гоняется в CI (job `characterization`) и входит в `slowTest` |
| `scipyVerify` | `ScipyCrossVerificationTest` — быстрый, но требует venv с Python | нет (job `scipy` в CI) |

## Состав `./gradlew check`

Проверено `check --dry-run`: `fastTest`, `characterizationTest`,
`extraCharacterizationTest`, `slowTest` — объявлены явно в `build.gradle.kts` —
плюс `koverVerify` от плагина покрытия. Полный прогон занимает около 9 минут, из
которых `slowTest` — 8 мин 39 с.

**Почему `test` НЕ входит в `check`, а `slowTest` входит.** `test` дублирует
`fastTest` и `slowTest` вместе — самые дорогие классы гонялись бы дважды, а состав
гейта стал бы неявным. `slowTest` содержит единственную сверку с внешним источником
правды (`PublishedValuesTest`, 708 опубликованных чисел): набор, который нигде не
запускается, — не гейт, а декорация.

Включён он **явным `dependsOn("slowTest")`**; вариант «сделать его источником
покрытия Kover» проверен и отвергнут — `check --rerun-tasks` падает по памяти
(`exit value 137`, SIGKILL), поскольку инструментация Kover поверх матриц до n = 64
не укладывается в память. Цена выбора: строки, покрытые только slow-классами, не
попадают в отчёт покрытия — но сами проверки исполняются.

Для быстрого цикла правка-проверка используйте `fastTest` (единицы секунд) либо
`fastTest characterizationTest extraCharacterizationTest` (~1 мин).

> **Ловушка при добавлении новых задач типа `Test`.** Kover делает источником
> покрытия ЛЮБУЮ такую задачу, не перечисленную в `disabledForTestTasks`, а вместе
> с этим — зависимостью `koverVerify`, который уже в `check`. Именно так
> `convergenceOrderTest` одно время молча попал в `check` и раздул его, не добавив
> ни одной покрытой строки сверх быстрого поднабора того же класса; сейчас он из
> источников покрытия исключён.

## Категории тестов по назначению

Ортогонально тегам по времени:

| Категория | Назначение |
|---|---|
| `numerics.*` | модульные тесты ядра: сетки, квадратура, сплайны, линейная алгебра |
| `verification.*` | **независимая верификация**: сверка со SciPy/NumPy, аналитически точные решения, опубликованные значения, перекрёстная согласованность схем |
| `convergence.*` | контракт сходимости и эмпирические порядки сходимости |
| `healthchecks.*` | математические инварианты: биортогональность, разбиение единицы, точность на span |
| `characterization.*` | фиксация численных результатов (1366 значений, допуск 1e-9) |
| `regression.*` | воспроизведение ранее обнаруженных дефектов |
| `solvers.*` | поведение решателей |

## Выбор бэкенда линейной алгебры в тестах

Все тестовые задачи получают системное свойство `numerics.backend` (по умолчанию
`multik`). Это существенно: эталон `baseline-eh.tsv` снят на multik/OpenBLAS и
**привязан к нему** — прогон `-Dnumerics.backend=reference` даёт 4/4 падения
`EhCharacterizationTest` с расхождением до 5.7e-2 при допуске 1e-9. Причина
закономерна: уравнение первого рода F1 плохо обусловлено, и разные реализации LU
на нём расходятся. Внешнее значение свойства уважается и перекрывает умолчание,
поэтому прогон на обоих бэкендах возможен явно.

## Служебные задачи

Не входят в обычный прогон:

| Задача | Назначение |
|---|---|
| `captureBaseline` | переснять эталонный снимок `E_h` в `build/baseline/` (источник `baseline-eh.tsv`) |
| `captureExtraBaseline` | переснять дополнительный снимок `baseline-extra.tsv` (`combinedNystrom`, неравномерные сетки, отрезок `[0,2]`) |
| `dumpVerificationArtifacts` | выгрузить внутренние артефакты для ручного разбора расхождения |
| `setupScipyVerification` | подготовить окружение Python со SciPy (вызывается автоматически из `scipyVerify`) |

Задачи `captureBaseline` и `captureExtraBaseline` следует запускать осознанно —
только когда изменение алгоритма обосновано, с фиксацией старых и новых значений
(см. [`baseline-changes.md`](baseline-changes.md)).

## Непрерывная интеграция

Конфигурация — [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Запуск:
push в любую ветку, pull request и ежедневно в 03:00 UTC. Разделение на job'ы
повторяет разделение тестов по тегам: обратная связь на PR обязана быть быстрой.

| Job | Что гоняет |
|---|---|
| `fast` | `fastTest --tests numerics.backend.BackendSpiTest` (диагностика бэкенда), затем `fastTest koverXmlReport compileDemoKotlin compileProblemsKotlin` |
| `characterization` | `characterizationTest` — гейт численной нейтральности, затем `convergenceOrderTest` — полная матрица порядков сходимости |
| `scipy` | `scipyVerify` с кэшированным `.venv-verify` |
| `full` | `slowTest` — 23 теста, включая единственную сверку с опубликованными таблицами (`timeout-minutes: 45`) |

Три детали, важные при правках CI:

1. Job `fast` начинается с отдельного прогона `BackendSpiTest` — без него молчаливый
   откат с multik на `ReferenceBackend` выглядел бы как «рефакторинг испортил числа».
2. `full` НЕ включен с `continue-on-error: true` — зелёная галочка при невыполненных
   проверках хуже отсутствующего job'а.
3. У `full` НЕТ условия `if:` на ветку `main` — гейт, работающий только после мержа,
   сообщает о поломке слишком поздно.
