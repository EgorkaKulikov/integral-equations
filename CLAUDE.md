# CLAUDE.md

Правила работы с этим репозиторием — в [`AGENTS.md`](AGENTS.md). Прочитай его перед
любой правкой; здесь правила не дублируются, чтобы не расходились.

Ключевое для быстрого старта:

- `./gradlew fastTest` — секунды; `./gradlew characterizationTest extraCharacterizationTest` —
  главный гейт численной нейтральности (~1 мин); `./gradlew check` — всё (~9 мин).
- Библиотеки `numerical-core` (`numerics.*`) и `minimal-splines` (`splines.*`)
  подключаются только как Maven-артефакты; код сплайнов и численной инфраструктуры
  здесь не пишется.
- Численные эталоны и допуски не подгоняются под зелёную сборку —
  см. `docs/baseline-changes.md`.
