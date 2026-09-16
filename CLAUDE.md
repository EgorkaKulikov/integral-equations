# CLAUDE.md

The rules for working with this repository are in [`AGENTS.md`](AGENTS.md); the rules for
making changes as a human contributor are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Read them
before any edit; the rules are not duplicated here, so that the two cannot drift apart.

Key points for a quick start:

- `./gradlew fastTest` — seconds; `./gradlew characterizationTest extraCharacterizationTest` —
  the main numerical-neutrality gate (~1 min); `./gradlew check` — everything (~9 min).
- The libraries `numerical-core` (`numerics.*`) and `minimal-splines` (`splines.*`) are
  consumed as Maven artifacts only; spline code and numerical infrastructure are not
  written here.
- Numerical baselines and tolerances are never tuned to make the build green —
  see `docs/baseline-changes.md`.
