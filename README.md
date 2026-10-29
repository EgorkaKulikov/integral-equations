# integral-equations

Kotlin code for spline collocation of integral equations on quadratic minimal splines, used to
produce the numerical results of the papers of E. K. Kulikov and A. A. Makarov.

## Requirements

- JDK 21 (Gradle comes with the wrapper).
- The libraries [`minimal-splines`](https://github.com/EgorkaKulikov/minimal-splines) and
  [`numerical-core`](https://github.com/EgorkaKulikov/numerical-core), versions from
  `gradle.properties`. They are downloaded from GitHub Packages, which needs a token with the
  `read:packages` scope: set `gpr.user` and `gpr.token` in `~/.gradle/gradle.properties`, or the
  variables `GITHUB_ACTOR` and `GITHUB_TOKEN`. Alternatively, run
  `./gradlew publishToMavenLocal` in each library.

## Reproducing the paper

*Choice of the spline space in superconvergent collocation schemes for linear integral
equations* (E. K. Kulikov, A. A. Makarov):

```sh
./gradlew paperTables
```

About 10 minutes on an 8-core laptop.

| Output | Contents |
|---|---|
| `build/paper-tables/*.tsv` | all tables and the figure, in the layout of the paper |
| `build/paper-tables/m4-diagnostics.tsv` | Problem M4 at α = 10⁻¹⁰: cond∞ of the base matrix and the spread of E_h between equivalent implementations |
| `build/sec4/*.txt`, `*.json` | raw values with full precision |

The accuracy check of the paper is the same command with
`-Dsec4.quad=16 -Dsec4.refine=1000`. Values below 10⁻¹² may differ in the last digits between
machines; for Problem M4, which is ill-conditioned, the values on the finest grids are
reproduced only to the digits printed in the paper.

### Model problems

All five problems are posed on [0, 1]. The right-hand side f is obtained by substituting the
exact solution u* into the equation; for every problem it is an elementary function of t, given
in the paper. The drivers compute f by the same composite Gauss–Legendre quadrature as the
matrix entries.

| Problem | Equation | Kernel K(t, s) | Exact solution u*(t) |
|---|---|---|---|
| M1 | Fredholm, second kind | 1/(1 + t + s) | 1/(1 + t) |
| M2 | Fredholm, second kind | cos(ω(t − s)) e^{−(t−s)²}, ω = 40 | (3 − 2t) e^{t²} cos ωt |
| M3 | Volterra, second kind | cos(ω(t − s)), ω = 30 | (2 − t) cos ωt |
| M4 | Fredholm, first kind, α = 10⁻¹⁰ | cos(ω(t − s)) e^{−(t−s)²}, ω = 20 | (9 − 5t) e^{t²} cos ωt |
| M5 | Volterra, first kind | 1 + (t − s) − sin(ω(t − s))/ω, ω = 30 | (2 − t) cos ωt |

The factor e^{t²} in M2 and M4 cancels the Gaussian factor of the kernel, so that f is
elementary. The table of solution frequencies (`tab:sol-freq`) keeps the kernels of M2 and M3
and replaces the solution by u₁, …, u₇ with the envelope 1/(1 + t) (see `PaperTablesTool`).

## Other commands

```sh
./gradlew fastTest       # quick checks, a few seconds
./gradlew check          # all tests and numerical gates, about 9 minutes
./gradlew runFredholm    # convergence tables: Fredholm equations
./gradlew runVolterra    # convergence tables: Volterra equations
./gradlew runUryson      # convergence tables: Uryson equations
./gradlew runBenchmark   # performance measurement
```

On a machine other than the one where the baselines were captured, add
`-PmachineDependentGates=false`.

## Repository layout

| Path | Contents |
|---|---|
| `src/main/kotlin/solvers/` | solvers: `fredholm`, `volterra`, `uryson`, `wsie` (weakly singular kernels), shared `core` |
| `src/problems/kotlin/problems/` | model problems |
| `src/demo/kotlin/demo/` | entry points of the `run*` tasks |
| `src/test/kotlin/` | tests and numerical gates |
| `src/test/kotlin/verification/` | drivers of the paper tables (`Sec4VerificationTool`, `PaperTablesTool`, `PaperTablesAssembler`) |
| `src/benchmark/` | performance benchmark |
| `tools/` | cross-check against SciPy |
| `docs/` | documentation, see below |

## Documentation

| File | Contents |
|---|---|
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | sources of the methods and how each is verified |
| [`docs/TESTING.md`](docs/TESTING.md) | test sets, Gradle tasks, continuous integration |
| [`docs/ACCURACY.md`](docs/ACCURACY.md) | attainable accuracy for equations of the first kind |
| [`docs/HPC.md`](docs/HPC.md) | performance and parallel matrix assembly |
| [`docs/baseline-changes.md`](docs/baseline-changes.md) | how numerical baselines are changed |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | rules for changes |

## License

Apache License 2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
