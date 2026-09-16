# AGENTS.md — rules for working with the `integral-equations` repository

The document is addressed to agents (and humans) making changes to this repository.
It must be read before any edit of the files.

## 1. Purpose of the repository

An applied research project: numerical solution of the **Fredholm, Volterra and Uryson**
integral equations of the first and second kind by spline collocation on quadratic minimal
splines. It contains:

- the solvers (`src/main/kotlin/solvers/{core,fredholm,volterra,uryson}`) and their common
  core (`solvers.core`: `SecondKindSolverCore`, `SolutionFunc`, `reportConvergence`,
  `RhsWithDerivatives`);
- model problems and analytical fixtures (`src/problems`);
- demonstrations of convergence tables and a benchmark (`src/demo`);
- end-to-end numerical verification: characterization baselines, cross-check against the
  publication, analytical solutions, cross-check against SciPy, convergence orders (`src/test`).

This is an **application, not a library**: the artifact is not published, and it has no
public API in the sense of external consumers.

## 2. Architectural boundaries

The direction of dependencies is strictly one-way and acyclic:

```
numerical-core  <--  minimal-splines  <--  integral-equations (this repository)
       ^                                          |
       +------------------------------------------+
```

| Library | Packages | Contents |
|---|---|---|
| `numerical-core` | `numerics.*`, `numerics.backend.*` | `GaussLegendre`, `LinearAlgebra`, backends, `NumericsContext`, `ParallelAssembly`, `Conditioning`, `orders`/`reliableOrders` |
| `minimal-splines` | `splines.*`, `splines.functionals.*`, `splines.metrics.*` | `Grid`, `GeneratingSystem`, `MinimalSplineBasis`, families of functionals, `SupportPoints`, `errorEh` |

Rules:

1. Importing `numerics.*` and `splines.*` is **allowed** — but ONLY through the
   published artifacts (`io.github.egorkakulikov:numerical-core`,
   `io.github.egorkakulikov:minimal-splines`), whose versions are pinned in
   `gradle.properties` (`numericalCoreVersion`, `minimalSplinesVersion`).
2. Adding spline code, quadrature, linear algebra, backends, reliability thresholds and the
   like here is **forbidden**. Such a change is made in the corresponding library, then
   `./gradlew publishToMavenLocal` there, then here — an update of the version in
   `gradle.properties` (if the version changed) and a run of the gates (section 6).
3. Dependencies `project(":...")` and `files("../numerical-core/...")`,
   `files("../minimal-splines/...")`, and any `srcDir` pointing at neighbouring repositories
   are **forbidden**. The `verifyArtifactDependencies` task (part of `check`) fails the build
   if a non-jar entry or a file from the `build/` directory of a neighbouring repository is
   found on the classpath.
4. Types `solvers.*`/`problems.*` must not appear in the libraries — if a solver needs a new
   capability of a library, that capability is formulated in the terms of the library
   (grid, basis, functional, linear system), not of the equation.
5. `SplineSpace` (the Tikhonov stabilizer, `gramR`, `omegaReg`) is part of the Uryson solver
   and stays here, even though it is built from `splines.*` objects.

## 3. Visibility policy

This is an application, not a library: the `public` modifier is not a compatibility promise.
The `main`, `problems`, `demo` and `benchmark` sets are compiled in `explicitApi()` mode,
so the visibility of every declaration is stated explicitly.
Symbols marked `internal` (`solvers.core.SecondKindDefaults`,
`solvers.core.IterationStopCriterion`, `solvers.uryson.NewtonRun`,
`solvers.uryson.runNewtonIterations`, `VolterraOperator.IntegrandCache` and its members)
stay `internal`; widening the visibility for the convenience of a test is not allowed — the
tests live in the same module and see `internal`.

## 4. Commands

| Command | What it does | Time |
|---|---|---|
| `./gradlew fastTest` | the fast set (tag `fast`): solvers, context propagation, regression, a fast subset of the orders | seconds |
| `./gradlew characterizationTest extraCharacterizationTest` | **the MAIN numerical-neutrality gate**: 1366 + 1344 values against `baseline-eh.tsv`/`baseline-extra.tsv`, tolerance 1e-9 | ~1 min |
| `./gradlew check` / `./gradlew build` | `fastTest` + both gates + `slowTest` + `verifyArtifactDependencies` + `koverVerify` | ~9 min |
| `./gradlew slowTest` | cross-check against the publication, cross-consistency, analytics, the full matrix of orders (grids up to n = 64) | ~8.5 min |
| `./gradlew convergenceOrderTest` | the full matrix of convergence orders, 168 combinations | 1.5–2 min |
| `./gradlew scipyVerify` | external cross-check against SciPy/NumPy; requires Python and network access (creates `.venv-verify`) | ~1 min + installation |
| `./gradlew captureBaseline` / `captureExtraBaseline` | capture `E_h` snapshots into `build/baseline/` — only by the protocol of section 6 | ~1 min |
| `./gradlew sec4Tables` | tables of §5–6 of the article into `build/sec4/` (`-Dsec4.quad=8|16`) | minutes |
| `./gradlew runFredholm` / `runVolterra` / `runUryson` / `runBenchmark` | demonstrations and the benchmark | — |
| `./gradlew verifyArtifactDependencies` | the libraries are on the classpath as jars only | seconds |

Before the first build the libraries must be present in `mavenLocal()`: run
`./gradlew publishToMavenLocal` in each of them. The alternative is
`-PnumericsRepositoryUrl=<url>`.

All test tasks receive `-Dnumerics.backend=auto` by default; an externally supplied value is
respected. The admissible values are `native` (the system BLAS/LAPACK implementation), `java`
(the portable netlib implementation) and `auto` (selection at startup). The baselines were
captured on the system implementation of the developer machine; a run with
`-Dnumerics.backend=java` diverges from them on the keys of the first-kind equation F1.

## 5. Numerical invariants that must not be violated

- **Convergence orders** from the expectation table of `convergence.ConvergenceOrderTest`
  (168 combinations of "scheme × basis × family"); degradation is caught by the fast subset in
  `fastTest`.
- **The convergence contract** of `solvers.core.reportConvergence`: an iterative scheme may not
  silently return a non-converged result; with `throwOnDivergence = true` it raises
  `IllegalStateException`, with `false` it returns `SolutionFunc.converged == false` and the
  `residual`.
- **Bitwise identity** of the result for `NumericsContext.parallel = true/false`
  (`solvers.core.NumericsContextWiringTest.parallelFlagDoesNotChangeResultBitwise`).
- **Context agreement**: a solver and its dependencies (`funcs`, `space`) must carry one and
  the same `NumericsContext`; a mismatch is a loud failure through
  `NumericsContext.requireSame`.
- **A single breakpoint inclusion tolerance** `Grid.breakpointInclusionEps` — the single
  source for `VolterraOperator` and `SplineSpace`.
- The characterization values of `baseline-eh.tsv`, `baseline-extra.tsv` and the published
  values of `published-values.tsv` (tolerance 2 %, a narrow class of 8 % for six keys of
  `table-f1.tex`).

## 6. Rules for changing the baselines

The full protocol is in [`docs/baseline-changes.md`](docs/baseline-changes.md). In brief:

1. Before changing the algorithm: `./gradlew captureBaseline captureExtraBaseline`, and save
   the snapshots outside `build/`.
2. After the change: capture the snapshots again and compare the keys pairwise; the
   intersection of the keys MUST coincide bitwise unless the change deliberately claims to
   alter the numbers.
3. If the numbers change deliberately (a defect fix) — an entry in
   `docs/baseline-changes.md` with the old and new values and the reason; the commit with the
   baseline is kept separate from the commit with the refactoring.
4. **Tuning the tolerances (1e-9, 2 %, 8 %) for the sake of a green build is FORBIDDEN.**
5. The machine-dependent gates (`@Tag("machine")`: both characterization classes and
   `PublishedValuesTest.fredholmFirstKindMatchesPublishedValues`) work only on the machine on
   which the baseline was captured (macOS aarch64, the system BLAS/LAPACK implementation).
   Locally they are always enabled; in CI they are disabled by `-PmachineDependentGates=false`.
   A difference in CPU architecture is not a proof of a mathematical regression — see
   `docs/TESTING.md`.

The procedure when a library changes: after modifying `numerical-core` or `minimal-splines` →
run `./gradlew test publishToMavenLocal` there → run
`./gradlew characterizationTest extraCharacterizationTest` here (then `check` before merging).
The libraries have no numerical baselines of their own — the gates of this repository serve as
their protection.

## 7. Regression tests

Every defect fix comes with a test in `regression.DefectRegressionTest` (or in the tests of the
corresponding solver) that **fails on the code before the fix**. The test is named after the
defect and contains a description: what the error was, why it was not caught earlier, and what
is checked now. The numbering of the defects is shared with the original monorepository
(defects 4 and 5 are in `numerical-core`).

## 8. New numerical algorithms

- A reference to the primary publication in [`docs/REFERENCES.md`](docs/REFERENCES.md) with a
  status ("Confirmed" / "Adaptation" / "No source"). If there is no source, this is stated
  explicitly; inventing references, DOIs and bibliographic data is forbidden.
- A new second-kind scheme goes into `SecondKindSolverCore` or next to it, with coverage in
  `ConvergenceOrderTest` (the expected order together with its justification) and, if an
  external reference exists, in `verification.*`.
- Distinguish between a proven estimate, a numerical observation, and an adaptation without
  a proof. The wording in KDoc and in the documentation must not overstate the status.
- The numerical baselines are extended by the protocol of section 6, not recaptured.

## 9. Language and style

Comments, KDoc, error messages, documentation and commit messages are written in English;
identifier names are in English as well. The only exception is
[`docs/ABSTRACT.md`](docs/ABSTRACT.md), which is kept in Russian as required by the state
registration procedure. KDoc explains **why** a decision was taken and which alternatives were
rejected, rather than paraphrasing the code.

## 10. Neighbouring repositories

| Repository | Role | Depends on |
|---|---|---|
| `numerical-core` | general-purpose numerical primitives | — |
| `minimal-splines` | minimal splines and functionals | `numerical-core` |
| `integral-equations` (this one) | solvers, problems, verification | both libraries |
| a future fourth project | other numerical methods on splines | `numerical-core`, `minimal-splines`; **not** on this repository |

The rules of the libraries are in their own `CONTRIBUTING.md`. The separation has been carried
out completely: no spline code or numerical infrastructure is present here.

## 11. Documents

| File | Purpose |
|---|---|
| [`README.md`](README.md) | purpose of the project, build, run, composition of the documentation |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | rules for changing the code and the documentation |
| [`docs/TESTING.md`](docs/TESTING.md) | Gradle tasks, composition of `check`, machine-dependent gates, CI, artifact dependencies |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | sources of the schemes, of the regularization and of the verification |
| [`docs/ACCURACY.md`](docs/ACCURACY.md) | the bound on `alpha` for equations of the first kind |
| [`docs/HPC.md`](docs/HPC.md) | the benchmark and parallel assembly in the solvers |
| [`docs/baseline-changes.md`](docs/baseline-changes.md) | protocol and history of the baseline edits |
| [`docs/ABSTRACT.md`](docs/ABSTRACT.md) | program abstract for state registration (kept in Russian) |
