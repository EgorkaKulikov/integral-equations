# Tests, gates and continuous integration

A detailed reference for the checks of the project. A short version is in the section
"Quality control" of the [README](../README.md).

The tests are separated by execution time through JUnit tags (`fast`, `slow`, `scipy`), and
each tag has a task of its own. The reason for the separation: a check after an edit must take
seconds, otherwise it stops being performed.

## Gradle tasks

```bash
./gradlew fastTest                  # the fast set: 134 tests, a few seconds
./gradlew slowTest                  # a run over large grids: 23 tests, ~8.5 min
./gradlew scipyVerify               # external cross-check against SciPy/NumPy: 9 tests, 60 checks
./gradlew characterizationTest      # the numerical-neutrality gate: 5 tests, ~27 s
./gradlew extraCharacterizationTest # the gate for combinedNystrom and non-uniform grids, ~19 s
./gradlew convergenceOrderTest      # the full matrix of convergence orders: 168 combinations, 1.5-2 min
./gradlew test                      # the whole set at once — see the warning below
./gradlew test --tests 'healthchecks.*'  # an arbitrary selection by name
./gradlew koverHtmlReport           # the coverage report
./gradlew verifyArtifactDependencies # the libraries are on the classpath as jars only (part of check)
```

| Task | Composition | In `check` |
|---|---|---|
| `fastTest` | everything except what is listed below: solvers, context propagation, regression, a fast subset of the convergence orders, diagnostics of the active backend | yes |
| `characterizationTest` | `EhCharacterizationTest` against `baseline-eh.tsv` | yes |
| `extraCharacterizationTest` | `ExtraCharacterizationTest` against `baseline-extra.tsv` (`combinedNystrom`, non-uniform grids) | yes |
| `slowTest` | `PublishedValuesTest`, `EhCharacterizationTest`, `ExtraCharacterizationTest`, `CrossSchemeConsistencyTest`, `AnalyticSolutionTest`, `ConvergenceOrderTest` — a run over grids up to n = 64 | yes (explicit `dependsOn`) |
| `convergenceOrderTest` | the full matrix of `ConvergenceOrderTest`, 168 combinations | no, but it runs in CI (job `characterization`) and is part of `slowTest` |
| `scipyVerify` | `ScipyCrossVerificationTest` — fast, but requires a venv with Python | no (job `scipy` in CI) |

## Composition of `./gradlew check`

Verified with `check --dry-run`: `fastTest`, `characterizationTest`,
`extraCharacterizationTest`, `slowTest`, `verifyArtifactDependencies` — declared explicitly
in `build.gradle.kts` — plus `koverVerify` from the coverage plugin. A full run takes about
9 minutes, of which `slowTest` accounts for 8 min 39 s.

**Why `test` is NOT part of `check` while `slowTest` is.** `test` duplicates
`fastTest` and `slowTest` taken together — the most expensive classes would be run twice and the
composition of the gate would become implicit. `slowTest` contains the only cross-check against
an external source of truth (`PublishedValuesTest`, 708 published numbers): a set that is run
nowhere is not a gate but a decoration.

It is included by an **explicit `dependsOn("slowTest")`**; the alternative of "making it a
source of Kover coverage" was tried and rejected — `check --rerun-tasks` fails on memory
(`exit value 137`, SIGKILL), since the Kover instrumentation on top of matrices up to n = 64
does not fit into memory. The price of the choice: the lines covered only by the slow classes do
not appear in the coverage report — but the checks themselves are executed.

For a fast edit-check cycle use `fastTest` (a few seconds) or
`fastTest characterizationTest extraCharacterizationTest` (~1 min).

> **A trap when adding new tasks of type `Test`.** Kover makes a source of coverage out of ANY
> such task not listed in `disabledForTestTasks`, and with that a dependency of `koverVerify`,
> which is already in `check`. That is exactly how `convergenceOrderTest` once silently ended up
> in `check` and inflated it without adding a single covered line beyond the fast subset of the
> same class; at present it is excluded from the sources of coverage.

## Categories of tests by purpose

Orthogonal to the tags by time:

| Category | Purpose |
|---|---|
| `solvers.core.*` | propagation of `NumericsContext` through the solvers (`NumericsContextWiringTest`), diagnostics of the active backend (`ActiveBackendDiagnosticTest`); the unit tests of quadrature and linear algebra are in the `numerical-core` repository, those of splines and functionals in `minimal-splines` |
| `verification.*` | **independent verification**: cross-check against SciPy/NumPy, analytically exact solutions, published values, cross-consistency of the schemes |
| `convergence.*` | the convergence contract and the empirical convergence orders |
| `healthchecks.*` | correctness checks of the solvers (Fredholm, Volterra, Uryson); the spline invariants (biorthogonality, partition of unity, exactness on the span) are in `splines.SplineCoreHealthCheckTest` in `minimal-splines` |
| `characterization.*` | recording of the numerical results (2710 values in two baselines; the comparison rule is selected by the class of the key) |
| `regression.*` | reproduction of previously discovered solver defects (the linear-algebra defects 4 and 5 are in `numerical-core`) |
| `solvers.*` | behaviour of the solvers |

## The machine-dependent gate (tag `machine`)

Exactly one method carries the tag `machine` —
`verification.PublishedValuesTest.fredholmFirstKindMatchesPublishedValues`.

| What is tagged | Volume | Where it is excluded |
|---|---|---|
| `PublishedValuesTest.fredholmFirstKindMatchesPublishedValues` (ONE method) | 42 keys of F1 | job `full` in CI (`-PmachineDependentGates=false`) |

**By default the flag is enabled**: locally the method is always executed, including in
`./gradlew check` and `build`. There is no separate option to "skip a whole task" in
`build.gradle.kts` any more — no task is left in which ALL tests are machine-dependent, and an
exclusion by tag for a single method suffices.

### Why only this one remains

The cross-check is made against the numbers printed in the article, and they cannot be
"recaptured" for another platform. The divergence of LU implementations arises in the last bits,
but on the ill-conditioned problems F1 (`cond(I − M) ~ 2.2e10`) it is amplified by many orders of
magnitude: in the `sloan` scheme two terms of the order 1.38e10 cancel down to O(1). With a full
replacement of the LU implementation (`-Dnumerics.backend=java` — a perturbation cruder than a
change of architecture) only 11 of the 655 published values diverge, and all of them are in F1 —
by up to 11.49 % against a tolerance of 2 %:

| Group | Values checked | Result |
|---|---|---|
| Fredholm of the second kind | 260 | all within tolerance |
| Volterra of the second kind | 341 | all within tolerance |
| Volterra of the first kind | 12 | all within tolerance |
| **Fredholm of the first kind (F1)** | 42 | **11 discrepancies, up to 11.49 %** |

### The characterization gates have ceased to be machine-dependent

Until 2026-09-16 the tag `machine` was carried by `EhCharacterizationTest` and
`ExtraCharacterizationTest` as well: all values were compared with a single tolerance of 1e-9,
and CI on `ubuntu-latest` (x86_64) was red **from the very first run (2026-07-30)** precisely on
those gates, although locally both classes were green. The cause has been named and removed in
substance. The baselines received a third column — the class of the key (`portable` 1984,
`sensitive` 54, `residual` 336, `exact` 336): the 54 sensitive keys of F1 are compared against
the bound `2·cond·max(ω, ε)·‖u‖∞`, computed in the same run, and the rest by the former strict
rule, which covers the measured divergence of the backends (≤ 1.0e-14) with a margin of a factor
of 60. The class is not assigned by hand: it is computed by the `classifyBaseline` task from the
snapshots of the two LU routes (`captureBaselineBothBackends`), and the composition of the
`sensitive` class is fixed by the guard test `BaselineClassGuardTest`. The protocol and the
measurements are in [`baseline-changes.md`](baseline-changes.md), the entry of 2026-09-16.

The outcome: both gates are green on `-Dnumerics.backend=native` and on `java` alike, and both
are executed in CI — in the job `characterization` together with `convergenceOrderTest`. The run
on ubuntu/OpenBLAS gives a third independent LU route and serves as the main confirmation of the
classification.

### Rejected alternatives

**A second baseline for linux-x86_64.** It cannot be captured locally — only through CI, while
capturing a baseline is a central routine of the project: every change of an algorithm would
require a double recapture with a round trip through CI. More importantly, this **does not fix
`PublishedValuesTest`**: there the cross-check is made against the numbers from the article, and
they cannot be "recaptured" for a platform.

**Moving the gates to `macos-latest`.** Fewer runners (a queue), 3 cores against 4 —
`slowTest` would grow from 8.5 min to 15-25; and it would still break on a change of the runner
image.

**Relaxing the tolerance** to ~12 % is forbidden by the rules of the project and would render the
cross-check meaningless.

### What CI checks

The fast set of this repository (the fast tests of the libraries are run in their own CI),
9 SciPy cross-check tests, 168 combinations of the convergence order, both characterization
gates (2710 values), cross-consistency of the schemes, the analytical solutions and 613 of the
655 published values. The only thing not checked is the 42 keys of F1 — that same
machine-dependent method; its gate lives in `./gradlew check` on the developer machine.

> A gate that is always red is no stricter than a missing one — it stops being read.
> A cross-check against the numbers from the article on foreign hardware does not make it
> stricter — it makes it permanently red.

## Origin of the baseline of published values

The 708 numbers in `src/test/resources/verification/published-values.tsv` are extracted
mechanically from the 18 files `table-*.tex` of the article by the script
[`tools/parse_published_values.py`](../tools/parse_published_values.py); its output,
including the comment header, coincides byte for byte with the committed file. Manual transfer
was deliberately not used: it is itself the "transfer error" for the detection of which the
cross-check exists. For that reason the values are never edited by hand — a change in the tables
of the article is reflected by re-running the script:

    python3 tools/parse_published_values.py --tables-dir <directory of the article tables>
    python3 tools/parse_published_values.py --tables-dir <directory> --check

The second command writes nothing: it compares the generated file with the committed one and
returns 1 on a discrepancy, listing the differing keys. It is not built into `check` or into CI —
the sources of the article lie outside the repository.

## Selection of the linear-algebra backend in the tests

All test tasks receive the system property `numerics.backend`; the admissible values are
`native` (native BLAS/LAPACK through netlib), `java` (pure-Java F2J) and the default `auto`
(native, if it came up, otherwise SILENTLY F2J). An externally supplied value of the property is
respected and overrides the default, so a run on both backends is possible explicitly — this is
what the matrix of the `fast` job in CI does (see below), and the classification of the baselines
(`captureBaselineBothBackends`) is built on the same mechanism.

Previously the baseline `baseline-eh.tsv` was bound to the native LAPACK: a run with
`-Dnumerics.backend=java` produced failures of `EhCharacterizationTest` with a divergence of up
to 5.7e-2 against a single tolerance of 1e-9, because the first-kind equation F1 is
ill-conditioned and different LU implementations diverge on it. After the introduction of the
class column both characterization gates are green on both backends; a silent fallback from the
native LAPACK to F2J is caught by `ActiveBackendDiagnosticTest`.

## Artifact dependencies

The libraries `numerical-core` (package `numerics.*`) and `minimal-splines` (packages
`splines.*`) are consumed **only as Maven artifacts** from `mavenLocal()`, from GitHub Packages
or from the repository given by the property `-PnumericsRepositoryUrl=...`; the versions are
pinned in `gradle.properties` (`numericalCoreVersion`, `minimalSplinesVersion`). No
`project(":...")` and no `files("../…")` — otherwise the three repositories would have remained
a monorepository.

The `verifyArtifactDependencies` task (part of `check`) inspects the compile classpath:
each of the two libraries is present as exactly one jar file, and no classpath entry is a
directory or a file from the `build/` directory of a neighbouring repository. An accidental
dependency on the sources of a neighbour fails `check`.

In CI both libraries are taken from GitHub Packages (the registries `EgorkaKulikov/numerical-core`
and `EgorkaKulikov/minimal-splines`, declared in `build.gradle.kts`): the workflow passes the
built-in `GITHUB_TOKEN` with the `packages: read` permission, and there is no cloning or local
publication of the neighbouring repositories.

## Auxiliary tasks

Not part of an ordinary run:

| Task | Purpose |
|---|---|
| `captureBaseline` | recapture the reference `E_h` snapshot into `build/baseline/` (the source of `baseline-eh.tsv`) |
| `captureExtraBaseline` | recapture the additional snapshot `baseline-extra.tsv` (`combinedNystrom`, non-uniform grids, the interval `[0,2]`) |
| `captureBaselineBothBackends` | capture both matrices on `-Dnumerics.backend=java` and `native` (a prerequisite of `classifyBaseline`) |
| `classifyBaseline` | compute the class column from the snapshots of both backends (`build/baseline/classified`) |
| `dumpVerificationArtifacts` | export the internal artifacts for a manual analysis of a discrepancy |
| `setupScipyVerification` | prepare the Python environment with SciPy (invoked automatically from `scipyVerify`) |

The tasks `captureBaseline` and `captureExtraBaseline` are to be run deliberately — only when a
change of the algorithm is justified, with the old and new values recorded
(see [`baseline-changes.md`](baseline-changes.md)).

## Continuous integration

The configuration is [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Triggers:
a push to `main`, a pull request (which covers all the other branches) and a daily run at
03:00 UTC. The former `push: ['**']` together with `pull_request` ran every branch with a PR
twice. The division into jobs repeats the division of the tests by tag: the feedback on a PR
must be fast.

The flag `-PmachineDependentGates=false` is passed only by the job `full` — it excludes the
single machine-dependent method (see the section above). There is NO publication of the
libraries into `mavenLocal` in CI: `numerical-core` and `minimal-splines` are taken from
GitHub Packages with the built-in `GITHUB_TOKEN` carrying the `packages: read` permission
(see the section "Artifact dependencies").

| Job | What it runs |
|---|---|
| `fast` | the MATRIX `backend: [java, native]`; a single command `fastTest koverXmlReport koverVerify verifyArtifactDependencies compileDemoKotlin compileProblemsKotlin -Dnumerics.backend=<leg>` |
| `characterization` | `characterizationTest`, `extraCharacterizationTest` and `convergenceOrderTest` — both numerical-neutrality gates (2710 values) and the full matrix of convergence orders |
| `scipy` | `scipyVerify` with a cached `.venv-verify` |
| `full` | `slowTest` without the machine-dependent method in a single step: the cross-check against the publication (613 values), cross-consistency, analytics, convergence orders (`timeout-minutes: 45`) |

### Where and why parts of `check` are run

`./gradlew check` is NOT run in CI as a whole: it pulls in `fastTest` + `slowTest` + both
characterization sets, that is, it would repeat the job `full` (another ~10 min for each leg of
the matrix). For that reason the job `fast` invokes by name the two parts of `check` that are not
tests and were therefore not checked in CI at all before:
`verifyArtifactDependencies` and `koverVerify`. Both cost almost nothing: the only source of
coverage is `fastTest`, which is executed in that job anyway.

### Diagnosability of failures in the job `full`

The logs of the job and its artifacts through the REST API require admin rights on the
repository (`403`), and the check-run annotations return only `Process completed with exit code 1`
— without the names of the failed tests. For that reason `slowTest` is followed by a step with
`if: ${{ !cancelled() }}` that parses the JUnit reports and prints the names of the failed tests
and the discrepancies themselves through the `::error::` command. The annotation is visible both
in the UI and through the API without privileges, that is, diagnosability has ceased to be a
privilege of the owner without splitting the set into per-class steps and without extra starts of
the Gradle daemon.

Three details that matter when editing CI:

1. `solvers.core.ActiveBackendDiagnosticTest` is part of `fastTest` (tag `fast`) and runs
   together with the whole set, without a second start of Gradle. It asserts that the active
   backend agrees with the property `numerics.backend` (`Backends.native()`/`java()` from
   `numerical-core`; it replaced `BackendSpiTest`, which moved there as well) — without it a
   silent fallback from the native LAPACK to F2J would look like "a refactoring spoilt the
   numbers". The line `numerical-core: ...` printed by the test is written to the log by a
   separate step from the `<system-out>` of the JUnit report — without `--info` on all 142 tests.
2. `full` is NOT enabled with `continue-on-error: true` — a green tick with unperformed checks is
   worse than a missing job.
3. `full` has NO `if:` condition on the `main` branch — a gate that works only after a merge
   reports a breakage too late.
