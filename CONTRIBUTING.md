# Rules for changing integral-equations

The repository contains solvers for integral equations written in Kotlin for the JVM platform.
The rules below apply to any change to the code or the documentation.

## Language

Comments, KDoc, error messages, documentation and commit messages are written in English;
identifiers in the code are in English as well. The only exception is `docs/ABSTRACT.md`,
the program abstract for state registration, which is kept in Russian as required by the
registration procedure. The convention matches the one adopted in the libraries
`numerical-core` and `minimal-splines`. The exposition is descriptive: no words in capital
letters, no jargon, no references to the history of edits.

## Repository structure

The source sets are separated by purpose: `main` — the solvers
(`solvers/{core,fredholm,volterra,uryson}`), `test` — tests and numerical gates,
`problems` — model problems and analytical fixtures, `demo` — convergence tables
and entry points, `benchmark` — performance measurement. The `main`, `problems`,
`demo` and `benchmark` sets are compiled in `explicitApi()` mode, so the visibility of every
declaration is stated explicitly.

Splines, approximation functionals, quadrature, linear algebra and the computation context
come from the libraries `minimal-splines` and `numerical-core` and are consumed only as
published artifacts. Dependencies of the form `project(":…")` and `files("../…")` are
forbidden; their absence is checked by the `verifyArtifactDependencies` task, which is part of
`check`. A change that concerns splines or numerical infrastructure is made in the
corresponding library, followed by a version update in `gradle.properties`.

## Commands

    ./gradlew fastTest                   # the fast-tagged set: 142 tests, a few seconds
    ./gradlew slowTest                   # the slow-tagged set: 23 tests, about 8.5 minutes
    ./gradlew scipyVerify                # the scipy-tagged set: 9 tests, requires a Python environment
    ./gradlew setupScipyVerification     # creation of the Python environment for the SciPy cross-check
    ./gradlew characterizationTest extraCharacterizationTest   # numerical-neutrality gates
    ./gradlew convergenceOrderTest       # the full matrix of convergence orders, 168 combinations
    ./gradlew check                      # all the gates listed above and the coverage threshold, about 9 minutes

The `check` task consists of `fastTest`, `characterizationTest`, `extraCharacterizationTest`,
`slowTest`, `verifyArtifactDependencies` and `koverVerify`. A full reference for the tasks,
the machine-dependent gates and continuous integration is given in `docs/TESTING.md`.

## Test tags

Every test class carries exactly one runtime tag: `fast`, `slow` or `scipy`. A class without
such a tag falls into no set and is therefore never executed, while a class with two tags is
executed twice. The convention is checked by the guard test `conventions.TagConventionTest`
over the compiled classes; the tag may be placed either on the class or on every test method.

## Numerical baselines

The values in `baseline-eh.tsv` (1366 keys) and `baseline-extra.tsv` (1344 keys) are changed
only by the protocol described in `docs/baseline-changes.md`: a snapshot before the change,
a snapshot after it, a pairwise comparison of the keys and an entry in the same document with
the old and new values and the reason. The commit with the baseline is kept separate from the
commit with the code change. Tuning the tolerances for the sake of a successful build is
not permitted.

The baseline `published-values.tsv` (708 published numbers) is never edited by hand at all:
it is generated from the .tex tables of the article by the script
`tools/parse_published_values.py`, which reproduces the file byte for byte; the order of the
steps is described in `docs/TESTING.md`.

## Coverage

The coverage threshold — 67 % of lines and 72 % of branches — is checked by the `koverVerify`
task as part of `check`. The only source of coverage is `fastTest`, so the thresholds are
measured for it; a change that drops the coverage below the threshold fails the build.

## Documentation and commits

After a change in solver behaviour, `README.md` and the corresponding document of the `docs`
directory are updated. A new numerical method comes with a reference to its primary publication
in `docs/REFERENCES.md`, including its status; if no source exists, this is stated explicitly.
Commit messages are written in English with an area prefix: `api:`, `algo:`, `test:`,
`docs:`, `build:`, `ci:`. One change corresponds to one commit.
