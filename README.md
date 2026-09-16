# integral-equations

Numerical solution of integral equations of the first and second kind by spline collocation
on quadratic minimal splines. Three classes of equations are considered: the linear Fredholm
equation `u(t) − λ∫ₐᵇ K(t,s) u(s) ds = f(t)` with constant limits of integration, the linear
Volterra equation `u(t) − λ∫ₐᵗ K(t,s) u(s) ds = f(t)` with a variable upper limit, and the
nonlinear Uryson equation `u(t) − ∫ₐᵇ K(t,s,u(s)) ds = f(t)`. The repository contains solvers,
model problems, demonstration runs and numerical verification; it is not published as a library.

## Rationale

For equations of the second kind the repository implements plain collocation, Sloan iteration,
the Kulkarni scheme and its iterated variant, the Nyström method, the combined operator
`L_n = P_χ L + (I − P_χ) L^N_h` and its iterated variant. All schemes are built over an
arbitrary generating system — polynomial, hyperbolic or trigonometric — and over five families
of approximation functionals, whereas known implementations are restricted to the polynomial
case. Equations of the first kind are ill-posed in the sense of Hadamard and are solved with
regularization: by reduction to an equation of the second kind with `c_L = −1/α` (Fredholm),
by differentiation (Volterra), and by a Tikhonov stabilizer with the parameter chosen by
Morozov's discrepancy principle (Uryson).

Every scheme comes with a reference to its primary publication, and its result is checked by
means independent of the project code: analytically exact solutions, published tables,
a cross-check against SciPy/NumPy and an independently written reference Nyström method.
The characterization baselines record that a result has not changed, but they do not prove
it correct, so the external verification means remain the primary ones.

Grids, splines, approximation functionals, quadrature and dense linear algebra are taken from
the libraries `minimal-splines` (packages `splines.*`) and `numerical-core` (packages
`numerics.*`); they are not duplicated in this repository.

## Build and run

JDK 21 or newer is required; Gradle is supplied by the wrapper. Both libraries are consumed as
published artifacts only, their versions are pinned in `gradle.properties`
(`numericalCoreVersion`, `minimalSplinesVersion`) and are resolved through `mavenLocal()` —
after `./gradlew publishToMavenLocal` in each library — or through GitHub Packages, reading
which requires a token with the `read:packages` scope (`gpr.user` and `gpr.token`
in `~/.gradle/gradle.properties`, or the variables `GITHUB_ACTOR` and `GITHUB_TOKEN`).

    ./gradlew fastTest      # fast set of checks, a few seconds
    ./gradlew build         # compilation of all source sets and numerical gates, about 9 minutes
    ./gradlew runFredholm   # convergence tables for the Fredholm equation
    ./gradlew runVolterra   # the same for the Volterra equation
    ./gradlew runUryson     # the same for the Uryson equation
    ./gradlew runBenchmark  # performance measurement

The `build` task includes gates bound to the machine on which the baselines were captured;
in continuous integration they are disabled by the property `-PmachineDependentGates=false`.

## Documentation

The limits of attainable accuracy for equations of the first kind are described in
`docs/ACCURACY.md`; computational efficiency, parallel matrix assembly and the benchmark
in `docs/HPC.md`; the sources of the schemes and the means of their verification in
`docs/REFERENCES.md`; the composition of the test sets, the Gradle tasks and continuous
integration in `docs/TESTING.md`; the protocol for changing the numerical baselines and the
history of their edits in `docs/baseline-changes.md`. The rules for making changes are set out
in `CONTRIBUTING.md`, and the program abstract for state registration — in `docs/ABSTRACT.md`
(kept in Russian, as required by the registration procedure).

## License

The project is distributed under the terms of the Apache License 2.0; the copyright holder is
Egor Konstantinovich Kulikov. The text of the license is in the file `LICENSE`, and information
about third-party components is in the file `NOTICE`.
