# Sources of the numerical methods

The document links every implemented numerical method to its primary publication and records
the status of the correspondence: confirmed by a comparison of the formulas, an adaptation, or
an original construction with no published counterpart.

The verification was carried out by matching the formulas of the code against the formulas of
the sources. For the methods that have no published primary source this is stated explicitly —
such places are **not to be** accompanied by references to the literature.

## Notation for the status

| Status | Meaning |
|---|---|
| **Confirmed** | the formula of the code coincides literally with the formula of the publication |
| **Adaptation** | the method is carried over to a class of problems for which it is not proved in the source |
| **No source** | no published counterpart has been found; the construction is original |

---

## 1–2. Minimal splines and approximation functionals

The implementation of the basis of minimal splines, of the generating systems (`B`, `H`, `T`),
of the grid with multiple nodes and of all the families of approximation functionals (`theta`,
`xi`, `xitilde`, `mu`, `lambda`) lives in the repository `minimal-splines`; the tables
"element → implementation → source → status" for those objects are there as well:
[`minimal-splines/docs/REFERENCES.md`](https://github.com/EgorkaKulikov/minimal-splines/blob/main/docs/REFERENCES.md).
The sources on which those tables rely are:
[Demyanovich 1994], [Makarov 2012], [Kosogorov, Makarov 2017], [Kulikov, Makarov 2019a],
[Kulikov, Makarov 2019b], [Kulikov, Makarov 2020], [Kulikov, Makarov 2022],
[Kulikov, Makarov 2025]. The bibliographic records are kept in the list of references below as
well, since some of them are cited by sections 3–6.

## 3. Schemes for equations of the second kind

| Element | Implementation | Source | Status |
|---|---|---|---|
| Plain collocation `(I - M)c = g` | `SecondKindSolverCore.base` (Fredholm, Volterra) | [Dagnino, Remogna, Sablonnière 2014] | Confirmed |
| Sloan iteration | `SecondKindSolverCore.sloan` | [Sloan 1976] | Confirmed |
| Kulkarni scheme (projectors) `(I - M - M2 + M^2)c = (I - M)g + d` | `kulkarniProjector` | [Kulkarni 2003] | Confirmed |
| Reconstruction `u^K = y_h + (I - P)[f + L y_h]` | `kulkarniProjector` | [Kulkarni 2003] | Confirmed |
| Iterated Kulkarni | `iteratedKulkarni` | [Kulkarni 2003] | Confirmed |
| Kulkarni scheme for quasi-interpolants (`mu`, `lambda`) | `kulkarniQuasi` | — | **No source**: the Kulkarni theory essentially requires `P^2 = P`; the status is a numerical observation. It is implemented by plain iteration and requires contraction (a spectral radius below one) — established on the problems `A-sep2`, `A-sep3` |
| Classical Nyström (bare quadrature) | `nystrom`, `iteratedNystrom` | [Atkinson 1997] | Confirmed; **it does not raise the order** |
| Combined Nyström `L_n = P_chi L + (I - P_chi)L^N_h` (Fredholm) | `combinedNystrom` | [Allouch et al. 2021], [Remogna, Sbibih, Tahrichi 2023] | Confirmed |
| Combined Nyström (Volterra) | `combinedNystrom` | — | **Adaptation**: there are no proofs for a variable upper limit |

### Remark on the convergence orders

The published estimates `O(h^7)` for the combined operator and `O(h^8)` for its iterated
variant are proved **only** for the Fredholm equation with a uniform grid, polynomial
B-splines and a quasi-interpolation projector.

The orders measured in the project (`CombinedNystromTest`):

* the Fredholm equation, problem F2, basis B, family `theta`: classical Nyström — about 4.2,
  combined — about 7.0 (in agreement with the theory);
* the Volterra equation: **no gain is observed**, the order of both variants is about 3.8
  (see `CombinedNystromVolterraTest`).

For the Volterra equation there are no counterparts of the superconvergence theorems in the
known literature: a variable upper limit leads to weights that depend on the collocation point
and to a truncation of the last cell, which requires a separate analysis.

## 4. Equations of the first kind

| Element | Implementation | Source | Status |
|---|---|---|---|
| Fredholm regularization `(alpha I + K)u = f` | `solvers/fredholm`, `FredholmFirstKindSolver` | [Wazwaz 2011a] | Confirmed |
| The value `alpha = 1e-10` | `FredholmFirstKindSolver` | [Kulikov, Makarov 2023] | An experimental choice of the authors, not a recommendation of [Wazwaz 2011a]. It falls into the range of accuracy loss: at `alpha = 1e-10` a spread of 15.19 % between algebraically equivalent formulations was measured at `cond_inf ≈ 2.636149e+10`, while at `1e-6`/`1e-8` there is no discrepancy (0.00 %/0.01 %) — see [`ACCURACY.md`](ACCURACY.md) |
| Reduction of Volterra I → II kind by differentiation | `solvers/volterra`, `VolterraFirstKindSolver` | [Wazwaz 2011b], [Brunner 2004] | Confirmed for the case `m = 1`; it requires `K(t,t) != 0` at every breakpoint, which is checked by `safeDiagonal` |
| The formulas `(Vu)'` and `(Vu)''` (the Leibniz rule) | `VolterraOperator.applyDeriv`, `applyDeriv2` | [Makarov, Kulikov 2026] | `(Vu)'` is confirmed; `(Vu)''` has no separate publication. Both formulas are cross-checked numerically against `scipy.integrate.quad` by the layer L4/L5 checks `V/V2/rhsDeriv`, `V/V2/rhsDeriv2`, `V/V2exp/rhsDeriv`, `V/V2exp/rhsDeriv2`, `V/V2win/rhsDeriv`, `V/V2win/rhsDeriv2` (the reference was derived independently of the code: `volterra_image_deriv` in `tools/verify_with_scipy.py`, including the FULL derivative of the diagonal `d/dt K(t,t) = K_t(t,t) + K_s(t,t)`); the actual deviation is at the machine precision, up to 6.7e-16 against a tolerance of 1e-10 |

## 5. The nonlinear Uryson equation

| Element | Implementation | Source | Status |
|---|---|---|---|
| The Uryson operator and the Fréchet derivative | `UrysohnOperator` | [Krasnoselskii 1964], [Zeidler 1986] | Confirmed |
| Newton with the analytical Jacobian `B(c)_{j,i} = theta_j(U'(x_h) omega_i)` | `CollocationCore.bMatrix` | [Atkinson 1997] | Confirmed |
| Kulkarni scheme (quasi-Newton with a preconditioner) | `UrysonSecondKindSolver.kulkarni` | [Kulkarni 2003], [Dagnino, Dallefrate, Remogna 2019] | Confirmed |
| Spline Nyström for Uryson | `UrysonSecondKindSolver.nystrom` | [Remogna, Sbibih, Tahrichi 2023] | Confirmed |
| Combined Nyström for Uryson `L_n = P_theta L + (I - P_theta) L^N_h`, Newton with the analytical Jacobian | `UrysonSecondKindSolver.combinedNystrom`, `CombinedNystromSolver` | [Remogna, Sbibih, Tahrichi 2023] | Confirmed: it is precisely this operator to which the superconvergence estimates of the source refer; cross-checked against `FredholmSecondKindSolver.combinedNystrom` on a linear kernel (`CombinedNystromTest`) |
| Iterated combined Nyström `u = f + lambda L u^N_h` | `UrysonSecondKindSolver.iteratedNystrom` | [Sloan 1976] | Adaptation: a single application of the exact operator to the combined-Nyström approximation |
| Iterated Kulkarni `u = f + lambda L u^K_h` | `UrysonSecondKindSolver.iteratedKulkarni` | [Sloan 1976], [Kulkarni 2003] | Adaptation: the Sloan iteration applied to the Kulkarni approximation for a nonlinear operator |
| Tikhonov regularization, the stabilizer `R_h` in the `W^{1,2}` norm | `SplineSpace.gramR` | [Tikhonov, Arsenin 1977], [Engl, Hanke, Neubauer 1996] | Confirmed |
| Gauss–Newton for the regularized problem | `UrysonFirstKindSolver.solveFixedAlpha` | [Engl, Hanke, Neubauer 1996] | Confirmed (sign for sign) |
| Morozov's discrepancy principle | `UrysonFirstKindSolver.solveMorozov` | [Engl, Hanke, Neubauer 1996], sect. 4.3 | Confirmed |
| Homotopy over a decreasing `alpha` with a warm start | `solveMorozov` | — | An implementation detail |
| The noise model (a piecewise-linear profile), the normalization `NoiseNorm { L2, SUP }` | `noisyRightHandSide`, `noisyThetaCoefficients` | — | An implementation detail; the determinism (an explicit `seed`) and the scaling to a prescribed norm correspond to the formulation |

---

## 6. Independent verification: by what each method is checked

Sections 1–5 link the methods to publications (a correspondence of FORMULAS). The present
section records something different: by what means **independent of the project code** the
RESULT of the computations has been checked.

The reason for a separate section: the characterization tests
(`characterization/baseline-eh.tsv`) were captured by the VERY implementation they check: they
record that the result is unchanged, but not that it is correct. An error introduced before the
baseline was created would have been recorded together with it.

### The means of verification

| Symbol | Means | Implementation | Degree of independence |
|---|---|---|---|
| **A** | Analytically exact solutions (separable kernels, convolution/Laplace, MMS) | `problems/analytic/AnalyticProblems.kt`, `verification/AnalyticSolutionTest.kt` | Full: the solution and the right-hand side were derived by hand, without the quadrature of the project |
| **P** | Published values of `E_h` and `p_h` (18 tables of the article) | `verification/published-values.tsv`, `PublishedValuesTest.kt` | Full: an external source, tolerance 2 % |
| **C** | Cross-consistency of the schemes with one another | `verification/CrossSchemeConsistencyTest.kt` | Partial: an error common to all schemes is not detected |
| **I** | Mathematical invariants (biorthogonality, partition of unity, idempotence, exactness on the span) | `splines.SplineCoreHealthCheckTest` (the minimal-splines repository) | High: the properties are derived from the theory rather than from the code |
| **F** | Comparison with explicit closed-form formulas (`ReferenceSplines`, `closedFormInternal`) | `splines.SplineCoreHealthCheckTest` (the minimal-splines repository) | High: the formulas are written out independently of the general construction |
| **S** | External cross-check through SciPy/NumPy | `ScipyCrossVerificationTest` + `tools/verify_with_scipy.py`; the task `scipyVerify` and the job `scipy` in CI. It is NOT part of `check` (it requires a venv with Python); in `test` without a venv it is skipped | Full: third-party libraries |
| **O** | The convergence order against an explicit table of expectations | `convergence/ConvergenceOrderTest.kt` (the task `convergenceOrderTest`, 168 combinations; the fast subset carries the tag `fast`) | Internal: the expected orders are matched against the theory and the publication, but the numbers themselves are computed by the project code |
| **N** | A reference Nyström method on Gauss–Legendre quadrature (without splines and functionals) | `verification/ReferenceNystromSolver.kt` + `ReferenceNystromCrossCheckTest` (tag `fast`); the external counterpart is the layers L6a/L6b of `tools/verify_with_scipy.py` | Full: a textbook method [Atkinson 1997]; its own Gauss nodes and Gaussian elimination, not a single import from `src/main` |

#### The limits of the external cross-check (means **S**): what is NOT checked and why

The layer L2 exports five blocks (`M`, `M2`, `g`, `d`, `c_base`) and checks three: `M`, `g`,
`c_base` (through `scipy.linalg.solve` on the same system `(I - M) c = g`).

**The blocks `M2` and `d` are NOT CHECKED externally.** This is stated explicitly rather than
passed over in silence: by the code (`solvers/core/SecondKindSolverCore.kt`)
`M2_{j,i} = chi_j(L(L omega_i))` — that is, a functional of a DOUBLE application of the integral
operator to a basis function, while `d_j = chi_j(L f)`. Both blocks are expressed through the
approximation functionals `chi` (the families `theta`, `xi`, `mu`, `lambda`) and the basis of
minimal splines. SciPy has no counterparts of those constructions; writing them out in the
script would mean carrying the very construction under test into it and obtaining a cross-check
closed upon itself. An imitation of an external cross-check is worse than an honest admission of
its absence: a green check without an independent source creates false confidence.

By what they are checked instead — and what that does NOT give.

* Means **I**: the biorthogonality `chi_j(omega_i) = delta_{ij}`, the idempotence of the
  projector, the exactness on the `span`. This checks the FUNCTIONALS `chi` out of which `M2`
  and `d` are built, but not the assembly of the blocks itself.
* An internal check BY CONSEQUENCES: `M2` and `d` enter exactly two schemes —
  `kulkarniProjector` and `iteratedKulkarni` (`solvers/core/SecondKindSolverCore.kt`,
  `kulkarni()` and `iteratedKulkarni()`). The convergence order of both is checked by
  `convergence.ConvergenceOrderTest` (the full matrix by the task `./gradlew convergenceOrderTest`,
  the fast subset by the tag `fast`) against an explicit table of expected orders: an error in
  `M2` or `d` would shift the observed order of `kulkarni`/`iteratedKulkarni` and would break
  that test. In addition, the numbers of those schemes are recorded by the characterization
  baselines (`baseline-eh.tsv`, `baseline-extra.tsv`) — that is, they are protected against
  change.

**What `M2` and `d` DO NOT have: an external cross-check, neither direct nor indirect.**
The layer L6b checks only the schemes `base` and `sloan`, and `sloan()` uses only `matrixM` and
`vectorG` (`SecondKindSolverCore.kt`, `sloan()`): neither `M2` nor `d` enters it, so L6b would
NOT react to an error in them. The opposite used to be asserted here — the assertion was false
and has been removed. The outcome: `M2` and `d` are covered by the internal check of the
convergence order and by the characterization baselines, but they have no independent external
source.

### The correspondence "method → means of verification"

| Method | Means | Status |
|---|---|---|
| Basis of minimal splines (`B`) | F, I, S | Checked (the invariants I/F are in minimal-splines; S is here) |
| Basis (`H`) | F, I | Checked (the invariants I/F are in minimal-splines) |
| Basis (`T`) | I | Invariants only (in minimal-splines): there is no explicit reference formula in `ReferenceSplines` |
| Gauss–Legendre quadrature | I, S | Checked (exactness on polynomials up to degree 15; the unit tests are in numerical-core; S is here) |
| Functionals `theta` | F, I | Checked (including the published coefficients `{1/14, -2/7, 10/7, -2/7, 1/14}`; the invariants I/F are in minimal-splines) |
| Functionals `xi<0>`, `xi<1>`, `xi<2>` | I | Checked by biorthogonality and idempotence (the invariants I are in minimal-splines) |
| Functionals `mu`, `lambda` | I | Checked by exactness on `span{1, rho, sigma}` (the invariants I are in minimal-splines) |
| Plain collocation (Fredholm) | A, P, C | Checked by three independent means |
| Plain collocation (Volterra) | A, P, C | Checked by three independent means |
| Sloan iteration | A, P, C | Checked |
| Kulkarni scheme (projectors) | A, P, C | Checked |
| Kulkarni scheme (quasi-interpolants) | A, C | Limitation: it diverges when the spectral radius exceeds one |
| Classical Nyström | P, C, I | Checked |
| Combined Nyström (Fredholm) | C | The order 7.0 agrees with the published estimate `O(h^7)` |
| Combined Nyström (Volterra) | C | Adaptation: superconvergence is not observed (the order is about 3.8) |
| Volterra of the first kind (reduction) | P | Checked against `table-v1.tex` |
| Fredholm of the first kind (regularization) | P | Checked against `table-f1.tex` and `table-xi-f1.tex`; six keys of `table-f1.tex` fall under a narrow tolerance, see the note below |
| Uryson (the nonlinear schemes) | C | Weaker than the rest: no analytical solutions have been derived |

#### CLOSED (stage 8.6): Fredholm of the first kind against `table-f1.tex`

HISTORY. Before stage 8.6 the test
`verification.PublishedValuesTest.fredholmFirstKindMatchesPublishedValues` was RED:
4 keys `F.F1.H.theta.*` diverged from the publication by 6.78 %, 4.23 %, 3.47 % and
3.14 % against a tolerance of 2 %. At present the whole `slowTest` is GREEN (23 tests).

What has been established by MEASUREMENT (rather than assumed) — the full report is in
`.tasks/code-review-remediation/stage8/MEASURE-8.6-f1-tolerance.md`:

* F1 is an equation of the FIRST kind with the Wazwaz regularization (`c_L = -1/alpha`,
  `alpha = 1e-10`): `cond_inf(I - M)` = 1.18e10..2.70e10, `‖g‖_inf` = 1.59e10, and in
  the Sloan scheme two terms of the order 1.38e10 cancel down to 2.7 — a loss of about
  9.7 of the 16 significant digits. For that reason `E_h` is only to a limited extent
  reproducible between LU implementations;
* the TWO tables of the article were captured on DIFFERENT arithmetic routes.
  `table-xi-f1.tex` (36 keys `Eh`) is reproduced on `multik` to within 0.033 %
  (median 0.0032 %), whereas `reference` gives up to 11.485 % there; `table-f1.tex` (6 keys)
  is the opposite: `reference` reproduces it with a median of 0.010 %. That is, a single
  "backend of the article" DOES NOT EXIST — this is recorded table by table in the header of
  `published-values.tsv`. The former assertion that "`reference` reproduces it while `multik`
  does not" holds only for the 6 keys of `table-f1.tex`; on the remaining 36 the converse is
  true.

HOW IT WAS CLOSED. The common tolerance `RELATIVE_TOLERANCE` was NOT RELAXED and remains 2 %
for 702 keys out of 708, including all the keys of `table-xi-f1.tex`. For the six keys of
`table-f1.tex` a NARROW tolerance class of 8 % was introduced
(`PublishedValuesTest.LU_PATH_DEPENDENT_TOLERANCE`), derived from the MEASURED
spread between `multik` and `reference` ON THOSE VERY keys (7.267 % + 0.05 %
of the precision of the publication, rounded up), and NOT from the observed discrepancy with
the publication — so the tolerance does not grow along with a degradation. Membership of the
class is determined BY THE SOURCE-FILE FIELD in the baseline rather than by a list of failing
keys; the restricting test `luPathDependentToleranceCoversExactlyTheDeclaredKeys`
compares the SET of relaxed keys with the measured one and therefore catches a silent
transfer of the relaxation to another quantity.

COMPENSATION FOR THE LOSS OF STRICTNESS (without it the relaxation would be inadmissible):
the coverage of F1 in `characterization/baseline-eh.tsv` was extended from 4 to 54 keys with a
tolerance of 1e-9 — a superset of all 42 quantities compared with the publication. It was shown
by measurement that this is not a decoration: when the quadrature is coarsened 8 → 6, the
characterization gate fails on all the keys of F1, whereas in the relaxed class the cross-check
catches it on only one key out of six. See `docs/baseline-changes.md`.

A SIDE FINDING. The `E_h` of the `sloan` scheme for F1 changes by 4.3–39.9 % from THE ORDER OF
SUMMATION ALONE (forward / backward / Kahan-compensated) — more than the discrepancy with the
publication and more than the spread of the backends. The keys `F1.*.sloan` in the baseline are
thereby bound to the order in which the nodes are traversed in `FredholmOperator.applyNodes`;
this is a property of the problem rather than a defect, but a refactoring of that loop will break
the 1e-9 gate without changing the mathematics.

### The result of the external cross-check against SciPy (an actual run)

The cross-check was performed on NumPy 2.5.1 / SciPy 1.18.0 and passed in full: 60 checks
(12 547 numbers compared, 83 skipped as inapplicable), with no discrepancies.
The numbers in the table below are taken from the `build/verification/scipy-report.json` of that
run.

**Status: a permanent check, not a one-off.** The cross-check is arranged as the smoke tests
`verification.ScipyCrossVerificationTest` and is performed by the task `./gradlew scipyVerify`
— in CI on every branch. The Python environment is prepared automatically by the task
`setupScipyVerification` (it creates `.venv-verify` and installs the versions pinned in
`tools/requirements-verify.txt`). The reason for the choice: a one-off cross-check protects only
at the moment it is run, and any subsequent edit of the numerical core could diverge from SciPy
unnoticed.

The behaviour without a Python environment depends on the task, and this is deliberate:

* `./gradlew scipyVerify` — FAILS (the task sets `scipy.required=true`). Here the cross-check is
  requested explicitly and the environment is prepared by the dependent tasks, so a silent skip
  would mean a green build without a single external piece of evidence;
* `./gradlew test` (and any other run) — SKIPS: the absence of a venv is a state of the machine,
  not a discrepancy of the numbers.

In both modes a DISCREPANCY OF THE NUMBERS always causes a failure — only the reaction to an
unprepared environment is softened.

The effectiveness was verified by introducing an artificial defect: a perturbation of the
quadrature weights by `1e-9` broke exactly the layers L1 and L4/L5 (reporting the deviation and
the tolerance), leaving L2, L3, L6a/L6b green; after the reversal all the tests pass again.

| Layer | What is checked | Reference | Checks | Largest deviation | Tolerance |
|---|---|---|---|---|---|
| L1 | Gauss–Legendre nodes and weights, `m = 1..16` | `numpy leggauss` | 2 | `1.1e-16` (nodes) / `1.8e-15` (weights), abs. | `1e-14` |
| L2 | The solution of `(I - M) c = g` and the residual | `scipy.linalg.solve` | 2 | `2.2e-16`, abs. | `1e-10` |
| L3 | The basis `B` and two derivatives, 4 types of grids | `scipy.interpolate.BSpline` | 12 | `1.1e-13` abs. (`omega'`) / `1.5e-14` rel. (`omega''`) | `1e-12` / `1e-13` |
| L4/L5 | The images `K u`, `V u` and the right-hand sides (5 problems) | `scipy.integrate.quad` | 20 | `8.9e-16`, abs. | `1e-10` |
| L6a | The reference Nyström against the exact solution (F2, F2exp) | 80 `leggauss` nodes | 2 | `1.3e-15`, abs. | `1e-12` |
| L6b (a) | `E_h` of the schemes `base`/`sloan` against the thresholds of the reference | the reference Nyström | 12 | `0.50`, rel. | `1.0` |
| L6b (b) | the convergence order of the same schemes is not below the threshold | the reference Nyström | 10 | `0.00` (deficit of p) | `0.0` |

An explanation for L6b: its 22 checks are of TWO DIFFERENT KINDS, and the layer has no single
"tolerance 1.0".

* Kind (a), 12 checks: the quantity `E_h` itself, with a RELATIVE tolerance of 1.0 — coarse by
  the very statement: what is compared is NOT the same numbers but the orders of magnitude of two
  different methods. The actual deviation in all 12 is about 0.50, that is, a twofold margin.
* Kind (b), 10 checks: the OBSERVED ORDER against the threshold of the reference. Here the
  tolerance 0.0 is neither a typo nor strictness down to the last bit: what is compared is NOT a
  deviation but the DEFICIT of the order `max(0, threshold − observed p)`, and "tolerance 0"
  means the requirement "the order is not BELOW the threshold" (an excess is permitted). The
  thresholds themselves are set with a margin against the theory.

The orders observed in that run: `base` — 3.02...3.05, `sloan` — 4.19...4.28.
Note that this is an external check of the order of two schemes only; the systematic check of the
orders of all the schemes is the means **O** (`ConvergenceOrderTest`), where 24 of the 56 rows of
the table of expectations are matched against `published-values.tsv` (the worst discrepancy
is 0.44).

**A discrepancy that was found and its analysis.** The first run produced two discrepancies on
the second derivative of the basis (the grids `quasiUniform` and `graded`): an absolute deviation
of `1.79e-12` against a tolerance of `1e-12`. The analysis showed that there is no defect and
that the criterion of comparison itself was wrong: `omega''` grows as `O(1/h^2)` and on those
grids reaches `|omega''| ~ 200...400`, so the observed deviation corresponds to a RELATIVE
quantity of `1.2e-14` — of the order of 50 ulp. This is the rounding noise of two different ways
of computing (the project inverts a matrix `M_k` of size 3x3, SciPy uses the de Boor recurrence
relations), not a divergence of the methods. The absolute threshold was replaced by a relative
one (`1e-13`); the tolerance was NOT relaxed — the dimensionality of the comparison was corrected.

**What the means of verification fundamentally do NOT cover.**
The trigonometric minimal splines and all four families of approximation functionals are
constructions from the works of the authors, and they have no counterparts in third-party
libraries. For them the external cross-check (S) is inapplicable in principle rather than because
of incompleteness of the work; their verification rests on the mathematical invariants (I), which
are derived from the theory independently of the implementation.

---

## List of references

1. **[Demyanovich 1994]** Demyanovich Yu. K. *Local Approximation on a Manifold and Minimal
   Splines* (in Russian). — St. Petersburg: St. Petersburg University Press, 1994. — 356 p.

2. **[Tikhonov, Arsenin 1977]** Tikhonov A. N., Arsenin V. Ya. *Solutions of Ill-Posed
   Problems.* — Winston & Sons, Washington, 1977.

3. **[Makarov 2012]** Makarov A. A. Construction of Splines of Maximal Smoothness //
   Journal of Mathematical Sciences. — 2012. — Vol. 178, No. 6. — P. 589–604.

4. **[Kosogorov, Makarov 2017]** Kosogorov O., Makarov A. On Some Piecewise Quadratic
   Spline Functions // Numerical Analysis and Its Applications. Lecture Notes in
   Computer Science, Vol. 10187. — 2017. — P. 448–455.

5. **[Kulikov, Makarov 2019a]** Kulikov E. K., Makarov A. A. On Approximation by
   Hyperbolic Splines // Journal of Mathematical Sciences. — 2019. — Vol. 240, No. 6. —
   P. 822–832.

6. **[Kulikov, Makarov 2019b]** Kulikov E. K., Makarov A. A. On de Boor–Fix Type
   Functionals for Minimal Splines // Topics in Classical and Modern Analysis (Applied
   and Numerical Harmonic Analysis). — Springer, 2019. — P. 211–225.

7. **[Kulikov, Makarov 2020]** Kulikov E. K., Makarov A. A. Quadratic minimal splines
   with multiple nodes // Journal of Mathematical Sciences. — 2020. — Vol. 249, No. 2. —
   P. 256–262.

8. **[Kulikov, Makarov 2022]** Kulikov E. K., Makarov A. A. Construction of
   Approximation Functionals for Minimal Splines // Journal of Mathematical Sciences. —
   2022. — Vol. 262, No. 1. — P. 84–98.

9. **[Kulikov, Makarov 2023]** Kulikov E. K., Makarov A. A. A Method for Solving the
   Fredholm Integral Equation of the First Kind // Journal of Mathematical Sciences. —
   2023. — Vol. 272, No. 4. — P. 558–565. DOI: 10.1007/s10958-023-06449-3.

10. **[Kulikov, Makarov 2025]** Kulikov E. K., Makarov A. A. On Projection-Type
    Approximation Functionals for Minimal Splines // Zapiski Nauchnykh Seminarov POMI. —
    2025. — Vol. 542. — P. 126–143.

11. **[Makarov, Kulikov 2026]** Makarov A., Kulikov E. Spline collocation for Volterra
    integral equations with improved accuracy // Numerical Algorithms. — 2026.
    DOI: 10.1007/s11075-026-02388-7.

12. **[Sloan 1976]** Sloan I. H. Improvement by iteration for compact operator
    equations // Mathematics of Computation. — 1976. — Vol. 30, No. 136. — P. 758–764.

13. **[Kulkarni 2003]** Kulkarni R. P. A superconvergence result for solutions of
    compact operator equations // Bulletin of the Australian Mathematical Society. —
    2003. — Vol. 68, No. 3. — P. 517–528. DOI: 10.1017/S0004972700037916.

14. **[Atkinson 1997]** Atkinson K. E. *The Numerical Solution of Integral Equations of
    the Second Kind.* — Cambridge University Press, 1997.

15. **[Brunner 2004]** Brunner H. *Collocation Methods for Volterra Integral and Related
    Functional Differential Equations.* Cambridge Monographs on Applied and
    Computational Mathematics, Vol. 15. — Cambridge University Press, 2004.

16. **[Wazwaz 2011a]** Wazwaz A. The regularization method for Fredholm integral
    equations of the first kind // Computers & Mathematics with Applications. — 2011. —
    Vol. 61, No. 10. — P. 2981–2986.

17. **[Wazwaz 2011b]** Wazwaz A.-M. *Linear and Nonlinear Integral Equations: Methods
    and Applications.* — Springer, Berlin, Heidelberg, 2011. — XVIII+639 p.
    DOI: 10.1007/978-3-642-21449-3.

18. **[Dagnino, Remogna, Sablonnière 2014]** Dagnino C., Remogna S., Sablonnière P. On
    the solution of Fredholm integral equations based on spline quasi-interpolating
    projectors // BIT Numerical Mathematics. — 2014. — Vol. 54, No. 4. — P. 979–1008.

19. **[Dagnino, Dallefrate, Remogna 2019]** Dagnino C., Dallefrate A., Remogna S. Spline
    quasi-interpolating projectors for the solution of nonlinear integral equations //
    Journal of Computational and Applied Mathematics. — 2019. — Vol. 354. — P. 360–372.
    DOI: 10.1016/j.cam.2018.06.054.

20. **[Allouch et al. 2021]** Allouch C., Remogna S., Sbibih D., Tahrichi M.
    Superconvergent methods based on quasi-interpolating operators for Fredholm integral
    equations of the second kind // Applied Mathematics and Computation. — 2021. —
    Vol. 404. — Art. 126227. DOI: 10.1016/j.amc.2021.126227.

21. **[Remogna, Sbibih, Tahrichi 2023]** Remogna S., Sbibih D., Tahrichi M.
    Superconvergent Nyström Method Based on Spline Quasi-Interpolants for Nonlinear
    Urysohn Integral Equations // Mathematics. — 2023. — Vol. 11, No. 14. — Art. 3236.
    DOI: 10.3390/math11143236.

22. **[Engl, Hanke, Neubauer 1996]** Engl H. W., Hanke M., Neubauer A. *Regularization
    of Inverse Problems.* — Kluwer Academic Publishers, Dordrecht, 1996.

23. **[Krasnoselskii 1964]** Krasnoselskii M. A. *Topological Methods in the Theory of
    Nonlinear Integral Equations.* — Pergamon Press, Oxford, 1964.

24. **[Zeidler 1986]** Zeidler E. *Nonlinear Functional Analysis and its Applications I:
    Fixed-Point Theorems.* — Springer, New York, 1986.
