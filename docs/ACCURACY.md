# Accuracy of the result: what the library guarantees and what it does not

The document answers a single question: **how far the number returned by a solver can be
trusted**. It is needed because a successful return from `solve` does NOT mean that the answer
contains even one correct significant digit, and the two cases cannot be told apart from the
result alone.

## Why this is a separate file

`docs/HPC.md` describes speed, `docs/TESTING.md` the gates, `docs/REFERENCES.md` the origin of
the formulas, `docs/baseline-changes.md` the protocol for editing the baselines. None of them
is about **the limits of attainable accuracy**: this is a separate subject, and appending it to
the end of an existing file would blur the subject of that file.

The general theory — the backward and forward error of a dense solver, the
`LinearAlgebra.solveDiagnosed` diagnostics, the types `ForwardError` and `ConditionEstimate`,
the two routes for estimating `cond` — belongs to the library `numerical-core` and is set out
in full in
[`numerical-core/docs/ACCURACY.md`](https://github.com/EgorkaKulikov/numerical-core/blob/main/docs/ACCURACY.md).
Here those sections are reduced to a summary; the subject of the present document is the
equations of the **first kind**, primarily the problem F1 and its regularization parameter
`alpha`.

---

## 1. Backward and forward error are different quantities

The dense solver `LinearAlgebra.solve` from `numerical-core` is **backward stable**:
the post-check inside `solve` controls only the backward error `‖Ax − b‖` and by construction
catches only outright garbage, whereas the forward error grows as `cond∞(A) · ε`
(at `cond ≈ 2.6e+10` about 6 correct decimal digits, at `1e+16` none at all).
The full account is in [`numerical-core/docs/ACCURACY.md`](https://github.com/EgorkaKulikov/numerical-core/blob/main/docs/ACCURACY.md).

For equations of the first kind the following decision is essential and must not be changed:
the check in `solve` is made **by the residual, not by the conditioning**. The problem F1
(an equation of the first kind with `alpha = 1e-10`) gives `cond ~ 1e10` **as a matter of
course**; rejecting it by the conditioning would be wrong in substance — this is a working
regime of the method, not a malfunction. The justification is in the KDoc of
`LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE`. Hence the design of the diagnostics: it is
**optional** and **rejects nothing**; the decision whether to trust the number is taken by the
caller.

---

## 2. How to measure the forward error

`LinearAlgebra.solveDiagnosed(a, b)` returns the same solution as `solve`, plus a
`ForwardError` with three modes separated by the **constructors of the type**: `Bounded`
(a bound on the forward error exists), `NoFiniteBound` (`σ_min = 0`, no finite `cond`) and
`Unreliable` (the estimate of `cond` is not reliable). The source of the estimate is selected by
`ConditionSource`: `INVERSION` (O(n³), with no requirements) or `SYMMETRIC_SPECTRUM`
(the Jacobi method, for symmetric matrices only, distinguishing "no cond" from "cond is large").
The diagnostics is **not built into** `solve` and costs a separate O(n³). The full account and
the measurements that justified the separation are in [`numerical-core/docs/ACCURACY.md`](https://github.com/EgorkaKulikov/numerical-core/blob/main/docs/ACCURACY.md).

The measurements that concern equations of the first kind: at `alpha <= 1e-12` the inversion
residual `‖A A⁻¹ − I‖∞` of the F1 matrix reached `0.08…760` — the result of the inversion was
meaningless, and the estimate of `cond` through `INVERSION` honestly reported `Unreliable`; on
the discretization matrix of the kernel `1/(1 + t + s)` the Jacobi method gives `σ_min` equal to
**exactly 0**, whereas the estimate through inversion produced non-monotone noise of the order
`1e16…1e19`.

---

## 3. The limit of applicability of the regularized F1 route in terms of `alpha`

`FredholmFirstKindSolver` reduces the equation of the first kind to an equation of the second
kind with `c_L = -1/alpha`. The entries of `M` grow as `alpha^{-1}` and those of `M2` as
`alpha^{-2}`, so a small `alpha` directly degrades the conditioning.

### Measured consequences

Established by independent verification (by a second, independently written implementation):

| `alpha` | discrepancy between 8 legitimate implementation variants |
|---|---|
| `1e-6` | `0.00 %` |
| `1e-8` | `0.01 %` |
| `1e-10` | **`15.19 %`** |

At `alpha = 1e-10` the conditioning of the assembled system amounted to
`cond_inf ≈ 2.636149e+10`, and eight legitimate implementation variants (different orders of
evaluation of the kernel and of the right-hand side, different dense linear solvers) produced
**four different values** of one and the same quantity:
`3.822e-05 / 4.108e-05 / 4.176e-05 / 4.402e-05`.

The a priori estimate `eps · cond / value ≈ 14.6 %` agrees with the observed spread.
**This is not an error in the code but a limitation of accuracy.**

How small a change suffices to shift the answer: writing the exact solution as the division
`cos(ωt)/(1+t)` versus the multiplication `cos(ωt) · p` with `p = 1/(1+t)` differs by exactly
**one rounding** — and changes the result by `7.49 %`.

### Practical conclusion

**Below roughly `alpha = 1e-8` few significant digits remain in the result.**
`DEFAULT_REGULARIZATION = 1e-10` falls **precisely into that range**: the value reproduces the
experimental choice of the authors of the cited work rather than a recommendation of the theory,
and requires a deliberate acceptance. If the problem tolerates a larger `alpha`, the accuracy
will be higher.

The limit is confirmed on the library itself as well
(`FredholmFirstKindConditionTest`): at `alpha = 1e-6` and `1e-8` the estimate of `cond` of the
assembled matrix is reliable, at `alpha = 1e-10` it is no longer so.

---

## 4. Self-check without an external reference

A technique that requires neither an exact solution nor a second implementation: compute the
quantity **in two algebraically equivalent ways** and compare — the discrepancy is a **lower**
estimate of the accuracy actually available, while from above it is bounded by `cond_inf · ε`
(for details see [`numerical-core/docs/ACCURACY.md`](https://github.com/EgorkaKulikov/numerical-core/blob/main/docs/ACCURACY.md)). It is exactly this
technique that revealed the limit in `alpha` for F1: the division `u/(1+t)` versus the
multiplication `u · p` with `p = 1/(1+t)` gives a discrepancy of `15.19 %` at `alpha = 1e-10`
and of `0.00 %` at `1e-6`.

---

## 5. How to see the `cond` of the assembled system

For F1 the condition number is measurable from the solver itself:

```kotlin
val solver = FredholmFirstKindSolver(basis, funcs, op, rhs, rhsDeriv, alpha = 1e-10)
val est = solver.baseCondition()

println(est.valueOrNull()?.let { "cond_inf = $it" }
    ?: "estimate is not reliable: inversion residual ${est.inversionResidual}")
```

`baseCondition()` takes **the same** matrix `I − M` with which the system is solved, and returns
a `ConditionEstimate` (a type from `numerical-core`) — carrying a reliability flag rather than
a bare `Double`. The method is not called by any scheme: its O(n⁴) cost is paid only by whoever
asks for it explicitly. This is the answer to the reviewers' requirement of a reproducibility
protocol: the condition number has become measurable from the library itself.

---

## Summary of the requirements

1. A successful return from `solve` means a small **backward** error — and nothing more.
2. The forward error is not estimated until it is requested through `solveDiagnosed`.
3. A large `cond` is **not** a ground for rejection: for F1 it is a normal regime.
4. An unreliable estimate of `cond` cannot be printed as a number — the type prevents it.
5. Below roughly `alpha = 1e-8` the significant digits of an F1 result cannot be trusted;
   `DEFAULT_REGULARIZATION = 1e-10` lies below that limit.
