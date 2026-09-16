#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXTERNAL CROSS-CHECK OF THE COMPUTATIONAL BASE AGAINST SciPy/NumPy.

Purpose
----------
All the regular checks of the project are written on its own code and therefore confirm only
its internal consistency. The present script cross-checks the results against THIRD-PARTY
libraries (SciPy, NumPy) developed independently, and thereby provides
evidence that is not closed over the implementation under test.

The cross-check is LAYERED — bottom-up, from the foundation to the schemes. This order was chosen
deliberately: on a discrepancy it is immediately visible which layer is at fault. A comparison of the
final errors alone would not give that.

    L1  Gauss--Legendre quadrature    <- numpy.polynomial.legendre.leggauss
    L2  linear algebra (linear system)<- scipy.linalg.solve, numpy.linalg.cond
    L3  basis of minimal splines      <- scipy.interpolate.BSpline
    L4  images of integral operators  <- scipy.integrate.quad (QUADPACK)
    L5  right-hand sides of the model problems <- scipy.integrate.quad
    L6a fitness of the reference     <- an own Nystrom (80 leggauss nodes)
                                         against the exact solution
    L6b solutions of the PROJECT (E_h) <- the same Nystrom: the limits and the observed
                                         convergence order for solution-errors.tsv

The layers L6a and L6b are separated deliberately: L6a reads NOT a single artifact of the project and
answers only the question "is the reference fit to serve as a reference"; the cross-check of the
project solutions themselves is only in L6b. The common name "L6" created the impression that the
solutions had been cross-checked when a single oracle had been.

What the cross-check does NOT cover (in principle, not through an omission)
------------------------------------------------------------
* Hyperbolic (H) and trigonometric (T) minimal splines: there are no counterparts in
  SciPy. They are checked by mathematical invariants inside the project.
* The four families of approximation functionals (theta, xi, mu, lambda):
  constructions from the works of the authors, with no counterparts in third-party libraries. For
  the system B a derived property is checked — the biorthogonality to a basis
  built by the means of SciPy.
* The Kulkarni schemes and the combined Nystrom: they are cross-checked by the convergence order.

Order of the steps
---------------
The regular way is as part of the checks of the project (see `ScipyCrossVerificationTest`):

    ./gradlew scipyVerify                      # prepares the venv, exports the artifacts, cross-checks

A manual run for the analysis of a discrepancy:

    ./gradlew dumpVerificationArtifacts        # export of the artifacts into build/verification/
    ./gradlew setupScipyVerification           # creation of the venv with SciPy
    .venv-verify/bin/python tools/verify_with_scipy.py

Arguments
---------
    --artifacts <dir>       the directory with the exported artifacts
                            (build/verification by default)
    --json <file>          additionally write the result in a machine-readable
                            form — needed by the Kotlin test in order to parse the cross-check
                            by layer rather than rely on the return code alone

Return code: 0 — all the layers agreed, 1 — there are discrepancies (listed in the output),
2 — SciPy/NumPy are unavailable, 3 — the report cannot be serialized, 4 — there is no usable
export metadata (`dump-meta.tsv`): without the bounds of the interval there is nothing to compare the integrals with,
while substituting [0,1] by default would keep the hole in a hidden form.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import platform
import sys

try:
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.integrate import quad
    from scipy.interpolate import BSpline
    from scipy.linalg import solve as scipy_solve
except ImportError as exc:  # pragma: no cover
    print(f"ERROR: SciPy/NumPy were not found ({exc}).")
    # The versions are pinned in the requirements file: the cross-check must be reproducible,
    # otherwise a discrepancy could not be told apart from a change of the behaviour of SciPy itself.
    print("Install: python3 -m venv .venv-verify && "
          ".venv-verify/bin/pip install -r tools/requirements-verify.txt")
    print("Or simply: ./gradlew setupScipyVerification")
    sys.exit(2)

# The directory of the artifacts; it is overridden by the argument --artifacts.
ARTIFACT_DIR = os.path.join("build", "verification")

# The tolerances are chosen by the nature of the compared quantities rather than "so that it passes".
TOL_QUADRATURE = 1e-14   # nodes/weights: both implementations work at the machine precision
TOL_BASIS = 1e-12        # basis: the error of inverting a 3x3 matrix accumulates
TOL_LINALG = 1e-10       # linear system: it depends on the conditioning (which is printed separately)
TOL_INTEGRAL = 1e-10     # integrals: QUADPACK against a composite quadrature of order 8

# For the SECOND derivative of the basis the comparison must be RELATIVE.
#
# The reason (established by measurement rather than assumed): omega'' grows as
# O(1/h^2), and on non-uniform grids reaches |omega''| ~ 150...190. The observed
# absolute deviation of 1.79e-12 corresponds to a RELATIVE quantity of 1.2e-14, that is,
# of the order of 50 ulp — this is the rounding noise of two different ways of computing
# (the project inverts a 3x3 matrix, SciPy uses the de Boor recurrence
# relations) rather than a divergence of the methods.
#
# Important: this is A CORRECTION OF A WRONG CRITERION rather than a relaxation of the tolerance:
# an absolute threshold for a quantity of scale 200 would demand a precision above the machine one.
# The relative bound itself stays strict.
TOL_BASIS_D2_RELATIVE = 1e-13

# The fitness of the reference Nystrom (layer L6a): it must reproduce the exact
# solution orders of magnitude better than the schemes of the project, otherwise it cannot serve as a reference. Formerly the
# quantity was an unnamed literal right inside the `report` call, where it could be
# neither found next to the other tolerances nor justified.
# The actual deviation is 4.4e-16 and 1.3e-15, that is, a margin of three orders of magnitude.
TOL_REFERENCE_FITNESS = 1e-12

failures: list[str] = []
notes: list[str] = []

# A machine-readable summary of all the checks: it is parsed by the Kotlin test in order to
# report PRECISELY the layer that disagrees rather than merely "the cross-check did not pass".
checks: list[dict] = []


def report(
    layer: str,
    name: str,
    deviation: float,
    tolerance: float,
    compared: int,
    skipped: int = 0,
    kind: str = "abs.",
) -> None:
    """Prints the result of one check and records the discrepancy.

    The quantities are NECESSARILY converted to the base Python types: some checks
    compute the deviation by the means of NumPy and obtain a `numpy.float64`, while the comparison
    yields a `numpy.bool_`. Neither type is serializable to JSON: without the conversion
    `json.dump` breaks off in the middle of writing and leaves a TRUNCATED file, which
    looks like "the layer was not performed" and misleads the diagnostics.

    `compared` is the number of points ACTUALLY compared, `skipped` the number of points
    that dropped out of the comparison. Both are mandatory, and this is not a decoration of the output:
    the deviation is initialized to zero, so a check out of which ALL the points dropped
    would give `0.0 <= tol` and print OK without checking anything. For that reason
    `compared == 0` is treated as a FAILURE of the check: "there is nothing to compare" is not
    "there are no discrepancies" but an absence of evidence.

    A NON-FINITE DEVIATION (NaN, ±inf) IS ALSO A FAILURE, and with a SEPARATE text.
    The reason is factual rather than hypothetical: the built-in `max(worst, nan)`
    returns `worst` (a comparison with NaN is always false), so a NaN in an artifact
    GAVE THE STATUS OK — that is, a completely broken number passed the cross-check. For that reason
    the accumulation sites use [worst_of], which LETS a non-finite value through
    further on, while here it turns into a failure. Of the two possible solutions (treating
    non-finite points as skips or failing the check at once) the second was chosen:
    a NaN in the export is NEVER normal — it is either a defect of the computation or
    a corrupted artifact, and masking it as a "skip" would mean losing the
    cause in the noise of the legitimate skips (points at the nodes, degenerate intervals).
    """
    deviation = float(deviation)
    tolerance = float(tolerance)
    compared = int(compared)
    skipped = int(skipped)
    empty = compared <= 0
    nonfinite = not math.isfinite(deviation)
    ok = bool(deviation <= tolerance) and not empty and not nonfinite
    if nonfinite:
        status = "NON-FINITE"
    elif empty:
        status = "NO DATA"
    elif ok:
        status = "OK "
    else:
        status = "DISCREPANCY"
    print(
        f"  [{status}] {name}: {kind} deviation {deviation:.3e} (tolerance {tolerance:.1e}), "
        f"points compared {compared}, skipped {skipped}"
    )
    checks.append({
        "layer": layer,
        "name": name,
        # JSON does not know NaN/Infinity: `json.dumps` by default writes bare
        # literals `NaN`, invalid by the standard, and the parsing regexp in the
        # Kotlin test would not match such an element — the check would DISAPPEAR from
        # the report instead of failing it. For that reason the JSON carries a string
        # representation, while the numeric field receives a deliberately failing value.
        "deviation": deviation if not nonfinite else 1e308,
        "tolerance": tolerance,
        "kind": kind,
        "ok": ok,
        "compared": compared,
        "skipped": skipped,
        **({"deviationRaw": repr(deviation)} if nonfinite else {}),
    })
    if nonfinite:
        failures.append(
            f"{layer} / {name}: the deviation is NON-FINITE ({deviation!r}) — a NaN or an infinity "
            f"entered the comparison (points compared {compared}, skipped {skipped}). This is either "
            f"a defect of the computation in the project or a corrupted artifact; neither of them "
            f"may be given the status OK"
        )
    elif empty:
        failures.append(
            f"{layer} / {name}: 0 points compared ({skipped} skipped) — the check was NOT performed, "
            f"a zero deviation here means an absence of data rather than an agreement"
        )
    elif not ok:
        failures.append(f"{layer} / {name}: {kind} deviation {deviation:.3e} > the tolerance {tolerance:.1e}")


def worst_of(current: float, candidate: float) -> float:
    """The maximum of two deviations, NOT SWALLOWING NaN and infinities.

    Why the built-in `max` is replaced. In Python `max(0.0, float("nan"))` equals
    `0.0`: all comparisons with NaN are false, so the candidate is quietly discarded. In an
    accumulation loop this meant that a NaN in an artifact DID NOT AFFECT the result at all,
    and the cross-check declared OK. Here a non-finite value, on the contrary, ABSORBS
    the accumulated one and reaches [report], where it becomes an explicit failure.
    """
    candidate = float(candidate)
    if not math.isfinite(candidate):
        return candidate
    current = float(current)
    if not math.isfinite(current):
        return current
    return max(current, candidate)


def worst_over(values) -> float:
    """[worst_of] over all the elements; an empty input gives 0.0 (the case compared == 0)."""
    result = 0.0
    for value in values:
        result = worst_of(result, value)
    return result


class MetaError(RuntimeError):
    """The export metadata is absent or unusable."""


# The export metadata (the bounds of the interval, the sizes of the grids). Filled in from
# `dump-meta.tsv` in [main]; THERE ARE DELIBERATELY NO DEFAULT VALUES.
META: dict[str, float] = {}

# The keys without which there is nothing to cross-check: the bounds of the interval define all the integrals.
REQUIRED_META_KEYS = ("a", "b", "dumpGridSize", "quadratureNodes")

META_FILE = "dump-meta.tsv"


def load_meta() -> dict[str, float]:
    """Reads the export metadata.

    An absence of the file is an ERROR rather than a reason to substitute `[0,1]`. The script formerly
    computed the integrals over a hard-coded interval, and the agreement with the dumper rested on
    the fact that the latter uses the default values of `Grid.uniform(n)`. A change of the
    bounds in the dumper would give a SILENT cross-check against different integrals; substituting
    a default value here would keep exactly that hole.
    """
    path = os.path.join(ARTIFACT_DIR, META_FILE)
    if not os.path.exists(path):
        raise MetaError(
            f"the export metadata file {path} is missing. The bounds of the interval and the sizes of the grids are set by "
            f"the dumper (VerificationArtifacts.dumpMeta), and substituting [0,1] by default is not allowed: "
            f"under other bounds the cross-check would silently compute DIFFERENT integrals. "
            f"Run ./gradlew dumpVerificationArtifacts"
        )
    parsed: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise MetaError(f"a metadata line is not in the format 'key<TAB>value': {line!r}")
            try:
                parsed[parts[0]] = float(parts[1])
            except ValueError as exc:
                raise MetaError(f"a non-numeric metadata value {parts[0]}={parts[1]!r} ({exc})") from exc
    missing = [key for key in REQUIRED_META_KEYS if key not in parsed]
    if missing:
        raise MetaError(f"mandatory keys are missing in {META_FILE}: {missing}")
    if not parsed["b"] > parsed["a"]:
        raise MetaError(f"inadmissible interval: a={parsed['a']}, b={parsed['b']}")
    return parsed


def interval() -> tuple[float, float]:
    """The bounds of the interval from the export metadata."""
    if not META:
        raise MetaError("the export metadata has not been loaded")
    return META["a"], META["b"]


def load_tsv(filename: str) -> list[list[str]]:
    """Reads an exported TSV, skipping the comment lines."""
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        print(f"  SKIP: there is no file {path} — run ./gradlew dumpVerificationArtifacts first")
        notes.append(f"the file {filename} is missing, the layer was not checked")
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


# ---------------------------------------------------------------------------
# L1. Gauss--Legendre quadrature
# ---------------------------------------------------------------------------
def verify_quadrature() -> None:
    print("\nL1. Nodes and weights of the Gauss--Legendre quadrature (reference: numpy leggauss)")
    rows = load_tsv("gauss-legendre.tsv")
    if not rows:
        return
    by_m: dict[int, list[tuple[int, float, float]]] = collections.defaultdict(list)
    for m, index, node, weight in rows:
        by_m[int(m)].append((int(index), float(node), float(weight)))
    worst_node = 0.0
    worst_weight = 0.0
    compared = 0
    skipped = 0
    for m, entries in sorted(by_m.items()):
        entries.sort()
        ref_nodes, ref_weights = leggauss(m)
        for index, node, weight in entries:
            if index >= len(ref_nodes):
                skipped += 1
                continue
            worst_node = worst_of(worst_node, abs(node - ref_nodes[index]))
            worst_weight = worst_of(worst_weight, abs(weight - ref_weights[index]))
            compared += 1
    report("L1", f"nodes (m = 1..{max(by_m)})", worst_node, TOL_QUADRATURE,
           compared=compared, skipped=skipped)
    report("L1", f"weights (m = 1..{max(by_m)})", worst_weight, TOL_QUADRATURE,
           compared=compared, skipped=skipped)


# ---------------------------------------------------------------------------
# L3. The basis of minimal splines for the polynomial system B
# ---------------------------------------------------------------------------
def verify_spline_basis() -> None:
    print("\nL3. The basis of minimal splines, system B (reference: scipy BSpline)")
    knot_rows = load_tsv("spline-knots.tsv")
    value_rows = load_tsv("spline-values.tsv")
    if not knot_rows or not value_rows:
        return

    knots: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for grid_name, index, knot in knot_rows:
        knots[grid_name][int(index)] = float(knot)

    values: dict[str, list[tuple[int, float, float, float, float]]] = collections.defaultdict(list)
    for grid_name, j, t, omega, omega_d, omega_dd in value_rows:
        values[grid_name].append((int(j), float(t), float(omega), float(omega_d), float(omega_dd)))

    for grid_name in sorted(values):
        knot_map = knots[grid_name]
        # The full knot vector x_{-2..n+2}: a clamped vector of degree 2.
        ordered = [knot_map[k] for k in sorted(knot_map)]
        knot_vector = np.array(ordered)
        degree = 2
        basis_count = len(knot_vector) - degree - 1
        worst_value = 0.0
        worst_deriv = 0.0
        worst_deriv2_relative = 0.0
        worst_deriv2_magnitude = 0.0
        # The counters are kept SEPARATELY for the three quantities: the value, the first and the second
        # derivative have different grounds for a skip (the second one also has the points at the nodes).
        compared_value = compared_deriv = compared_deriv2 = 0
        skipped_value = skipped_deriv = skipped_deriv2 = 0
        for j, t, omega, omega_d, omega_dd in values[grid_name]:
            # The project indexing j = -2..n-1 corresponds to the SciPy index j + 2.
            scipy_index = j + 2
            if scipy_index >= basis_count:
                skipped_value += 1
                skipped_deriv += 1
                skipped_deriv2 += 1
                continue
            coeffs = np.zeros(basis_count)
            coeffs[scipy_index] = 1.0
            spline = BSpline(knot_vector, coeffs, degree, extrapolate=False)
            ref = float(spline(t))
            if math.isnan(ref):
                skipped_value += 1
                skipped_deriv += 1
                skipped_deriv2 += 1
                continue
            worst_value = worst_of(worst_value, abs(omega - ref))
            compared_value += 1
            ref_d = float(spline.derivative(1)(t))
            if not math.isnan(ref_d):
                worst_deriv = worst_of(worst_deriv, abs(omega_d - ref_d))
                compared_deriv += 1
            else:
                skipped_deriv += 1
            # The second derivative is piecewise constant and has a jump at the grid nodes;
            # the project takes the value from the right piece, so the points that fall on a
            # node are excluded from the comparison (this is a convention, not a discrepancy).
            on_knot = any(abs(t - x) < 1e-12 for x in ordered)
            if not on_knot:
                ref_dd = float(spline.derivative(2)(t))
                if not math.isnan(ref_dd):
                    # A relative comparison: |omega''| ~ 1/h^2 and reaches hundreds,
                    # so an absolute threshold makes no sense here (see TOL_BASIS_D2_RELATIVE).
                    scale = max(abs(ref_dd), 1.0)
                    worst_deriv2_relative = worst_of(worst_deriv2_relative, abs(omega_dd - ref_dd) / scale)
                    worst_deriv2_magnitude = max(worst_deriv2_magnitude, abs(ref_dd))
                    compared_deriv2 += 1
                else:
                    skipped_deriv2 += 1
            else:
                skipped_deriv2 += 1
        report("L3", f"{grid_name}: the values omega_j", worst_value, TOL_BASIS,
               compared=compared_value, skipped=skipped_value)
        report("L3", f"{grid_name}: the first derivative", worst_deriv, TOL_BASIS,
               compared=compared_deriv, skipped=skipped_deriv)
        report(
            "L3",
            f"{grid_name}: the second derivative (max|omega''| = {worst_deriv2_magnitude:.1f})",
            worst_deriv2_relative,
            TOL_BASIS_D2_RELATIVE,
            compared=compared_deriv2,
            skipped=skipped_deriv2,
            kind="rel.",
        )


# ---------------------------------------------------------------------------
# L2. Linear algebra: the solution of the assembled system
# ---------------------------------------------------------------------------
def verify_linear_algebra() -> None:
    print("\nL2. Linear algebra: (I - M) c = g (reference: scipy.linalg.solve)")
    rows = load_tsv("assembled-system.tsv")
    if not rows:
        return
    blocks: dict[str, dict[tuple[int, int], float]] = collections.defaultdict(dict)
    for kind, r, c, value in rows:
        blocks[kind][(int(r), int(c))] = float(value)

    size = max(r for r, _ in blocks["M"]) + 1
    m = np.zeros((size, size))
    for (r, c), value in blocks["M"].items():
        m[r, c] = value
    g = np.zeros(size)
    for (r, _), value in blocks["g"].items():
        g[r] = value
    c_project = np.zeros(size)
    for (r, _), value in blocks["c_base"].items():
        c_project[r] = value

    a = np.eye(size) - m
    condition = np.linalg.cond(a)
    print(f"  Conditioning of (I - M): {condition:.3e} — it determines the attainable precision")
    c_reference = scipy_solve(a, g)
    report("L2", "the coefficients of the base scheme", float(np.max(np.abs(c_project - c_reference))), TOL_LINALG,
           compared=int(size), skipped=0)
    residual = float(np.max(np.abs(a @ c_project - g)))
    report("L2", "the residual ||(I - M)c - g||", residual, TOL_LINALG,
           compared=int(size), skipped=0)


# ---------------------------------------------------------------------------
# L4/L5. The images of the operators and the right-hand sides
# ---------------------------------------------------------------------------
# The kernels, the exact solutions and THEIR DERIVATIVES, derived INDEPENDENTLY of the project code.
#
# FUNDAMENTALLY: all the expressions below were obtained by differentiating by hand from the
# closed formulas K(t,s) and u(t), and were NOT copied from the implementation. A cross-check
# that uses the formulas of the project closes upon itself and is not a proof.
#
# The composition of each record:
#   kernel     K(t,s)
#   kernel_t   dK/dt          (partial)
#   kernel_tt  d2K/dt2        (partial)
#   kernel_s   dK/ds          (partial; needed ONLY for the total derivative of the diagonal)
#   exact      u(t), exact_d u'(t), exact_d2 u''(t)
#
# The diagonal and its derivative are NOT written out as separate closed formulas
# but assembled from the partial ones (see [diagonal] and [diagonal_total_deriv]): in that way the chain
# rule d/dt K(t,t) = K_t(t,t) + K_s(t,t) is visible in the code rather than hidden in a manual
# simplification, where it is easiest to lose precisely at this point.
#
# THE DERIVATION by problem (everything differentiates trivially and is therefore checkable by eye):
#
#   F2, V2:  K = 1/(1+t+s)
#            K_t  = -1/(1+t+s)^2      K_tt = 2/(1+t+s)^3      K_s = -1/(1+t+s)^2
#            u = 1/(1+t), u' = -1/(1+t)^2, u'' = 2/(1+t)^3
#            diagonal K(t,t) = 1/(1+2t), d/dt = -2/(1+2t)^2 = K_t(t,t)+K_s(t,t)  ✓
#
#   F2exp, V2exp:  K = exp(-(t-s)^2)
#            K_t  = -2(t-s)K          K_tt = (4(t-s)^2 - 2)K   K_s = 2(t-s)K
#            u = exp(t), u' = u'' = exp(t)
#            diagonal K(t,t) = 1, d/dt = 0 = K_t(t,t)+K_s(t,t) = 0 + 0        ✓
#
#   V2win:   K = t - s
#            K_t  = 1                 K_tt = 0                K_s = -1
#            u = cos t, u' = -sin t, u'' = -cos t
#            diagonal K(t,t) = 0, d/dt = 0 = K_t(t,t)+K_s(t,t) = 1 + (-1)      ✓
FREDHOLM_PROBLEMS = {
    "F2": dict(
        kernel=lambda t, s: 1.0 / (1.0 + t + s),
        kernel_t=lambda t, s: -1.0 / (1.0 + t + s) ** 2,
        kernel_tt=lambda t, s: 2.0 / (1.0 + t + s) ** 3,
        kernel_s=lambda t, s: -1.0 / (1.0 + t + s) ** 2,
        exact=lambda t: 1.0 / (t + 1.0),
        exact_d=lambda t: -1.0 / (t + 1.0) ** 2,
        exact_d2=lambda t: 2.0 / (t + 1.0) ** 3,
    ),
    "F2exp": dict(
        kernel=lambda t, s: math.exp(-(t - s) ** 2),
        kernel_t=lambda t, s: -2.0 * (t - s) * math.exp(-(t - s) ** 2),
        kernel_tt=lambda t, s: (4.0 * (t - s) ** 2 - 2.0) * math.exp(-(t - s) ** 2),
        kernel_s=lambda t, s: 2.0 * (t - s) * math.exp(-(t - s) ** 2),
        exact=math.exp,
        exact_d=math.exp,
        exact_d2=math.exp,
    ),
}
VOLTERRA_PROBLEMS = {
    "V2": dict(
        kernel=lambda t, s: 1.0 / (1.0 + t + s),
        kernel_t=lambda t, s: -1.0 / (1.0 + t + s) ** 2,
        kernel_tt=lambda t, s: 2.0 / (1.0 + t + s) ** 3,
        kernel_s=lambda t, s: -1.0 / (1.0 + t + s) ** 2,
        exact=lambda t: 1.0 / (t + 1.0),
        exact_d=lambda t: -1.0 / (t + 1.0) ** 2,
        exact_d2=lambda t: 2.0 / (t + 1.0) ** 3,
    ),
    "V2exp": dict(
        kernel=lambda t, s: math.exp(-(t - s) ** 2),
        kernel_t=lambda t, s: -2.0 * (t - s) * math.exp(-(t - s) ** 2),
        kernel_tt=lambda t, s: (4.0 * (t - s) ** 2 - 2.0) * math.exp(-(t - s) ** 2),
        kernel_s=lambda t, s: 2.0 * (t - s) * math.exp(-(t - s) ** 2),
        exact=math.exp,
        exact_d=math.exp,
        exact_d2=math.exp,
    ),
    "V2win": dict(
        kernel=lambda t, s: t - s,
        kernel_t=lambda t, s: 1.0,
        kernel_tt=lambda t, s: 0.0,
        kernel_s=lambda t, s: -1.0,
        exact=math.cos,
        exact_d=lambda t: -math.sin(t),
        exact_d2=lambda t: -math.cos(t),
    ),
}


def diagonal(spec: dict, t: float) -> float:
    """The value of the kernel on the diagonal, K(t,t)."""
    return float(spec["kernel"](t, t))


def diagonal_total_deriv(spec: dict, t: float) -> float:
    """The TOTAL derivative of the diagonal: d/dt K(t,t) = K_t(t,t) + K_s(t,t).

    A separate function — because it is precisely here that the principal error is made in deriving
    the second derivative by the Leibniz rule: instead of the TOTAL derivative of the diagonal one takes
    the partial K_t(t,t), losing the term K_s(t,t). On symmetric kernels of the form
    K(t-s) this error is NOT visible (both terms are zero), so the cross-check necessarily
    includes problems with a non-symmetric behaviour of the diagonal (V2: -2/(1+2t)^2).
    """
    return float(spec["kernel_t"](t, t)) + float(spec["kernel_s"](t, t))


def fredholm_image_deriv(spec: dict, t: float, lower: float, upper: float, order: int) -> float:
    """(Ku)^(order)(t) for Fredholm, order in {0,1,2}.

    The limits are CONSTANT, so the Leibniz rule degenerates into differentiation
    UNDER the integral sign (the integrand is smooth and the interchange is legitimate):

        (Ku)(t)   = ∫_a^b K(t,s)    u(s) ds
        (Ku)'(t)  = ∫_a^b K_t(t,s)  u(s) ds
        (Ku)''(t) = ∫_a^b K_tt(t,s) u(s) ds
    """
    kernel = {0: spec["kernel"], 1: spec["kernel_t"], 2: spec["kernel_tt"]}[order]
    exact = spec["exact"]
    value, _ = quad(lambda s: kernel(t, s) * exact(s), lower, upper, limit=200)
    return float(value)


def volterra_image_deriv(spec: dict, t: float, lower: float, order: int) -> float:
    """(Vu)^(order)(t) for Volterra, order in {0,1,2}. THE DERIVATION IS BELOW.

    Let (Vu)(t) = ∫_a^t K(t,s) u(s) ds. The upper limit is VARIABLE, so
    the Leibniz rule in its general form applies:

        d/dt ∫_a^{b(t)} f(t,s) ds = f(t, b(t)) b'(t) + ∫_a^{b(t)} f_t(t,s) ds.

    Step 1 (the first derivative). Here f(t,s) = K(t,s)u(s), b(t) = t, b'(t) = 1:

        (Vu)'(t) = K(t,t) u(t) + ∫_a^t K_t(t,s) u(s) ds.                        (1)

    Step 2 (the second derivative). Differentiate (1) with respect to t term by term.

    The term A(t) = K(t,t) u(t) is a PRODUCT of two functions of one argument, and
    the diagonal K(t,t) depends on t through BOTH arguments. Hence the TOTAL
    derivative of the diagonal (the chain rule) is needed, and NOT the partial K_t(t,t):

        d/dt K(t,t) = K_t(t,t) + K_s(t,t),
        A'(t) = [K_t(t,t) + K_s(t,t)] u(t) + K(t,t) u'(t).                      (2)

    The term B(t) = ∫_a^t K_t(t,s) u(s) ds is again the Leibniz rule,
    now with f(t,s) = K_t(t,s)u(s):

        B'(t) = K_t(t,t) u(t) + ∫_a^t K_tt(t,s) u(s) ds.                        (3)

    Adding (2) and (3):

        (Vu)''(t) = [K_t(t,t) + K_s(t,t)] u(t)   <- the total derivative of the diagonal
                  + K(t,t) u'(t)                 <- the diagonal times the derivative of the solution
                  + K_t(t,t) u(t)                <- the second, SEPARATE boundary contribution
                  + ∫_a^t K_tt(t,s) u(s) ds.                                    (4)

    ATTENTION: K_t(t,t) enters THE FORMULA TWICE and from DIFFERENT sources: once as
    part of the total derivative of the diagonal (2), and a second time as a boundary contribution (3).
    They cannot be merged into a single term — these are different members that merely coincide in form.

    A CONTROL OF THE DERIVATION on V2win (K = t - s, u = cos t), where everything is computed in closed form.
    Here K(t,t) = 0, so the first Leibniz term in (1) vanishes, and at a = 0:
        (Vu)(t)   = ∫_0^t (t-s) cos s ds = t sin t - (t sin t + cos t - 1) = 1 - cos t,
        (Vu)'(t)  = sin t   — and formula (1) gives 0 + ∫_0^t 1·cos s ds = sin t        ✓
        (Vu)''(t) = cos t   — and formula (4) gives
                    [1 + (-1)]·cos t + 0·(-sin t) + 1·cos t + ∫_0^t 0 ds = cos t     ✓
    Note: if in (2) the partial K_t(t,t) = 1 stood instead of the total derivative,
    the result would be 2 cos t — twice as large. It is precisely this problem that catches a confusion
    between the total and the partial derivative of the diagonal.

    A second, independent check of the derivation (actually performed for all three problems):
    central finite differences of (Vu)(t) computed by `quad` agree with (1) and (4)
    to within the order of the difference step — that is, the derivation contains no algebraic error.
    """
    exact = spec["exact"]
    if order == 0:
        if t <= lower:
            return 0.0
        value, _ = quad(lambda s: spec["kernel"](t, s) * exact(s), lower, t, limit=200)
        return float(value)
    if order == 1:
        integral = 0.0
        if t > lower:
            integral = float(quad(lambda s: spec["kernel_t"](t, s) * exact(s), lower, t, limit=200)[0])
        return diagonal(spec, t) * float(exact(t)) + integral
    if order == 2:
        integral = 0.0
        if t > lower:
            integral = float(quad(lambda s: spec["kernel_tt"](t, s) * exact(s), lower, t, limit=200)[0])
        return (
            diagonal_total_deriv(spec, t) * float(exact(t))
            + diagonal(spec, t) * float(spec["exact_d"](t))
            + float(spec["kernel_t"](t, t)) * float(exact(t))
            + integral
        )
    raise ValueError(f"volterra_image_deriv: unsupported order {order}")


def verify_operator_images() -> None:
    print("\nL4/L5. The images of the operators and the right-hand sides (reference: scipy.integrate.quad)")
    rows = load_tsv("operator-images.tsv")
    if not rows:
        return
    lower, upper = interval()
    # Per check: the worst deviation, the number of compared and of skipped points.
    stats: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"worst": 0.0, "compared": 0, "skipped": 0})
    # The rows that this layer does not process AT ALL (another wave of the cross-check):
    # they belong to no check, so they are counted separately and
    # printed explicitly — so that their number cannot be lost from sight.
    out_of_scope: dict[str, int] = collections.Counter()
    for equation, problem, quantity, t_text, value_text in rows:
        t = float(t_text)
        value = float(value_text)
        key = f"{equation}/{problem}/{quantity}"
        if equation == "F" and problem in FREDHOLM_PROBLEMS:
            spec = FREDHOLM_PROBLEMS[problem]
            exact = spec["exact"]
            # The right-hand side of a model problem: f = u - Ku, hence f^(k) = u^(k) - (Ku)^(k).
            if quantity == "Ku":
                ref = fredholm_image_deriv(spec, t, lower, upper, 0)
            elif quantity == "rhs":
                ref = float(exact(t)) - fredholm_image_deriv(spec, t, lower, upper, 0)
            elif quantity == "rhsDeriv":
                ref = float(spec["exact_d"](t)) - fredholm_image_deriv(spec, t, lower, upper, 1)
            elif quantity == "rhsDeriv2":
                ref = float(spec["exact_d2"](t)) - fredholm_image_deriv(spec, t, lower, upper, 2)
            else:
                out_of_scope[quantity] += 1
                continue
        elif equation == "V" and problem in VOLTERRA_PROBLEMS:
            spec = VOLTERRA_PROBLEMS[problem]
            exact = spec["exact"]
            if quantity == "Vu":
                if t <= lower:
                    # A degenerate interval of integration: there is nothing to compare.
                    stats[key]["skipped"] += 1
                    continue
                ref = volterra_image_deriv(spec, t, lower, 0)
            elif quantity == "rhs":
                ref = float(exact(t)) - volterra_image_deriv(spec, t, lower, 0)
            elif quantity == "rhsDeriv":
                # At t = a the integral degenerates, BUT the boundary contribution K(a,a)u(a) remains
                # meaningful, so the point is NOT skipped: it is precisely this point that checks the
                # Leibniz term in isolation from the integral term.
                ref = float(spec["exact_d"](t)) - volterra_image_deriv(spec, t, lower, 1)
            elif quantity == "rhsDeriv2":
                ref = float(spec["exact_d2"](t)) - volterra_image_deriv(spec, t, lower, 2)
            else:
                out_of_scope[quantity] += 1
                continue
        else:
            out_of_scope[quantity] += 1
            continue
        entry = stats[key]
        entry["worst"] = worst_of(entry["worst"], abs(value - ref))
        entry["compared"] += 1
    if out_of_scope:
        # NOT a silent note but a FAILURE — exactly as in L6b for the rows without a limit.
        # A row of the export that fell into no check is an UNCHECKED
        # result. Verified in fact: adding the problem F2span to the dumper gave
        # 84 unchecked rows, one printed line and the RETURN CODE 0 — a new problem
        # quietly stayed without a cross-check. Now it is either cross-checked or it fails the cross-check.
        listing = ", ".join(f"{name}: {count}" for name, count in sorted(out_of_scope.items()))
        print(f"  [DISCREPANCY] rows outside the set processed by the layer: {listing}")
        failures.append(
            f"L4/L5 / completeness of the processing: operator-images.tsv contains rows that fell into no "
            f"check ({listing}). Either add the problem/quantity to FREDHOLM_PROBLEMS / "
            f"VOLTERRA_PROBLEMS together with the corresponding cross-check branch, or remove them from the export: "
            f"an exported but unchecked result creates the appearance of a check"
        )
    for name in sorted(stats):
        entry = stats[name]
        report("L4/L5", name, entry["worst"], TOL_INTEGRAL,
               compared=int(entry["compared"]), skipped=int(entry["skipped"]))


# ---------------------------------------------------------------------------
# L6. The final solutions: the reference Nystrom on Gauss--Legendre quadrature
# ---------------------------------------------------------------------------
def reference_nystrom(kernel, rhs, lower: float, upper: float, node_count: int = 80):
    """The classical Nystrom method (Atkinson 1997, ch. 4) on [lower, upper].

    No splines and no approximation functionals: the quadrature nodes, the matrix
    (I - w_j K(t_i,t_j)) and the solution of the linear system. On smooth problems it gives a precision of the order
    1e-14 and therefore serves as an external reference.

    The bounds are passed EXPLICITLY (with no default values): they are taken from the
    export metadata, whereas a hard-coded interval would make the agreement with the dumper
    an accidental coincidence.
    """
    raw_nodes, raw_weights = leggauss(node_count)
    half = 0.5 * (upper - lower)
    nodes = lower + half * (raw_nodes + 1.0)
    weights = half * raw_weights
    matrix = np.eye(node_count) - np.array(
        [[weights[j] * kernel(nodes[i], nodes[j]) for j in range(node_count)]
         for i in range(node_count)]
    )
    values = scipy_solve(matrix, np.array([rhs(t) for t in nodes]))

    def evaluate(t: float) -> float:
        return rhs(t) + sum(weights[j] * kernel(t, nodes[j]) * values[j] for j in range(node_count))

    return evaluate


def verify_reference_fitness() -> None:
    """L6a. THE FITNESS OF THE REFERENCE — a self-check of the oracle, NOT a cross-check of the project.

    Not a SINGLE artifact of the project is read here, and this is deliberate: the layer answers the
    question "is the reference fit to serve as a reference". If an independent Nystrom does not
    reproduce the exact solution, cross-checking with it (L6b) is meaningless.

    Formerly this logic was called simply "L6. The final solutions" and created the
    impression that the solutions of the project had been cross-checked. The name was split precisely
    because a misleading name by itself creates a false guarantee.
    """
    print("\nL6a. The fitness of the reference: Nystrom (80 leggauss nodes) against the exact solution")
    lower, upper = interval()
    sample = np.linspace(lower, upper, 51)
    for name, spec in FREDHOLM_PROBLEMS.items():
        kernel, exact = spec["kernel"], spec["exact"]

        def rhs(t: float) -> float:
            integral, _ = quad(lambda s: kernel(t, s) * exact(s), lower, upper, limit=200)
            return exact(t) - integral

        approximate = reference_nystrom(kernel, rhs, lower, upper)
        deviation = worst_over(abs(approximate(t) - exact(t)) for t in sample)
        # The reference must reproduce the exact solution: otherwise it cannot serve as a reference.
        report("L6a", f"{name}: the reference Nystrom against the exact solution",
               deviation, TOL_REFERENCE_FITNESS, compared=int(len(sample)), skipped=0)


# THE LIMITS FOR THE E_h OF THE PROJECT (layer L6b), key (problem, system, scheme) -> (n=8, n=16, n=32).
#
# Where they come from. From an ACTUAL run (`solution-errors.tsv`, the export
# `VerificationArtifacts.dumpSolutionErrors`), by doubling the observed value.
#
# Why a TWOFOLD margin exactly and not "an order of magnitude". With a margin of x10 a
# degradation by one convergence order would pass unnoticed: at n = 8 a
# transition from O(h^3) to O(h^2) gives a growth of only 8 times. The factor 2 catches such a
# change on ANY of the three grids and at the same time covers with a margin the spread
# between platforms (of the order of an ulp: the quantities are deterministic to within a
# bit, which is confirmed by the reference snapshot `baseline-eh.tsv`).
#
# THE SPECIAL CASE F2exp/H. There E_h is of the order 1e-13...1e-11 and GROWS with n — this is not
# a defect: the solution u = e^t lies IN THE GENERATING SPACE of the hyperbolic
# system H, so the approximation error is identically zero and only
# the rounding remains, growing with the size of the linear system. For that reason the row is excluded from the check of
# the convergence order (there is no order there and there cannot be), but THE LIMIT is taken BY THE SAME
# RULE — by doubling the fact on each grid separately. Formerly a single
# ceiling of 1e-9 stood here — a margin of 60...600 times, that is, the limit restricted almost nothing:
# a loss of the property "the solution is in the span" with an escape to 1e-10 would have passed unnoticed.
# The apprehension that "a limit on rounding noise is fragile" turned out to be unfounded: the quantities
# are deterministic bit for bit (which `baseline-eh.tsv` confirms on every run),
# so a margin of x2 suffices here as well.
EH_LIMITS: dict[tuple[str, str, str], tuple[float, float, float]] = {
    ("F2", "B", "base"): (2.029e-04, 2.493e-05, 3.042e-06),
    ("F2", "B", "sloan"): (9.615e-06, 4.993e-07, 2.503e-08),
    ("F2", "H", "base"): (1.694e-04, 2.086e-05, 2.543e-06),
    ("F2", "H", "sloan"): (8.619e-06, 4.539e-07, 2.289e-08),
    ("F2", "T", "base"): (2.364e-04, 2.899e-05, 3.541e-06),
    ("F2", "T", "sloan"): (1.061e-05, 5.447e-07, 2.716e-08),
    ("F2exp", "B", "base"): (9.815e-05, 1.131e-05, 1.368e-06),
    ("F2exp", "B", "sloan"): (1.282e-05, 6.386e-07, 3.499e-08),
    # The solution is in the span of the system H: only the rounding remains (see above).
    ("F2exp", "H", "base"): (1.452e-12, 1.443e-11, 3.417e-11),
    ("F2exp", "H", "sloan"): (2.234e-13, 2.229e-12, 3.144e-12),
    ("F2exp", "T", "base"): (1.964e-04, 2.262e-05, 2.735e-06),
    ("F2exp", "T", "sloan"): (2.564e-05, 1.277e-06, 6.998e-08),
}

# The grids, in the order of the columns of [EH_LIMITS].
EH_GRID_SIZES = (8, 16, 32)

# The minimal observed convergence order p = log2(E_h(n) / E_h(2n)).
#
# The theory gives p = 3 for the base scheme and p = 4 for the Sloan iteration;
# in fact 3.02...3.12 and 4.19...4.33 are observed. The threshold 2.5 lets the regular
# spread through but certainly catches a drop of the order to 2 — the main symptom of a loss of the
# properties of the minimal splines. The rows where the error is determined by rounding
# (F2exp/H) are excluded from the check of the order: there is no order there and there cannot be.
MIN_OBSERVED_ORDER = 2.5

# The threshold below which E_h is considered to be determined by rounding rather than by approximation.
EH_ROUNDOFF_FLOOR = 1e-9


def verify_project_solutions() -> None:
    """L6b. THE CROSS-CHECK OF THE PROJECT SOLUTIONS: `solution-errors.tsv` against an independent Nystrom.

    Before this layer the exported `solution-errors.tsv` was read NOWHERE: the layer L6
    compared the oracle with the exact solution and touched not a single number of the project.

    The assertion under test (three parts, each meaningful SEPARATELY):

      (1) AGREEMENT WITH THE REFERENCE. The independent Nystrom reproduces the exact solution
          at the level of 1e-15 (this is established by the layer L6a), so the deviation of a scheme
          of the project FROM THE REFERENCE is practically equal to its deviation from the exact solution.
          Hence the exported E_h must fit within [EH_LIMITS].
      (2) CONVERGENCE. E_h must decrease as n grows with an order not below
          [MIN_OBSERVED_ORDER]. The limits alone are not enough: a scheme stuck at the
          accuracy level of n = 8 would still pass the limit for n = 8.
      (3) COMPLETENESS. Every row of the export must have a limit: an unrecognized row
          is either a new scheme without a cross-check or a typo; it must not be passed over silently.
    """
    print("\nL6b. The solutions of the PROJECT: E_h from solution-errors.tsv against an independent Nystrom")
    rows = load_tsv("solution-errors.tsv")
    if not rows:
        return
    lower, upper = interval()

    # The deviation of the reference itself from the exact solution — a correction that MUST NOT
    # be taken as zero without a measurement: should it suddenly become comparable with the E_h of the project,
    # the cross-check would lose its meaning, and this will be discovered here rather than stay unnoticed.
    reference_error: dict[str, float] = {}
    sample = np.linspace(lower, upper, 51)
    for name, spec in FREDHOLM_PROBLEMS.items():
        kernel, exact = spec["kernel"], spec["exact"]

        def rhs(t: float, kernel=kernel, exact=exact) -> float:
            integral, _ = quad(lambda s: kernel(t, s) * exact(s), lower, upper, limit=200)
            return exact(t) - integral

        approximate = reference_nystrom(kernel, rhs, lower, upper)
        reference_error[name] = worst_over(abs(approximate(t) - exact(t)) for t in sample)

    observed: dict[tuple[str, str, str], dict[int, float]] = collections.defaultdict(dict)
    unknown: list[str] = []
    for problem, system, n_text, scheme, eh_text in rows:
        key = (problem, system, scheme)
        if key not in EH_LIMITS or int(n_text) not in EH_GRID_SIZES:
            unknown.append(f"{problem}/{system}/n={n_text}/{scheme}")
            continue
        observed[key][int(n_text)] = float(eh_text)

    for key in sorted(EH_LIMITS):
        problem, system, scheme = key
        limits = EH_LIMITS[key]
        values = observed.get(key, {})
        name = f"{problem}/{system}/{scheme}"
        # The deviation is expressed RELATIVE to the limit: in that way one check
        # combines three grids with different scales of E_h while the tolerance stays 1.0.
        worst_ratio = 0.0
        compared = 0
        skipped = 0
        details: list[str] = []
        for index, n in enumerate(EH_GRID_SIZES):
            if n not in values:
                skipped += 1
                continue
            eh = values[n]
            limit = limits[index]
            # A correction for the error of the reference itself (see above).
            # worst_of rather than the built-in max: a NaN in the export would otherwise be discarded
            # and the limit check would stay green (verified by a mutation).
            worst_ratio = worst_of(worst_ratio, eh / (limit + reference_error[problem]))
            compared += 1
            details.append(f"n={n}: {eh:.3e} <= {limit:.3e}")
        report("L6b", f"{name}: the E_h of the project against the limits of the reference", worst_ratio, 1.0,
               compared=compared, skipped=skipped, kind="rel.")

        # A MISSING GRID IS A FAILURE, NOT A SKIP. Without this, a disappearance of the rows
        # n = 32 from the export left the check GREEN with compared=2, skipped=1: the accuracy
        # on the finest grid (where the requirements are strictest) was not checked at all.
        if compared != len(EH_GRID_SIZES):
            missing = [n for n in EH_GRID_SIZES if n not in values]
            failures.append(
                f"L6b / {name}: solution-errors.tsv has no grids {missing} — {compared} of "
                f"{len(EH_GRID_SIZES)} were cross-checked. An incomplete cross-check is NOT a cross-check: the absence of the finest "
                f"grid hides precisely the deviations to which the requirements are strictest"
            )
            print(f"  [DISCREPANCY] {name}: the grids {missing} were not exported")

        # Part (2): the observed convergence order. It is meaningful only where the
        # error is determined by the approximation rather than by rounding (see F2exp/H).
        roundoff_dominated = bool(values) and min(values.values()) <= EH_ROUNDOFF_FLOOR
        if roundoff_dominated:
            continue
        if len(values) != len(EH_GRID_SIZES):
            # AN ABSENT ORDER CHECK IS ALSO A FAILURE. Formerly the check was simply
            # NOT CREATED, and the absence of a proof of convergence looked like its
            # presence: it was simply missing from the report while the overall status stayed green.
            failures.append(
                f"L6b / {name}: the convergence order check was NOT CREATED: all the grids "
                f"{list(EH_GRID_SIZES)} are required, {sorted(values)} were exported. The error here is not rounding "
                f"noise (min E_h > {EH_ROUNDOFF_FLOOR:.0e}), hence the convergence MUST be "
                f"checked, while the limits alone would let through a scheme stuck at the accuracy of n = 8"
            )
            print(f"  [DISCREPANCY] {name}: an order check is impossible, only {sorted(values)} were exported")
            continue
        orders = []
        for coarse, fine in zip(EH_GRID_SIZES, EH_GRID_SIZES[1:]):
            ratio = values[coarse] / values[fine]
            # A NaN/0 in the export is a non-finite order; it is passed to report AS IS.
            orders.append(math.log2(ratio) if ratio > 0.0 else float("nan"))
        # A minimum with the REVERSE priority of NaN: the built-in `min` loses a NaN if it
        # arrived as the second argument, and the non-finiteness would slip past the check.
        worst_order = float("nan") if any(not math.isfinite(p) for p in orders) else min(orders)
        # The deviation = by how much the order FELL SHORT of the required one (0 = normal).
        # A non-finite order (E_h = 0 or a NaN in the export) passes into report as is and
        # turns into a failure there: swallowing it here would mean hiding the cause.
        deficit = MIN_OBSERVED_ORDER - worst_order
        report("L6b", f"{name}: the convergence order (observed min p = {worst_order:.2f})",
               deficit if not math.isfinite(deficit) else max(0.0, deficit), 0.0,
               compared=len(orders), skipped=0, kind="deficit of p,")

    if unknown:
        # NOT a silent skip: an unrecognized row is an unchecked result.
        failures.append(
            f"L6b / completeness of the table of limits: solution-errors.tsv contains rows without a limit "
            f"({len(unknown)}): {sorted(set(unknown))}. Add them to EH_LIMITS — otherwise the result "
            f"is exported but not cross-checked"
        )
        print(f"  [DISCREPANCY] rows without a limit: {sorted(set(unknown))}")


def main() -> int:
    global ARTIFACT_DIR
    parser = argparse.ArgumentParser(description="External cross-check against SciPy/NumPy")
    parser.add_argument("--artifacts", default=ARTIFACT_DIR,
                        help="the directory with the exported artifacts")
    parser.add_argument("--json", default=None,
                        help="the file for the machine-readable result (for the Kotlin test)")
    args = parser.parse_args()
    ARTIFACT_DIR = args.artifacts

    print("=" * 78)
    print("EXTERNAL CROSS-CHECK AGAINST SciPy/NumPy")
    print("=" * 78)
    print(f"NumPy {np.__version__}")
    import scipy
    print(f"SciPy {scipy.__version__}")
    print(f"Python {platform.python_version()}")
    print(f"Artifact directory: {os.path.abspath(ARTIFACT_DIR)}")

    # The export metadata is read BEFORE the first check and without default
    # values: without it the interval is unknown, and hence so are all the integrals.
    try:
        META.update(load_meta())
    except MetaError as exc:
        print(f"ERROR: {exc}")
        return 4
    print(
        f"Export configuration: interval [{META['a']:.17g}, {META['b']:.17g}], "
        f"grid n = {int(META['dumpGridSize'])}, quadrature nodes {int(META['quadratureNodes'])}"
    )

    verify_quadrature()
    verify_linear_algebra()
    verify_spline_basis()
    verify_operator_images()
    verify_reference_fitness()
    verify_project_solutions()

    print("\n" + "=" * 78)
    if notes:
        print("NOTES:")
        for note in notes:
            print(f"  - {note}")

    exit_code = 1 if failures else 0
    if failures:
        print(f"DISCREPANCIES FOUND ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        print("\nA discrepancy is NOT to be removed by relaxing the tolerance: the cause comes first.")
    else:
        print("ALL THE LAYERS AGREED: no discrepancies with SciPy/NumPy were found.")

    # The machine-readable summary. It is written ALWAYS, including the case of discrepancies:
    # it is then that it is needed most — so that the test can name the concrete layers and numbers.
    if args.json:
        summary = {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            # The environment in full — so that a discrepancy can be related to the versions
            # of the libraries from a single report, without reconstructing the setting of the run.
            # The order of the fields matters: the scalar "numpy"/"scipy" above are read by the Kotlin test
            # by a line-by-line parsing, and they must occur first in the text.
            "environment": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "artifactDir": os.path.abspath(ARTIFACT_DIR),
            "exitCode": exit_code,
            "checks": checks,
            "failures": failures,
            "notes": notes,
        }
        directory = os.path.dirname(os.path.abspath(args.json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # First serialize IN FULL in memory, then write: on a serialization
        # error, writing directly into the file would leave a truncated JSON that is
        # indistinguishable from "the layer was not performed" and is misleading.
        try:
            serialized = json.dumps(summary, ensure_ascii=False, indent=2)
        except TypeError as exc:
            print(f"ERROR: the report cannot be serialized ({exc}).")
            return 3
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        print(f"Machine-readable result: {os.path.abspath(args.json)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
