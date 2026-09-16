#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GENERATION OF THE BASELINE published-values.tsv FROM THE .tex TABLES OF THE ARTICLE.

Purpose
----------
The file `src/test/resources/verification/published-values.tsv` contains 708
published numbers against which `verification.PublishedValuesTest` checks the
results of the current code. This is the only external source of truth in the
repository: unlike the characterization baselines, it was not captured by this
implementation but printed in the article.

The numbers are extracted MECHANICALLY from the 18 files `table-*.tex` of the article. A manual
transfer is forbidden: it is itself the source of the "transfer error" for the detection of which
the cross-check exists. Accordingly, the values in the baseline are not to be edited by hand —
a change in the tables of the article is reflected by re-running the present script.

The script generates the file IN FULL, including the comment header: the output coincides
byte for byte with the committed baseline.

Order of the steps
---------------
Regeneration of the baseline (`--out` points by default at the committed file):

    python3 tools/parse_published_values.py --tables-dir <directory of the article tables>

A check without writing — it generates the baseline into a temporary file and compares it with
the committed one:

    python3 tools/parse_published_values.py --tables-dir <directory> --check

Arguments
---------
    --tables-dir <dir>      the directory with the 18 files table-*.tex of the article
                            (mandatory: the sources of the article lie OUTSIDE the
                            repository and at different paths on different machines)
    --out <file>            where to write (by default the committed baseline)
    --check                 do not write but compare with the existing --out
    --stamp                 append to the header a line about the origin of the file;
                            disabled by default, since the line contains the
                            path of a particular machine and would break the byte-for-byte
                            coincidence with the committed baseline

Return codes
-------------
    0   success, or `--check` with no discrepancies
    1   `--check` found a discrepancy with the committed baseline
    2   an input error: the directory is unavailable, there are fewer than 18 tables,
        a table cannot be read or yielded no records at all
    3   an error writing the result

Discrepancies BETWEEN THE TABLES of the article themselves are not return codes: the three
known cases lie at the level of the machine precision (of the order 1e-15), and the script
keeps the value from the first table and marks the discrepancy in the location
column — exactly as recorded in the baseline. A warning is printed.

There are no dependencies: only the standard library, Python >= 3.8.
"""
import argparse
import os
import re
import sys
import tempfile

# The agreement tolerance for a value duplicated in several tables.
# Relative: the numbers are printed with 4 significant digits, so a coincidence
# is required only within the printed precision.
DUPLICATE_TOLERANCE = 5e-3

# Exactly these 18 tables form the published set. The list is part of the structure
# of the article rather than a setting: the appearance or disappearance of a file means that the article
# has changed and the parsing must be reconsidered rather than silently continued.
TABLE_FILES = (
    "table-f1.tex",
    "table-families.tex",
    "table-t1-f2.tex",
    "table-t1-f2exp.tex",
    "table-t1-v2.tex",
    "table-t1-v2exp.tex",
    "table-t1-v2win.tex",
    "table-t2-fredholm.tex",
    "table-t2-volterra.tex",
    "table-t3-fredholm.tex",
    "table-t3-volterra.tex",
    "table-v1.tex",
    "table-xi-f1.tex",
    "table-xi-special.tex",
    "table-xi-t1-f2.tex",
    "table-xi-t1-v2.tex",
    "table-xi-t2-fredholm.tex",
    "table-xi-t2-volterra.tex",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "src", "test", "resources", "verification", "published-values.tsv"
)

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_INPUT_ERROR = 2
EXIT_OUTPUT_ERROR = 3

# The header of the baseline asserts two numbers derived from the tables themselves. If the parsing
# stops producing them, the prose description in the header will diverge from the data below
# it; the check further down makes such a divergence visible.
HEADER_DUPS_OK = 220
HEADER_CONFLICT_KEYS = frozenset((
    "F.F2exp.B.theta.n16.kulkarni.ph",
    "F.F2exp.B.theta.n32.kulkarni.Eh",
    "F.F2exp.B.theta.n64.kulkarni.Eh",
))

# The header of the baseline. It is reproduced literally so that the generated file coincides
# byte for byte with the committed one. The prose part about the LU routes (the section "ON WHICH
# LINEAR-ALGEBRA ROUTE THE TABLES WERE CAPTURED") is not derived from the tables — it is the result
# of a separate measurement, and it is edited here, in the template.
HEADER = """\
# REFERENCE VALUES FROM THE PUBLICATION (an external source of truth).
#
# Extracted MECHANICALLY (by a parser script) from the .tex tables of the article marked
# "Auto-prepared from verified runs. Do not alter numbers":
#   Scientific Agents/papers/new-01/experiments/tables/  (18 files)
# A manual transfer was deliberately not used: it is itself the source of the "transfer error".
#
# Purpose: a cross-check of the results of the current code against the PUBLISHED numbers. Unlike
# characterization/baseline-eh.tsv (captured by this same implementation and recording only
# the invariance of the behaviour), the present file does not depend on the project code.
#
# Format:  key <TAB> value <TAB> source file <TAB> location in the file
#
# Key:  <equation>.<problem>.<basis>.<family>.n<N>.<scheme>.<quantity>
#   equation — F (Fredholm) or V (Volterra);
#   problem  — F2, F2exp, F1, V2, V2exp, V2win, V1;
#   quantity — Eh (the error max|u*-u_h|) or ph (the order log2(E_h/E_{h/2})).
# The naming scheme agrees with characterization/baseline-eh.tsv.
#
# The values are given with 4 significant digits — exactly as many as are printed in the article.
# Hence the cross-check tolerance of 2 %: see PublishedValuesTest.RELATIVE_TOLERANCE.
#
# KNOWN DISCREPANCIES BETWEEN THE TABLES THEMSELVES. Found during the extraction: one and the
# same quantity is printed differently in two tables. This is not a transfer error but different
# runs (table-t2-* are the "Phase-8 runs", table-t3-* the run of commit 403fa1d). All three
# cases lie AT THE LEVEL OF THE MACHINE PRECISION and are therefore excluded from the cross-check (NOISE_FLOOR);
# in the location field they are marked with a divergence marker:
#   F.F2exp.B.theta.n32.kulkarni.Eh : 7.327e-15 (t2) against 7.994e-15 (t3), 9 %;
#   F.F2exp.B.theta.n64.kulkarni.Eh : 5.995e-15 (t2) against 6.439e-15 (t3), 7 %;
#   F.F2exp.B.theta.n16.kulkarni.ph : 6.45 (t2) against 6.32 (t3) — the order is computed
#     from E_h(n=32) taken from the noise zone and is therefore unreliable in itself.
# The value from table-t2-* is kept; the alternative is given in the note.
# The other 220 values duplicated in several tables coincided completely.
#
# ============================================================================
# ON WHICH LINEAR-ALGEBRA ROUTE THE TABLES WERE CAPTURED (measured, stage 8.6)
# ============================================================================
# The article itself does not state the backend. It has been recovered BY FACT — by running all
# 42 F1 keys on both backends (`-Dnumerics.backend=multik|reference`, JDK 21,
# macOS aarch64) and comparing with the published numbers. The result:
#
#   table-xi-f1.tex  (36 F1 keys) — captured on MULTIK/OpenBLAS.
#       multik    against the publication: max. 0.033 %, median 0.0032 %;
#       reference against the publication: max. 11.485 %, median 1.162 %.
#       It is reproduced practically bit for bit; the tolerance is the common 2 %.
#
#   table-f1.tex     (6 keys F.F1.H.theta.*) — captured on a JVM LU ROUTE (ReferenceBackend
#       or an arithmetically close implementation) AND NOT on multik.
#       reference against the publication: max. 4.234 %, median 0.010 %
#                   (4 keys out of 6 coincide to the 4th significant digit);
#       multik    against the publication: max. 6.780 %, median 3.471 %.
#       The tolerance is 8 %, see PublishedValuesTest.LU_PATH_DEPENDENT_TOLERANCE. The common
#       tolerance of 2 % is NOT RELAXED: the relaxation concerns only these 6 keys.
#
# THE REASON why different LU routes give different numbers precisely in F1: this is an
# equation of the FIRST kind with the Wazwaz regularization (alpha = 1e-10, c_L = -1e10).
# Measured: cond_inf(I-M) = 1.18e10..2.70e10, ||g||_inf = 1.59e10, and in the Sloan scheme two
# terms of the order 1.38e10 cancel down to 2.7 (a loss of ~9.7 of the 16 digits).
# The measured spread of multik against reference INSIDE table-f1.tex itself (6 keys):
# max. 7.267 % (the key F.F1.H.theta.n8.sloan) — it is from this number that the tolerance of
# 8 % is derived (= 0.05 % of the precision of the publication + 7.267 %, rounded up).
# For comparison, OVER THE WHOLE F1 group (42 keys) the same spread would be wider:
# max. 11.483 % (the key F.F1.B.xi1.n32.sloan from ANOTHER table), median 0.98 %.
# The NARROW variant was taken: the spread of a foreign table must not relax this one.
# The detailed measurement: .tasks/code-review-remediation/stage8/MEASURE-8.6-f1-tolerance.md.
#
# THE OTHER TABLES (F2/F2exp/V1/V2/V2exp/V2win, 700 keys) DO NOT DEPEND on the LU route:
# there is no scaling by 1/alpha there, cond(I-M) is of the order of units, and all of them pass
# under the common tolerance of 2 % on BOTH backends. Hence a single "backend of the article" does not
# exist and cannot be recorded in a single line — only table by table.
"""

NUM = re.compile(r'\$?(-?\d+\.\d+)\{?\\+times\}?10\^\{(-?\d+)\}\$?')
PLAIN = re.compile(r'^\$?\(?(-?\d+\.\d+)\)?\$?$')

FAMMAP = {'theta': 'theta', 'xi': 'xi1', 'mu': 'mu', 'lambda': 'lambda'}

T2SCHEMES = ['base', 'sloan', 'kulkarni', 'iterKulkarni']
T3SCHEMES = ['base', 'sloan', 'kulkarni', 'nystrom', 'iterNystrom']


class TableError(Exception):
    """A table of the article is unavailable or cannot be parsed."""


def cells(line):
    """Splits a row of a table into cells."""
    line = line.strip()
    line = re.sub(r'\\+\\\s*$', '', line)          # the trailing \\
    return [c.strip() for c in line.split('&')]


def num(tok):
    """The value from a cell of the form $1.014{\\times}10^{-4}$ (or None)."""
    m = NUM.search(tok)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    return None


def order(tok):
    """p_h from the tail of a cell: '(3.02)' or '($-3.31$)'; '---' -> None."""
    m = re.search(r'\(\s*\$?(-?\d+\.\d+)\$?\s*\)', tok)
    return float(m.group(1)) if m else None


def fmt(v):
    """4 significant digits - exactly as many as are printed in the publication."""
    return f"{v:.4g}"


def lines_of(tables_dir, name):
    """The lines of a table file; an unavailable file is an input error."""
    path = os.path.join(tables_dir, name)
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.readlines()
    except OSError as exc:
        raise TableError(f"table {name} cannot be read: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TableError(f"table {name} is not in the UTF-8 encoding: {exc}") from exc


def rows(tables_dir, path):
    """Data rows: they start with the number n."""
    out = []
    for raw in lines_of(tables_dir, path):
        s = raw.strip()
        if re.match(r'^\d+\s*&', s):
            out.append((int(s.split('&')[0].strip()), cells(s), raw))
        elif 'multicolumn' in s:
            out.append((None, None, raw))
    return out


def block_of(raw):
    """Recognizes the heading of a subsection of a table."""
    if 'mathcal{B}' in raw:
        return 'B'
    if 'mathcal{H}' in raw:
        return 'H'
    if 'mathcal{T}' in raw:
        return 'T'
    return None


def extract(tables_dir):
    """Parsing of the 18 tables of the article. Returns a list of (key, value, file, note)."""
    records = []

    def rec(key, val, src, note):
        if val is None:
            return
        records.append((key, val, src, note))

    # ---- table-t1-*: the base scheme, theta, blocks by basis, columns n,h,E,p,C
    for fname, eq, prob in [('table-t1-f2.tex', 'F', 'F2'), ('table-t1-f2exp.tex', 'F', 'F2exp'),
                            ('table-t1-v2.tex', 'V', 'V2'), ('table-t1-v2exp.tex', 'V', 'V2exp'),
                            ('table-t1-v2win.tex', 'V', 'V2win')]:
        sysname = None
        for n, c, raw in rows(tables_dir, fname):
            if n is None:
                b = block_of(raw)
                if b:
                    sysname = b
                continue
            # n & h & E_h & p_h & C_h
            rec(f"{eq}.{prob}.{sysname}.theta.n{n}.base.Eh", num(c[2]), fname, f"basis {sysname}")
            p = PLAIN.match(c[3].replace('$', ''))
            if p:
                rec(f"{eq}.{prob}.{sysname}.theta.n{n}.base.ph", float(p.group(1)), fname, f"basis {sysname}")

    # ---- table-t2-*: basis B, theta, columns base/Sloan/Kulkarni/iter.Kulkarni, blocks by problem
    for fname, eq, probs in [('table-t2-fredholm.tex', 'F', ['F2', 'F2exp']),
                             ('table-t2-volterra.tex', 'V', ['V2', 'V2exp', 'V2win'])]:
        pi, prob = -1, None
        for n, c, raw in rows(tables_dir, fname):
            if n is None:
                if 'textit' in raw and 'K=' in raw:
                    pi += 1
                    prob = probs[pi]
                continue
            for k, sch in enumerate(T2SCHEMES):
                rec(f"{eq}.{prob}.B.theta.n{n}.{sch}.Eh", num(c[k + 1]), fname, prob)
                rec(f"{eq}.{prob}.B.theta.n{n}.{sch}.ph", order(c[k + 1]), fname, prob)

    # ---- table-t3-*: basis B, theta, + Nyström
    for fname, eq, probs in [('table-t3-fredholm.tex', 'F', ['F2', 'F2exp']),
                             ('table-t3-volterra.tex', 'V', ['V2', 'V2exp'])]:
        pi, prob = -1, None
        for n, c, raw in rows(tables_dir, fname):
            if n is None:
                if 'textit' in raw and 'K=' in raw:
                    pi += 1
                    prob = probs[pi]
                continue
            for k, sch in enumerate(T3SCHEMES):
                rec(f"{eq}.{prob}.B.theta.n{n}.{sch}.Eh", num(c[k + 1]), fname, prob)
                rec(f"{eq}.{prob}.B.theta.n{n}.{sch}.ph", order(c[k + 1]), fname, prob)

    # ---- table-xi-t1-*: the base scheme, blocks by xi<r>, columns B/H/T
    for fname, eq, prob in [('table-xi-t1-f2.tex', 'F', 'F2'), ('table-xi-t1-v2.tex', 'V', 'V2')]:
        fam = None
        for n, c, raw in rows(tables_dir, fname):
            if n is None:
                m = re.search(r'xi\^\{?\\+langle(\d)', raw)
                if m:
                    fam = 'xi' + m.group(1)
                continue
            for k, s in enumerate(['B', 'H', 'T']):
                rec(f"{eq}.{prob}.{s}.{fam}.n{n}.base.Eh", num(c[k + 1]), fname, fam)
                rec(f"{eq}.{prob}.{s}.{fam}.n{n}.base.ph", order(c[k + 1]), fname, fam)

    # ---- table-xi-t2-*: basis B, blocks by xi<r>, columns of the schemes
    for fname, eq, prob in [('table-xi-t2-fredholm.tex', 'F', 'F2'), ('table-xi-t2-volterra.tex', 'V', 'V2')]:
        fam = None
        for n, c, raw in rows(tables_dir, fname):
            if n is None:
                m = re.search(r'xi\^\{?\\+langle(\d)', raw)
                if m:
                    fam = 'xi' + m.group(1)
                continue
            for k, sch in enumerate(T2SCHEMES):
                rec(f"{eq}.{prob}.B.{fam}.n{n}.{sch}.Eh", num(c[k + 1]), fname, fam)
                rec(f"{eq}.{prob}.B.{fam}.n{n}.{sch}.ph", order(c[k + 1]), fname, fam)

    # ---- table-families: basis B, the base scheme, blocks by problem, rows by family
    prob = None
    for raw in lines_of(tables_dir, 'table-families.tex'):
        s = raw.strip()
        if 'multicolumn' in s and 'textit' in s:
            m = re.search(r'textit\}?\{(F2exp|F2|V2exp|V2win|V2)\}', s) or re.search(r'\{(F2exp|F2|V2exp|V2win|V2)\}', s)
            if m:
                prob = m.group(1)
            continue
        m = re.match(r'^\$\\+(theta|xi|mu|lambda)\$?\s*&', s)
        if m and prob:
            fam = FAMMAP[m.group(1)]
            eq = 'F' if prob.startswith('F') else 'V'
            c = cells(s)
            for k, n in enumerate([8, 16, 32, 64]):
                rec(f"{eq}.{prob}.B.{fam}.n{n}.base.Eh", num(c[k + 1]), 'table-families.tex', prob)
                rec(f"{eq}.{prob}.B.{fam}.n{n}.base.ph", order(c[k + 1]), 'table-families.tex', prob)

    # ---- table-xi-special: xi1, the base scheme, blocks by problem, columns B/H/T
    prob = None
    for n, c, raw in rows(tables_dir, 'table-xi-special.tex'):
        if n is None:
            m = re.search(r'textit\}?\{(F2exp|V2exp|V2win)', raw)
            if m:
                prob = m.group(1)
            continue
        eq = 'F' if prob.startswith('F') else 'V'
        for k, s in enumerate(['B', 'H', 'T']):
            rec(f"{eq}.{prob}.{s}.xi1.n{n}.base.Eh", num(c[k + 1]), 'table-xi-special.tex', prob)
            rec(f"{eq}.{prob}.{s}.xi1.n{n}.base.ph", order(c[k + 1]), 'table-xi-special.tex', prob)

    # ---- table-f1: F1, basis H, theta, columns n,h,base,Sloan
    for n, c, raw in rows(tables_dir, 'table-f1.tex'):
        if n is None:
            continue
        rec(f"F.F1.H.theta.n{n}.base.Eh", num(c[2]), 'table-f1.tex', 'alpha=1e-10')
        rec(f"F.F1.H.theta.n{n}.sloan.Eh", num(c[3]), 'table-f1.tex', 'alpha=1e-10')

    # ---- table-v1: V1, basis B, theta, columns n,base,Sloan,Kulkarni
    for n, c, raw in rows(tables_dir, 'table-v1.tex'):
        if n is None:
            continue
        for k, sch in enumerate(['base', 'sloan', 'kulkarni']):
            rec(f"V.V1.B.theta.n{n}.{sch}.Eh", num(c[k + 1]), 'table-v1.tex', 'reduction to V2')

    # ---- table-xi-f1: F1, xi1/xi2, bases B/H/T, schemes base/Sloan
    fam, sysname = None, None
    for n, c, raw in rows(tables_dir, 'table-xi-f1.tex'):
        if n is None:
            m = re.search(r'xi\^\{?\\+langle(\d)', raw)
            if m:
                fam = 'xi' + m.group(1)
            b = block_of(raw)
            if b:
                sysname = b
            continue
        for k, sch in enumerate(['base', 'sloan']):
            rec(f"F.F1.{sysname}.{fam}.n{n}.{sch}.Eh", num(c[k + 1]), 'table-xi-f1.tex', f"{fam}/{sysname}")
            rec(f"F.F1.{sysname}.{fam}.n{n}.{sch}.ph", order(c[k + 1]), 'table-xi-f1.tex', f"{fam}/{sysname}")

    silent = [name for name in TABLE_FILES if not any(r[2] == name for r in records)]
    if silent:
        raise TableError(
            "the parsing yielded no records for the tables: " + ", ".join(silent) +
            " — the layout of a table in the article has probably changed"
        )
    return records


def reconcile(records):
    """Folding of the duplicates into a dictionary key -> (value, file, note).

    One and the same number occurs in several tables. Those coinciding within
    DUPLICATE_TOLERANCE are considered consistent; diverging ones keep the value
    from the first table, while the alternative is appended to the note.
    """
    seen = {}
    dups_ok, dups_bad = 0, []
    for key, val, src, note in records:
        if key in seen:
            old = seen[key]
            if abs(old[0] - val) <= DUPLICATE_TOLERANCE * max(abs(old[0]), abs(val)):
                dups_ok += 1
            else:
                dups_bad.append((key, old, (val, src)))
        else:
            seen[key] = (val, src, note)
    for k, o, nv in dups_bad:
        seen[k] = (o[0], o[1], seen[k][2] + f"; DISCREPANCY WITH {nv[1]}: {fmt(nv[0])}")
    return seen, dups_ok, dups_bad


def render(seen, tables_dir, stamp):
    """The text of the baseline in full: the header plus the data rows sorted by key."""
    parts = [HEADER]
    if stamp:
        parts.append(
            f"# Generated by: tools/parse_published_values.py --tables-dir {tables_dir}\n"
        )
    for key in sorted(seen):
        v, src, note = seen[key]
        parts.append(f"{key}\t{fmt(v)}\t{src}\t{note}\n")
    return "".join(parts)


def report(seen, records, dups_ok, dups_bad):
    """A summary of the parsing. Discrepancies between the tables are a warning, not a failure."""
    print(f"records extracted: {len(records)}, unique keys: {len(seen)}")
    print(f"consistent duplicates between the tables: {dups_ok}")
    if dups_bad:
        print("WARNING: discrepancies between the tables of the article "
              "(the value from the first table is kept):")
        for k, o, nv in dups_bad:
            print("   ", k, o, nv)
    else:
        print("there are no contradictions between the tables")
    eh = sum(1 for k in seen if k.endswith('.Eh'))
    print(f"E_h: {eh}, p_h: {len(seen) - eh}")

    # The header describes the parsing in words; on a divergence the description is out of date.
    if dups_ok != HEADER_DUPS_OK:
        print(f"WARNING: the header of the baseline names {HEADER_DUPS_OK} consistent "
              f"duplicates, the parsing gave {dups_ok} — the header text in the script needs updating")
    actual_conflicts = frozenset(k for k, _o, _nv in dups_bad)
    if actual_conflicts != HEADER_CONFLICT_KEYS:
        print("WARNING: the set of discrepancies between the tables differs from "
              "the one listed in the header of the baseline — the header text in the script needs updating")


def data_map(text):
    """The data rows of the baseline text as key -> the rest of the row."""
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        key, _, rest = line.partition('\t')
        result[key] = rest
    return result


def compare(expected_text, actual_text, out_path):
    """Comparison of the generated text with the committed one. 0 — it matched, 1 — it did not."""
    if expected_text == actual_text:
        print(f"comparison with {out_path}: coincides byte for byte")
        return EXIT_OK

    expected = data_map(expected_text)
    actual = data_map(actual_text)
    differing = sorted(
        set(expected) ^ set(actual) |
        {k for k in set(expected) & set(actual) if expected[k] != actual[k]}
    )
    print(f"DISCREPANCY: {out_path} does not coincide with the text generated from the tables of the article.")
    if differing:
        print(f"differing keys: {len(differing)} (the first 20 are shown)")
        for key in differing[:20]:
            print(f"    {key}: in the file {expected.get(key, '<none>')} | "
                  f"generated {actual.get(key, '<none>')}")
    else:
        print("the values coincide, the comment header differs")
    print("The baseline is edited only by re-running this script, never by hand.")
    return EXIT_CHECK_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generation of the baseline published-values.tsv from the .tex tables of the article",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The sources of the article lie outside the repository, so --tables-dir is mandatory\n"
               "and this script is not built into the Gradle build.",
    )
    parser.add_argument("--tables-dir", required=True,
                        help=f"the directory with the {len(TABLE_FILES)} files table-*.tex of the article")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="the result file (by default the committed baseline)")
    parser.add_argument("--check", action="store_true",
                        help="do not write but compare with the existing --out; "
                             "return code 1 on a discrepancy")
    parser.add_argument("--stamp", action="store_true",
                        help="append to the header a line about the origin of the file; "
                             "it breaks the byte-for-byte coincidence with the committed baseline")
    args = parser.parse_args()

    tables_dir = args.tables_dir
    if not os.path.isdir(tables_dir):
        print(f"ERROR: the directory of tables was not found: {tables_dir}")
        return EXIT_INPUT_ERROR
    missing = [name for name in TABLE_FILES
               if not os.path.isfile(os.path.join(tables_dir, name))]
    if missing:
        print(f"ERROR: {len(missing)} of the {len(TABLE_FILES)} tables are missing "
              f"in {tables_dir}: {', '.join(missing)}")
        return EXIT_INPUT_ERROR

    try:
        records = extract(tables_dir)
    except TableError as exc:
        print(f"ERROR: {exc}")
        return EXIT_INPUT_ERROR

    seen, dups_ok, dups_bad = reconcile(records)
    report(seen, records, dups_ok, dups_bad)
    text = render(seen, tables_dir, args.stamp)

    if args.check:
        if not os.path.isfile(args.out):
            print(f"ERROR: there is nothing to compare with, the file was not found: {args.out}")
            return EXIT_INPUT_ERROR
        # Generation into a temporary file: the comparison must be free of side effects,
        # otherwise it would silently repair what it is supposed to detect.
        with tempfile.TemporaryDirectory() as tmp:
            probe = os.path.join(tmp, "published-values.tsv")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write(text)
            with open(probe, encoding="utf-8") as handle:
                actual_text = handle.read()
        with open(args.out, encoding="utf-8") as handle:
            expected_text = handle.read()
        return compare(expected_text, actual_text, args.out)

    directory = os.path.dirname(os.path.abspath(args.out))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        print(f"ERROR: failed to write {args.out}: {exc}")
        return EXIT_OUTPUT_ERROR
    print(f"written: {os.path.abspath(args.out)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
