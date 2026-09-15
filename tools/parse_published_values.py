#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ПОРОЖДЕНИЕ ЭТАЛОНА published-values.tsv ИЗ .tex-ТАБЛИЦ СТАТЬИ.

Назначение
----------
Файл `src/test/resources/verification/published-values.tsv` содержит 708
опубликованных чисел, с которыми `verification.PublishedValuesTest` сверяет
результаты текущего кода. Это единственный внешний источник истины в
репозитории: в отличие от характеризационных эталонов, он снят не этой
реализацией, а напечатан в статье.

Числа извлекаются МЕХАНИЧЕСКИ из 18 файлов `table-*.tex` статьи. Ручной перенос
запрещён: он и есть источник «ошибки переноса», ради обнаружения которой сверка
и существует. Соответственно, значения в эталоне правке руками не подлежат —
изменение таблиц статьи отражается повторным запуском настоящего скрипта.

Скрипт порождает файл ЦЕЛИКОМ, включая шапку комментариев: вывод побайтово
совпадает с закоммиченным эталоном.

Порядок запуска
---------------
Пересоздание эталона (`--out` по умолчанию указывает на закоммиченный файл):

    python3 tools/parse_published_values.py --tables-dir <каталог таблиц статьи>

Проверка без записи — порождает эталон во временный файл и сверяет с
закоммиченным:

    python3 tools/parse_published_values.py --tables-dir <каталог> --check

Аргументы
---------
    --tables-dir <каталог>  каталог с 18 файлами table-*.tex статьи
                            (обязателен: исходники статьи лежат ВНЕ
                            репозитория и на разных машинах по разным путям)
    --out <файл>            куда писать (по умолчанию — закоммиченный эталон)
    --check                 не писать, а сверить с существующим --out
    --stamp                 дописать в шапку строку о происхождении файла;
                            по умолчанию выключено, поскольку строка содержит
                            путь конкретной машины и нарушила бы побайтовое
                            совпадение с закоммиченным эталоном

Коды возврата
-------------
    0   успех либо `--check` без расхождений
    1   `--check` обнаружил расхождение с закоммиченным эталоном
    2   ошибка входных данных: каталог недоступен, таблиц меньше 18,
        таблица не читается или не дала ни одной записи
    3   ошибка записи результата

Расхождения МЕЖДУ САМИМИ таблицами статьи кодом возврата не являются: три
известных случая лежат на уровне машинной точности (порядка 1e-15), скрипт
оставляет значение из первой таблицы и помечает расхождение в колонке
расположения — ровно так, как записано в эталоне. Печатается предупреждение.

Зависимостей нет: только стандартная библиотека, Python >= 3.8.
"""
import argparse
import os
import re
import sys
import tempfile

# Допуск согласования значения, продублированного в нескольких таблицах.
# Относительный: числа напечатаны с 4 значащими цифрами, поэтому совпадение
# требуется лишь в пределах напечатанной точности.
DUPLICATE_TOLERANCE = 5e-3

# Ровно эти 18 таблиц образуют опубликованный набор. Список — часть схемы
# статьи, а не настройка: появление или пропажа файла означает, что статья
# изменилась и разбор надо пересматривать, а не молча продолжать.
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

# Шапка эталона утверждает два числа, выводимых из самих таблиц. Если разбор
# перестанет их давать, прозаическое описание в шапке разойдётся с данными под
# ним; проверка ниже делает такое расхождение видимым.
HEADER_DUPS_OK = 220
HEADER_CONFLICT_KEYS = frozenset((
    "F.F2exp.B.theta.n16.kulkarni.ph",
    "F.F2exp.B.theta.n32.kulkarni.Eh",
    "F.F2exp.B.theta.n64.kulkarni.Eh",
))

# Шапка эталона. Воспроизводится дословно, чтобы порождённый файл побайтово
# совпадал с закоммиченным. Прозаическая часть про пути LU (раздел «НА КАКОМ
# ПУТИ ЛИНЕЙНОЙ АЛГЕБРЫ СНЯТЫ ТАБЛИЦЫ») из таблиц не выводится — это результат
# отдельного измерения, и правится она здесь, в шаблоне.
HEADER = """\
# ЭТАЛОННЫЕ ЗНАЧЕНИЯ ИЗ ПУБЛИКАЦИИ (внешний источник истины).
#
# Извлечены МЕХАНИЧЕСКИ (скриптом-парсером) из .tex-таблиц статьи, помеченных
# «Auto-prepared from verified runs. Do not alter numbers»:
#   Scientific Agents/papers/new-01/experiments/tables/  (18 файлов)
# Ручной перенос сознательно не применялся: он и есть источник «ошибки переноса».
#
# Назначение: сверка результатов текущего кода с ОПУБЛИКОВАННЫМИ числами. В отличие
# от characterization/baseline-eh.tsv (снят этой же реализацией и фиксирует лишь
# неизменность поведения), настоящий файл от кода проекта не зависит.
#
# Формат:  ключ <TAB> значение <TAB> файл-источник <TAB> расположение в файле
#
# Ключ:  <уравнение>.<задача>.<базис>.<семейство>.n<N>.<схема>.<величина>
#   уравнение — F (Фредгольм) либо V (Вольтерра);
#   задача    — F2, F2exp, F1, V2, V2exp, V2win, V1;
#   величина  — Eh (погрешность max|u*-u_h|) либо ph (порядок log2(E_h/E_{h/2})).
# Схема именования согласована с characterization/baseline-eh.tsv.
#
# Значения приведены с 4 значащими цифрами — ровно столько напечатано в статье.
# Отсюда допуск сверки 2 %: см. PublishedValuesTest.RELATIVE_TOLERANCE.
#
# ИЗВЕСТНЫЕ РАСХОЖДЕНИЯ МЕЖДУ САМИМИ ТАБЛИЦАМИ. Обнаружены при извлечении: одна и та
# же величина напечатана в двух таблицах по-разному. Это не ошибка переноса, а разные
# прогоны (table-t2-* — «Phase-8 runs», table-t3-* — прогон commit 403fa1d). Все три
# случая лежат НА УРОВНЕ МАШИННОЙ ТОЧНОСТИ и потому из сверки исключены (NOISE_FLOOR);
# в поле расположения они помечены словом РАСХОЖДЕНИЕ:
#   F.F2exp.B.theta.n32.kulkarni.Eh : 7.327e-15 (t2) против 7.994e-15 (t3), 9 %;
#   F.F2exp.B.theta.n64.kulkarni.Eh : 5.995e-15 (t2) против 6.439e-15 (t3), 7 %;
#   F.F2exp.B.theta.n16.kulkarni.ph : 6.45 (t2) против 6.32 (t3) — порядок вычислен
#     по E_h(n=32) из шумовой зоны, поэтому недостоверен сам по себе.
# Оставлено значение из table-t2-*; альтернативное указано в примечании.
# Прочие 220 значений, продублированных в нескольких таблицах, совпали полностью.
#
# ============================================================================
# НА КАКОМ ПУТИ ЛИНЕЙНОЙ АЛГЕБРЫ СНЯТЫ ТАБЛИЦЫ (измерено, этап 8.6)
# ============================================================================
# В самой статье бэкенд не указан. Он восстановлен ПО ФАКТУ — прогоном всех
# 42 ключей F1 на обоих бэкендах (`-Dnumerics.backend=multik|reference`, JDK 21,
# macOS aarch64) и сравнением с опубликованными числами. Результат:
#
#   table-xi-f1.tex  (36 ключей F1) — снята на MULTIK/OpenBLAS.
#       multik    против публикации: макс. 0.033 %, медиана 0.0032 %;
#       reference против публикации: макс. 11.485 %, медиана 1.162 %.
#       Воспроизводится практически бит-в-бит, допуск — общие 2 %.
#
#   table-f1.tex     (6 ключей F.F1.H.theta.*) — снята на JVM-пУТИ LU (ReferenceBackend
#       либо арифметически близкая реализация), А НЕ на multik.
#       reference против публикации: макс. 4.234 %, медиана 0.010 %
#                   (4 ключа из 6 совпадают до 4-й значащей цифры);
#       multik    против публикации: макс. 6.780 %, медиана 3.471 %.
#       Допуск — 8 %, см. PublishedValuesTest.LU_PATH_DEPENDENT_TOLERANCE. Общий
#       допуск 2 % НЕ ОСЛАБЛЕН: послабление касается только этих 6 ключей.
#
# ПРИЧИНА, почему разные пути LU вообще дают разные числа именно в F1: это
# уравнение ПЕРВОГО рода с регуляризацией Вазваза (alpha = 1e-10, c_L = -1e10).
# Измерено: cond_inf(I-M) = 1.18e10..2.70e10, ‖g‖_inf = 1.59e10, а в схеме Слоана два
# слагаемых порядка 1.38e10 сокращаются до 2.7 (потеря ~9.7 из 16 цифр).
# Измеренный разброс multik против reference ВНУТРИ САМОЙ table-f1.tex (6 ключей):
# макс. 7.267 % (ключ F.F1.H.theta.n8.sloan) — именно из этого числа выведен допуск
# 8 % (= 0.05 % точности публикации + 7.267 %, округлено вверх).
# Для сравнения, ПО ВСЕЙ группе F1 (42 ключа) тот же разброс был бы шире:
# макс. 11.483 % (ключ F.F1.B.xi1.n32.sloan из ДРУГОЙ таблицы), медиана 0.98 %.
# Взят УЗКИЙ вариант: разброс чужой таблицы не должен послаблять эту.
# Подробный замер: .tasks/code-review-remediation/stage8/MEASURE-8.6-f1-tolerance.md.
#
# ОСТАЛЬНЫЕ ТАБЛИЦЫ (F2/F2exp/V1/V2/V2exp/V2win, 700 ключей) от пути LU НЕ ЗАВИСЯТ:
# там нет масштабирования на 1/alpha, cond(I-M) порядка единиц, и все они проходят
# под общим допуском 2 % на ОБОИХ бэкендах. Поэтому единого «бэкенда статьи» не
# существует и быть записано одной строкой не может — только по-таблично.
"""

NUM = re.compile(r'\$?(-?\d+\.\d+)\{?\\+times\}?10\^\{(-?\d+)\}\$?')
PLAIN = re.compile(r'^\$?\(?(-?\d+\.\d+)\)?\$?$')

FAMMAP = {'theta': 'theta', 'xi': 'xi1', 'mu': 'mu', 'lambda': 'lambda'}

T2SCHEMES = ['base', 'sloan', 'kulkarni', 'iterKulkarni']
T3SCHEMES = ['base', 'sloan', 'kulkarni', 'nystrom', 'iterNystrom']


class TableError(Exception):
    """Таблица статьи недоступна либо не поддаётся разбору."""


def cells(line):
    """Разбивает строку таблицы на ячейки."""
    line = line.strip()
    line = re.sub(r'\\+\\\s*$', '', line)          # хвостовой \\
    return [c.strip() for c in line.split('&')]


def num(tok):
    """Значение из ячейки вида $1.014{\\times}10^{-4}$ (или None)."""
    m = NUM.search(tok)
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    return None


def order(tok):
    """p_h из хвоста ячейки: '(3.02)' либо '($-3.31$)'; '---' -> None."""
    m = re.search(r'\(\s*\$?(-?\d+\.\d+)\$?\s*\)', tok)
    return float(m.group(1)) if m else None


def fmt(v):
    """4 значащие цифры - ровно столько, сколько напечатано в публикации."""
    return f"{v:.4g}"


def lines_of(tables_dir, name):
    """Строки файла таблицы; недоступность файла — ошибка входных данных."""
    path = os.path.join(tables_dir, name)
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.readlines()
    except OSError as exc:
        raise TableError(f"не читается таблица {name}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TableError(f"таблица {name} не в кодировке UTF-8: {exc}") from exc


def rows(tables_dir, path):
    """Строки данных: начинаются с числа n."""
    out = []
    for raw in lines_of(tables_dir, path):
        s = raw.strip()
        if re.match(r'^\d+\s*&', s):
            out.append((int(s.split('&')[0].strip()), cells(s), raw))
        elif 'multicolumn' in s:
            out.append((None, None, raw))
    return out


def block_of(raw):
    """Опознаёт заголовок подраздела таблицы."""
    if 'mathcal{B}' in raw:
        return 'B'
    if 'mathcal{H}' in raw:
        return 'H'
    if 'mathcal{T}' in raw:
        return 'T'
    return None


def extract(tables_dir):
    """Разбор 18 таблиц статьи. Возвращает список (ключ, значение, файл, пометка)."""
    records = []

    def rec(key, val, src, note):
        if val is None:
            return
        records.append((key, val, src, note))

    # ---- table-t1-*: базовая схема, theta, блоки по базисам, колонки n,h,E,p,C
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
            rec(f"{eq}.{prob}.{sysname}.theta.n{n}.base.Eh", num(c[2]), fname, f"базис {sysname}")
            p = PLAIN.match(c[3].replace('$', ''))
            if p:
                rec(f"{eq}.{prob}.{sysname}.theta.n{n}.base.ph", float(p.group(1)), fname, f"базис {sysname}")

    # ---- table-t2-*: базис B, theta, колонки база/Слоан/Кулкарни/итер.Кулкарни, блоки по задачам
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

    # ---- table-t3-*: базис B, theta, + Nyström
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

    # ---- table-xi-t1-*: базовая схема, блоки по xi<r>, колонки B/H/T
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

    # ---- table-xi-t2-*: базис B, блоки по xi<r>, колонки схем
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

    # ---- table-families: базис B, базовая схема, блоки по задачам, строки по семействам
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

    # ---- table-xi-special: xi1, базовая схема, блоки по задачам, колонки B/H/T
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

    # ---- table-f1: F1, базис H, theta, колонки n,h,база,Слоан
    for n, c, raw in rows(tables_dir, 'table-f1.tex'):
        if n is None:
            continue
        rec(f"F.F1.H.theta.n{n}.base.Eh", num(c[2]), 'table-f1.tex', 'alpha=1e-10')
        rec(f"F.F1.H.theta.n{n}.sloan.Eh", num(c[3]), 'table-f1.tex', 'alpha=1e-10')

    # ---- table-v1: V1, базис B, theta, колонки n,база,Слоан,Кулкарни
    for n, c, raw in rows(tables_dir, 'table-v1.tex'):
        if n is None:
            continue
        for k, sch in enumerate(['base', 'sloan', 'kulkarni']):
            rec(f"V.V1.B.theta.n{n}.{sch}.Eh", num(c[k + 1]), 'table-v1.tex', 'сведение к V2')

    # ---- table-xi-f1: F1, xi1/xi2, базисы B/H/T, схемы база/Слоан
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
            "разбор не дал ни одной записи по таблицам: " + ", ".join(silent) +
            " — вероятно, изменилась вёрстка таблицы в статье"
        )
    return records


def reconcile(records):
    """Свёртка дубликатов в словарь ключ -> (значение, файл, пометка).

    Одно и то же число встречается в нескольких таблицах. Совпадающие в пределах
    DUPLICATE_TOLERANCE считаются согласованными; расходящиеся оставляют значение
    из первой таблицы, а альтернатива дописывается в пометку.
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
        seen[k] = (o[0], o[1], seen[k][2] + f"; РАСХОЖДЕНИЕ С {nv[1]}: {fmt(nv[0])}")
    return seen, dups_ok, dups_bad


def render(seen, tables_dir, stamp):
    """Текст эталона целиком: шапка плюс отсортированные по ключу строки данных."""
    parts = [HEADER]
    if stamp:
        parts.append(
            f"# Порождено: tools/parse_published_values.py --tables-dir {tables_dir}\n"
        )
    for key in sorted(seen):
        v, src, note = seen[key]
        parts.append(f"{key}\t{fmt(v)}\t{src}\t{note}\n")
    return "".join(parts)


def report(seen, records, dups_ok, dups_bad):
    """Сводка разбора. Расхождения между таблицами — предупреждение, не отказ."""
    print(f"извлечено записей: {len(records)}, уникальных ключей: {len(seen)}")
    print(f"согласованных дубликатов между таблицами: {dups_ok}")
    if dups_bad:
        print("ПРЕДУПРЕЖДЕНИЕ: расхождения между таблицами статьи "
              "(оставлено значение из первой таблицы):")
        for k, o, nv in dups_bad:
            print("   ", k, o, nv)
    else:
        print("противоречий между таблицами нет")
    eh = sum(1 for k in seen if k.endswith('.Eh'))
    print(f"E_h: {eh}, p_h: {len(seen) - eh}")

    # Шапка описывает разбор словами; при расхождении описание устарело.
    if dups_ok != HEADER_DUPS_OK:
        print(f"ПРЕДУПРЕЖДЕНИЕ: шапка эталона называет {HEADER_DUPS_OK} согласованных "
              f"дубликатов, разбор дал {dups_ok} — текст шапки в скрипте пора обновить")
    actual_conflicts = frozenset(k for k, _o, _nv in dups_bad)
    if actual_conflicts != HEADER_CONFLICT_KEYS:
        print("ПРЕДУПРЕЖДЕНИЕ: набор расхождений между таблицами отличается от "
              "перечисленного в шапке эталона — текст шапки в скрипте пора обновить")


def data_map(text):
    """Строки данных текста эталона в виде ключ -> остаток строки."""
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        key, _, rest = line.partition('\t')
        result[key] = rest
    return result


def compare(expected_text, actual_text, out_path):
    """Сверка порождённого текста с закоммиченным. 0 — совпало, 1 — нет."""
    if expected_text == actual_text:
        print(f"сверка с {out_path}: совпадает побайтово")
        return EXIT_OK

    expected = data_map(expected_text)
    actual = data_map(actual_text)
    differing = sorted(
        set(expected) ^ set(actual) |
        {k for k in set(expected) & set(actual) if expected[k] != actual[k]}
    )
    print(f"РАСХОЖДЕНИЕ: {out_path} не совпадает с порождённым из таблиц статьи.")
    if differing:
        print(f"различающихся ключей: {len(differing)} (показаны первые 20)")
        for key in differing[:20]:
            print(f"    {key}: в файле {expected.get(key, '<нет>')} | "
                  f"порождено {actual.get(key, '<нет>')}")
    else:
        print("значения совпадают, различается шапка комментариев")
    print("Эталон правится только повторным запуском этого скрипта, не руками.")
    return EXIT_CHECK_FAILED


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Порождение эталона published-values.tsv из .tex-таблиц статьи",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Исходники статьи лежат вне репозитория, поэтому --tables-dir обязателен\n"
               "и в сборку Gradle этот скрипт не встроен.",
    )
    parser.add_argument("--tables-dir", required=True,
                        help=f"каталог с {len(TABLE_FILES)} файлами table-*.tex статьи")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="файл результата (по умолчанию закоммиченный эталон)")
    parser.add_argument("--check", action="store_true",
                        help="не писать, а сверить с существующим --out; "
                             "код возврата 1 при расхождении")
    parser.add_argument("--stamp", action="store_true",
                        help="дописать в шапку строку о происхождении файла; "
                             "нарушает побайтовое совпадение с закоммиченным эталоном")
    args = parser.parse_args()

    tables_dir = args.tables_dir
    if not os.path.isdir(tables_dir):
        print(f"ОШИБКА: каталог таблиц не найден: {tables_dir}")
        return EXIT_INPUT_ERROR
    missing = [name for name in TABLE_FILES
               if not os.path.isfile(os.path.join(tables_dir, name))]
    if missing:
        print(f"ОШИБКА: в {tables_dir} не хватает {len(missing)} из "
              f"{len(TABLE_FILES)} таблиц: {', '.join(missing)}")
        return EXIT_INPUT_ERROR

    try:
        records = extract(tables_dir)
    except TableError as exc:
        print(f"ОШИБКА: {exc}")
        return EXIT_INPUT_ERROR

    seen, dups_ok, dups_bad = reconcile(records)
    report(seen, records, dups_ok, dups_bad)
    text = render(seen, tables_dir, args.stamp)

    if args.check:
        if not os.path.isfile(args.out):
            print(f"ОШИБКА: нечего сверять, файл не найден: {args.out}")
            return EXIT_INPUT_ERROR
        # Порождаем во временный файл: сверка обязана быть без побочного эффекта,
        # иначе она молча чинила бы то, что должна обнаруживать.
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
        print(f"ОШИБКА: не удалось записать {args.out}: {exc}")
        return EXIT_OUTPUT_ERROR
    print(f"записано: {os.path.abspath(args.out)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
