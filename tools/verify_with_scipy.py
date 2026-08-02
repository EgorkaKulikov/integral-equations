#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ВНЕШНЯЯ СВЕРКА ВЫЧИСЛИТЕЛЬНОЙ БАЗЫ СО SciPy/NumPy.

Назначение
----------
Все штатные проверки проекта написаны на его же коде, поэтому подтверждают лишь
внутреннюю согласованность. Настоящий скрипт сверяет результаты со СТОРОННИМИ
библиотеками (SciPy, NumPy), разработанными независимо, и тем самым даёт
доказательство, не замкнутое на проверяемую реализацию.

Сверка ПОСЛОЙНАЯ — снизу вверх, от фундамента к схемам. Такой порядок выбран
сознательно: при расхождении сразу видно, какой слой виноват. Сравнение только
итоговых погрешностей этого не дало бы.

    L1  квадратура Гаусса--Лежандра   <- numpy.polynomial.legendre.leggauss
    L2  линейная алгебра (СЛАУ)       <- scipy.linalg.solve, numpy.linalg.cond
    L3  базис минимальных сплайнов    <- scipy.interpolate.BSpline
    L4  образы интегральных операторов<- scipy.integrate.quad (QUADPACK)
    L5  правые части модельных задач  <- scipy.integrate.quad
    L6a пригодность эталона          <- собственный Nystrom (80 узлов leggauss)
                                         против точного решения
    L6b решения ПРОЕКТА (E_h)        <- тот же Nystrom: пороги и наблюдаемый
                                         порядок сходимости для solution-errors.tsv

Слои L6a и L6b разделены намеренно: L6a НЕ читает ни одного артефакта проекта и
отвечает лишь на вопрос «годится ли эталон в качестве эталона»; сверка самих
решений проекта — только в L6b. Общее имя «L6» создавало впечатление, будто
решения сверены, когда сверялся один оракул.

Чего сверка НЕ покрывает (принципиально, а не по недоработке)
------------------------------------------------------------
* Гиперболические (H) и тригонометрические (T) минимальные сплайны: аналогов в
  SciPy нет. Проверяются математическими инвариантами внутри проекта.
* Четыре семейства аппроксимационных функционалов (theta, xi, mu, lambda):
  конструкции из работ авторов, аналогов в сторонних библиотеках нет. Для
  системы B проверяется производное свойство — биортогональность к базису,
  построенному средствами SciPy.
* Схемы Кулкарни и комбинированный Nystrom: сверяются порядком сходимости.

Порядок запуска
---------------
Штатный способ — в составе проверок проекта (см. `ScipyCrossVerificationTest`):

    ./gradlew scipyVerify                      # готовит venv, выгружает артефакты, сверяет

Ручной запуск для разбора расхождения:

    ./gradlew dumpVerificationArtifacts        # выгрузка артефактов в build/verification/
    ./gradlew setupScipyVerification           # создание venv со SciPy
    .venv-verify/bin/python tools/verify_with_scipy.py

Аргументы
---------
    --artifacts <каталог>   каталог с выгруженными артефактами
                            (по умолчанию build/verification)
    --json <файл>          дополнительно записать результат в машиночитаемом
                            виде — нужно Kotlin-тесту, чтобы разобрать сверку
                            по слоям, а не полагаться только на код возврата

Код возврата: 0 — все слои сошлись, 1 — есть расхождения (перечислены в выводе),
2 — недоступны SciPy/NumPy, 3 — отчёт не сериализуется, 4 — нет пригодных
метаданных выгрузки (`dump-meta.tsv`): без границ отрезка сверять интегралы нечем,
а подстановка [0,1] по умолчанию сохранила бы дыру в скрытом виде.
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
    print(f"ОШИБКА: не найдены SciPy/NumPy ({exc}).")
    # Версии закреплены в файле требований: сверка обязана быть воспроизводимой,
    # иначе расхождение не отличить от смены поведения самой SciPy.
    print("Установите: python3 -m venv .venv-verify && "
          ".venv-verify/bin/pip install -r tools/requirements-verify.txt")
    print("Либо просто: ./gradlew setupScipyVerification")
    sys.exit(2)

# Каталог артефактов; переопределяется аргументом --artifacts.
ARTIFACT_DIR = os.path.join("build", "verification")

# Допуски подобраны по природе сравниваемых величин, а не «чтобы прошло».
TOL_QUADRATURE = 1e-14   # узлы/веса: обе реализации работают на машинной точности
TOL_BASIS = 1e-12        # базис: накапливается ошибка обращения матрицы 3x3
TOL_LINALG = 1e-10       # СЛАУ: зависит от обусловленности (та печатается отдельно)
TOL_INTEGRAL = 1e-10     # интегралы: QUADPACK против составной квадратуры 8-го порядка

# Для ВТОРОЙ производной базиса сравнение обязано быть ОТНОСИТЕЛЬНЫМ.
#
# Причина (установлена измерением, а не предположена): omega'' растёт как
# O(1/h^2), и на неравномерных сетках достигает |omega''| ~ 150...190. Наблюдавшееся
# абсолютное отклонение 1.79e-12 отвечает ОТНОСИТЕЛЬНОЙ величине 1.2e-14, то есть
# порядка 50 ulp — это шум округления двух разных способов вычисления
# (проект обращает матрицу 3x3, SciPy использует рекуррентные соотношения
# де Бура), а не расхождение методов.
#
# Важно: это ИСПРАВЛЕНИЕ НЕВЕРНОГО КРИТЕРИЯ, а не ослабление допуска:
# абсолютный порог для величины масштаба 200 требовал бы точности выше машинной.
# Сама относительная граница остаётся строгой.
TOL_BASIS_D2_RELATIVE = 1e-13

# Пригодность эталонного Nystrom (слой L6a): он обязан воспроизводить точное
# решение на порядки лучше схем проекта, иначе сверять им нельзя. Раньше
# величина была безымянным литералом прямо в вызове `report`, где её нельзя было
# ни найти рядом с остальными допусками, ни обосновать.
# Фактическое отклонение — 4.4e-16 и 1.3e-15, то есть запас три порядка.
TOL_REFERENCE_FITNESS = 1e-12

failures: list[str] = []
notes: list[str] = []

# Машиночитаемая сводка всех проверок: её разбирает Kotlin-тест, чтобы
# сообщить ИМЕННО тот слой, который рассогласован, а не просто «сверка не прошла».
checks: list[dict] = []


def report(
    layer: str,
    name: str,
    deviation: float,
    tolerance: float,
    compared: int,
    skipped: int = 0,
    kind: str = "абс.",
) -> None:
    """Печатает результат одной проверки и запоминает расхождение.

    Величины ОБЯЗАТЕЛЬНО приводятся к базовым типам Python: часть проверок
    вычисляет отклонение средствами NumPy и получает `numpy.float64`, а сравнение
    даёт `numpy.bool_`. Оба типа не сериализуются в JSON: без приведения
    `json.dump` прерывается посередине записи и оставляет ОБРЕЗАННЫЙ файл, что
    выглядит как «слой не выполнен» и сбивает диагностику.

    `compared` — число ФАКТИЧЕСКИ сравнённых точек, `skipped` — число точек,
    выпавших из сравнения. Оба обязательны, и это не украшение вывода:
    отклонение инициализируется нулём, поэтому проверка, из которой выпали ВСЕ
    точки, давала бы `0.0 <= tol` и печатала OK, ничего не проверив. Поэтому
    `compared == 0` трактуется как ПРОВАЛ проверки: «сравнить нечего» — это не
    «расхождений нет», а отсутствие доказательства.

    НЕФИНИТНОЕ ОТКЛОНЕНИЕ (NaN, ±inf) — ТОЖЕ ПРОВАЛ, и с ОТДЕЛЬНЫМ текстом.
    Причина фактическая, а не гипотетическая: встроенный `max(worst, nan)`
    возвращает `worst` (сравнение с NaN всегда ложно), поэтому NaN в артефакте
    ДАВАЛ СТАТУС OK — то есть полностью битое число проходило сверку. Поэтому
    места накопления пользуются [worst_of], который ПРОПУСКАЕТ нефинитное значение
    дальше, а здесь оно превращается в отказ. Из двух возможных решений (считать
    нефинитные точки пропусками либо сразу валить проверку) выбрано второе:
    NaN в выгрузке НИКОГДА не бывает нормой — это либо дефект вычисления, либо
    повреждённый артефакт, и маскировать его под «пропуск» значило бы терять
    причину в шуме легитимных пропусков (точки на узлах, вырожденные отрезки).
    """
    deviation = float(deviation)
    tolerance = float(tolerance)
    compared = int(compared)
    skipped = int(skipped)
    empty = compared <= 0
    nonfinite = not math.isfinite(deviation)
    ok = bool(deviation <= tolerance) and not empty and not nonfinite
    if nonfinite:
        status = "НЕФИНИТНО"
    elif empty:
        status = "НЕТ ДАННЫХ"
    elif ok:
        status = "OK "
    else:
        status = "РАСХОЖДЕНИЕ"
    print(
        f"  [{status}] {name}: {kind} отклонение {deviation:.3e} (допуск {tolerance:.1e}), "
        f"сравнено точек {compared}, пропущено {skipped}"
    )
    checks.append({
        "layer": layer,
        "name": name,
        # JSON не знает NaN/Infinity: `json.dumps` по умолчанию пишет голые
        # литералы `NaN`, невалидные по стандарту, и регексп разбора в
        # Kotlin-тесте на такой элемент не сойдётся — проверка ПРОПАЛА бы из
        # отчёта вместо того, чтобы его уронить. Поэтому в JSON идёт строковое
        # представление, а числовое поле получает заведомо провальное значение.
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
            f"{layer} / {name}: отклонение НЕФИНИТНО ({deviation!r}) — в сравнение попало "
            f"NaN или бесконечность (сравнено точек {compared}, пропущено {skipped}). Это либо "
            f"дефект вычисления в проекте, либо повреждённый артефакт; ни то, ни другое "
            f"не смеет давать статус OK"
        )
    elif empty:
        failures.append(
            f"{layer} / {name}: сравнено 0 точек (пропущено {skipped}) — проверка НЕ выполнена, "
            f"нулевое отклонение здесь означает отсутствие данных, а не согласие"
        )
    elif not ok:
        failures.append(f"{layer} / {name}: {kind} отклонение {deviation:.3e} > допуска {tolerance:.1e}")


def worst_of(current: float, candidate: float) -> float:
    """Максимум двух отклонений, НЕ ГЛОТАЮЩИЙ NaN и бесконечности.

    Зачем нужна замена встроенного `max`. В Python `max(0.0, float("nan"))` равно
    `0.0`: все сравнения с NaN ложны, поэтому кандидат тихо отбрасывается. В цикле
    накопления это значило, что NaN в артефакте вообще НЕ ВЛИЯЛ на результат,
    и сверка объявляла OK. Здесь нефинитное значение, наоборот, ПОГЛОЩАЕТ
    накопленное и доходит до [report], где становится явным провалом.
    """
    candidate = float(candidate)
    if not math.isfinite(candidate):
        return candidate
    current = float(current)
    if not math.isfinite(current):
        return current
    return max(current, candidate)


def worst_over(values) -> float:
    """[worst_of] по всем элементам; пустой ввод даёт 0.0 (случай compared == 0)."""
    result = 0.0
    for value in values:
        result = worst_of(result, value)
    return result


class MetaError(RuntimeError):
    """Метаданные выгрузки отсутствуют или непригодны."""


# Метаданные выгрузки (границы отрезка, размеры сеток). Заполняются из
# `dump-meta.tsv` в [main]; ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ НЕТ СОЗНАТЕЛЬНО.
META: dict[str, float] = {}

# Ключи, без которых сверять нечем: границы отрезка задают все интегралы.
REQUIRED_META_KEYS = ("a", "b", "dumpGridSize", "quadratureNodes")

META_FILE = "dump-meta.tsv"


def load_meta() -> dict[str, float]:
    """Читает метаданные выгрузки.

    Отсутствие файла — ОШИБКА, а не повод подставить `[0,1]`. Скрипт раньше
    считал интегралы по зашитому отрезку, и совпадение с дампером держалось на
    том, что тот пользуется значениями по умолчанию `Grid.uniform(n)`. Смена
    границ в дампере дала бы МОЛЧАЛИВУЮ сверку с другими интегралами; подстановка
    значения по умолчанию здесь сохранила бы ровно эту дыру.
    """
    path = os.path.join(ARTIFACT_DIR, META_FILE)
    if not os.path.exists(path):
        raise MetaError(
            f"нет файла метаданных выгрузки {path}. Границы отрезка и размеры сеток задаёт "
            f"дампер (VerificationArtifacts.dumpMeta), и подставлять [0,1] по умолчанию нельзя: "
            f"при других границах сверка молча считала бы ДРУГИЕ интегралы. "
            f"Выполните ./gradlew dumpVerificationArtifacts"
        )
    parsed: dict[str, float] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise MetaError(f"строка метаданных не в формате «ключ<TAB>значение»: {line!r}")
            try:
                parsed[parts[0]] = float(parts[1])
            except ValueError as exc:
                raise MetaError(f"нечисловое значение метаданных {parts[0]}={parts[1]!r} ({exc})") from exc
    missing = [key for key in REQUIRED_META_KEYS if key not in parsed]
    if missing:
        raise MetaError(f"в {META_FILE} отсутствуют обязательные ключи: {missing}")
    if not parsed["b"] > parsed["a"]:
        raise MetaError(f"недопустимый отрезок: a={parsed['a']}, b={parsed['b']}")
    return parsed


def interval() -> tuple[float, float]:
    """Границы отрезка из метаданных выгрузки."""
    if not META:
        raise MetaError("метаданные выгрузки не загружены")
    return META["a"], META["b"]


def load_tsv(filename: str) -> list[list[str]]:
    """Читает выгруженный TSV, пропуская строки-комментарии."""
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        print(f"  ПРОПУСК: нет файла {path} — сначала запустите ./gradlew dumpVerificationArtifacts")
        notes.append(f"файл {filename} отсутствует, слой не проверен")
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
# L1. Квадратура Гаусса--Лежандра
# ---------------------------------------------------------------------------
def verify_quadrature() -> None:
    print("\nL1. Узлы и веса квадратуры Гаусса--Лежандра (эталон: numpy leggauss)")
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
    report("L1", f"узлы (m = 1..{max(by_m)})", worst_node, TOL_QUADRATURE,
           compared=compared, skipped=skipped)
    report("L1", f"веса (m = 1..{max(by_m)})", worst_weight, TOL_QUADRATURE,
           compared=compared, skipped=skipped)


# ---------------------------------------------------------------------------
# L3. Базис минимальных сплайнов для полиномиальной системы B
# ---------------------------------------------------------------------------
def verify_spline_basis() -> None:
    print("\nL3. Базис минимальных сплайнов, система B (эталон: scipy BSpline)")
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
        # Полный вектор узлов x_{-2..n+2}: клампованный вектор степени 2.
        ordered = [knot_map[k] for k in sorted(knot_map)]
        knot_vector = np.array(ordered)
        degree = 2
        basis_count = len(knot_vector) - degree - 1
        worst_value = 0.0
        worst_deriv = 0.0
        worst_deriv2_relative = 0.0
        worst_deriv2_magnitude = 0.0
        # Счётчики ведутся ОТДЕЛЬНО по трём величинам: у значения, первой и второй
        # производной разные основания пропуска (у второй ещё и точки на узлах).
        compared_value = compared_deriv = compared_deriv2 = 0
        skipped_value = skipped_deriv = skipped_deriv2 = 0
        for j, t, omega, omega_d, omega_dd in values[grid_name]:
            # Индексация проекта j = -2..n-1 отвечает индексу SciPy j + 2.
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
            # Вторая производная кусочно-постоянна и терпит разрыв в узлах сетки;
            # проект берёт значение по правому куску, поэтому точки, попавшие на
            # узел, из сравнения исключаются (это соглашение, а не расхождение).
            on_knot = any(abs(t - x) < 1e-12 for x in ordered)
            if not on_knot:
                ref_dd = float(spline.derivative(2)(t))
                if not math.isnan(ref_dd):
                    # Относительное сравнение: |omega''| ~ 1/h^2 и достигает сотен,
                    # поэтому абсолютный порог здесь не имеет смысла (см. TOL_BASIS_D2_RELATIVE).
                    scale = max(abs(ref_dd), 1.0)
                    worst_deriv2_relative = worst_of(worst_deriv2_relative, abs(omega_dd - ref_dd) / scale)
                    worst_deriv2_magnitude = max(worst_deriv2_magnitude, abs(ref_dd))
                    compared_deriv2 += 1
                else:
                    skipped_deriv2 += 1
            else:
                skipped_deriv2 += 1
        report("L3", f"{grid_name}: значения omega_j", worst_value, TOL_BASIS,
               compared=compared_value, skipped=skipped_value)
        report("L3", f"{grid_name}: первая производная", worst_deriv, TOL_BASIS,
               compared=compared_deriv, skipped=skipped_deriv)
        report(
            "L3",
            f"{grid_name}: вторая производная (max|omega''| = {worst_deriv2_magnitude:.1f})",
            worst_deriv2_relative,
            TOL_BASIS_D2_RELATIVE,
            compared=compared_deriv2,
            skipped=skipped_deriv2,
            kind="отн.",
        )


# ---------------------------------------------------------------------------
# L2. Линейная алгебра: решение собранной системы
# ---------------------------------------------------------------------------
def verify_linear_algebra() -> None:
    print("\nL2. Линейная алгебра: (I - M) c = g (эталон: scipy.linalg.solve)")
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
    print(f"  Обусловленность (I - M): {condition:.3e} — определяет достижимую точность")
    c_reference = scipy_solve(a, g)
    report("L2", "коэффициенты базовой схемы", float(np.max(np.abs(c_project - c_reference))), TOL_LINALG,
           compared=int(size), skipped=0)
    residual = float(np.max(np.abs(a @ c_project - g)))
    report("L2", "невязка ||(I - M)c - g||", residual, TOL_LINALG,
           compared=int(size), skipped=0)


# ---------------------------------------------------------------------------
# L4/L5. Образы операторов и правые части
# ---------------------------------------------------------------------------
# Ядра, точные решения и ИХ ПРОИЗВОДНЫЕ, выведенные НЕЗАВИСИМО от кода проекта.
#
# ПРИНЦИПИАЛЬНО: все выражения ниже получены дифференцированием от руки по
# закрытым формулам K(t,s) и u(t), а НЕ списаны из реализации. Сверка,
# использующая формулы проекта, замыкается сама на себя и не является доказательством.
#
# Состав каждой записи:
#   kernel     K(t,s)
#   kernel_t   dK/dt          (частная)
#   kernel_tt  d2K/dt2        (частная)
#   kernel_s   dK/ds          (частная; нужна ТОЛЬКО для полной производной диагонали)
#   exact      u(t), exact_d u'(t), exact_d2 u''(t)
#
# Диагональ и её производная НЕ выписываются отдельными закрытыми формулами,
# а собираются из частных (см. [diagonal] и [diagonal_total_deriv]): так цепное
# правило d/dt K(t,t) = K_t(t,t) + K_s(t,t) видно в коде, а не скрыто в ручном
# упрощении, где его легче всего потерять именно в этом месте.
#
# ВЫВОД по задачам (всё тривиально дифференцируется, поэтому проверяемо глазом):
#
#   F2, V2:  K = 1/(1+t+s)
#            K_t  = -1/(1+t+s)^2      K_tt = 2/(1+t+s)^3      K_s = -1/(1+t+s)^2
#            u = 1/(1+t), u' = -1/(1+t)^2, u'' = 2/(1+t)^3
#            диагональ K(t,t) = 1/(1+2t), d/dt = -2/(1+2t)^2 = K_t(t,t)+K_s(t,t)  ✓
#
#   F2exp, V2exp:  K = exp(-(t-s)^2)
#            K_t  = -2(t-s)K          K_tt = (4(t-s)^2 - 2)K   K_s = 2(t-s)K
#            u = exp(t), u' = u'' = exp(t)
#            диагональ K(t,t) = 1, d/dt = 0 = K_t(t,t)+K_s(t,t) = 0 + 0        ✓
#
#   V2win:   K = t - s
#            K_t  = 1                 K_tt = 0                K_s = -1
#            u = cos t, u' = -sin t, u'' = -cos t
#            диагональ K(t,t) = 0, d/dt = 0 = K_t(t,t)+K_s(t,t) = 1 + (-1)      ✓
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
    """Значение ядра на диагонали, K(t,t)."""
    return float(spec["kernel"](t, t))


def diagonal_total_deriv(spec: dict, t: float) -> float:
    """ПОЛНАЯ производная диагонали: d/dt K(t,t) = K_t(t,t) + K_s(t,t).

    Отдельная функция — потому что именно здесь делается главная ошибка вывода
    второй производной по Лейбницу: вместо ПОЛНОЙ производной диагонали берут
    частную K_t(t,t), теряя слагаемое K_s(t,t). На симметричных ядрах вида
    K(t-s) эта ошибка НЕ видна (оба слагаемых нулевые), поэтому в сверке обязательно
    присутствуют задачи с несимметричным поведением диагонали (V2: -2/(1+2t)^2).
    """
    return float(spec["kernel_t"](t, t)) + float(spec["kernel_s"](t, t))


def fredholm_image_deriv(spec: dict, t: float, lower: float, upper: float, order: int) -> float:
    """(Ku)^(order)(t) для Фредгольма, order в {0,1,2}.

    Пределы ПОСТОЯННы, поэтому правило Лейбница вырождается в дифференцирование
    ПОД знаком интеграла (подынтегральная функция гладкая, перестановка законна):

        (Ku)(t)   = ∫_a^b K(t,s)    u(s) ds
        (Ku)'(t)  = ∫_a^b K_t(t,s)  u(s) ds
        (Ku)''(t) = ∫_a^b K_tt(t,s) u(s) ds
    """
    kernel = {0: spec["kernel"], 1: spec["kernel_t"], 2: spec["kernel_tt"]}[order]
    exact = spec["exact"]
    value, _ = quad(lambda s: kernel(t, s) * exact(s), lower, upper, limit=200)
    return float(value)


def volterra_image_deriv(spec: dict, t: float, lower: float, order: int) -> float:
    """(Vu)^(order)(t) для Вольтерры, order в {0,1,2}. ВЫВОД НИЖЕ.

    Пусть (Vu)(t) = ∫_a^t K(t,s) u(s) ds. Верхний предел ПЕРЕМЕННЫЙ, поэтому
    работает правило Лейбница в общем виде:

        d/dt ∫_a^{b(t)} f(t,s) ds = f(t, b(t)) b'(t) + ∫_a^{b(t)} f_t(t,s) ds.

    Шаг 1 (первая производная). Здесь f(t,s) = K(t,s)u(s), b(t) = t, b'(t) = 1:

        (Vu)'(t) = K(t,t) u(t) + ∫_a^t K_t(t,s) u(s) ds.                        (1)

    Шаг 2 (вторая производная). Дифференцируем (1) по t почленно.

    Слагаемое A(t) = K(t,t) u(t) — ПРОИЗВЕДЕНИЕ двух функций одного аргумента, и
    диагональ K(t,t) зависит от t ОБОИМИ аргументами. Значит, нужна ПОЛНАЯ
    производная диагонали (цепное правило), а НЕ частная K_t(t,t):

        d/dt K(t,t) = K_t(t,t) + K_s(t,t),
        A'(t) = [K_t(t,t) + K_s(t,t)] u(t) + K(t,t) u'(t).                      (2)

    Слагаемое B(t) = ∫_a^t K_t(t,s) u(s) ds — снова правило Лейбница,
    теперь с f(t,s) = K_t(t,s)u(s):

        B'(t) = K_t(t,t) u(t) + ∫_a^t K_tt(t,s) u(s) ds.                        (3)

    Сложив (2) и (3):

        (Vu)''(t) = [K_t(t,t) + K_s(t,t)] u(t)   <- полная производная диагонали
                  + K(t,t) u'(t)                 <- диагональ на производную решения
                  + K_t(t,t) u(t)                <- второй, ОТДЕЛЬНЫЙ граничный вклад
                  + ∫_a^t K_tt(t,s) u(s) ds.                                    (4)

    ВНИМАНИЕ: K_t(t,t) входит В ФОРМУЛУ ДВАЖДЫ и из РАЗНЫХ источников: одинраз в
    составе полной производной диагонали (2), второй — как граничный вклад (3).
    Слить их в одно слагаемое нельзя — это разные члены, случайно совпавшие видом.

    КОНТРОЛЬ ВЫВОДА на V2win (K = t - s, u = cos t), где всё считается в закрытом виде.
    Здесь K(t,t) = 0, так что первый член Лейбница в (1) обнуляется, и при a = 0:
        (Vu)(t)   = ∫_0^t (t-s) cos s ds = t sin t - (t sin t + cos t - 1) = 1 - cos t,
        (Vu)'(t)  = sin t   — и формула (1) даёт 0 + ∫_0^t 1·cos s ds = sin t        ✓
        (Vu)''(t) = cos t   — и формула (4) даёт
                    [1 + (-1)]·cos t + 0·(-sin t) + 1·cos t + ∫_0^t 0 ds = cos t     ✓
    Заметьте: если бы в (2) вместо полной производной стояла частная K_t(t,t) = 1,
    получилось бы 2 cos t — вдвое больше. Именно эта задача ловит путаницу
    между полной и частной производной диагонали.

    Вторая, независимая проверка вывода (выполнена фактически для всех трёх задач):
    центральные конечные разности от (Vu)(t), считанного `quad`, совпадают с (1) и (4)
    с точностью порядка шага разности — то есть вывод не содержит алгебраической ошибки.
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
    raise ValueError(f"volterra_image_deriv: неподдерживаемый порядок {order}")


def verify_operator_images() -> None:
    print("\nL4/L5. Образы операторов и правые части (эталон: scipy.integrate.quad)")
    rows = load_tsv("operator-images.tsv")
    if not rows:
        return
    lower, upper = interval()
    # На каждую проверку: худшее отклонение, число сравнённых и пропущенных точек.
    stats: dict[str, dict[str, float]] = collections.defaultdict(
        lambda: {"worst": 0.0, "compared": 0, "skipped": 0})
    # Строки, которые этот слой не обрабатывает ВООБЩЕ (иная волна сверки):
    # они не относятся ни к одной проверке, поэтому учитываются отдельно и
    # печатаются явно — чтобы их количество нельзя было потерять из вида.
    out_of_scope: dict[str, int] = collections.Counter()
    for equation, problem, quantity, t_text, value_text in rows:
        t = float(t_text)
        value = float(value_text)
        key = f"{equation}/{problem}/{quantity}"
        if equation == "F" and problem in FREDHOLM_PROBLEMS:
            spec = FREDHOLM_PROBLEMS[problem]
            exact = spec["exact"]
            # Правая часть модельной задачи: f = u - Ku, значит f^(k) = u^(k) - (Ku)^(k).
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
                    # Вырожденный отрезок интегрирования: сравнивать нечего.
                    stats[key]["skipped"] += 1
                    continue
                ref = volterra_image_deriv(spec, t, lower, 0)
            elif quantity == "rhs":
                ref = float(exact(t)) - volterra_image_deriv(spec, t, lower, 0)
            elif quantity == "rhsDeriv":
                # При t = a интеграл вырождается, НО граничный вклад K(a,a)u(a) остаётся
                # осмысленным, поэтому точка НЕ пропускается: именно она проверяет
                # член Лейбница в изоляции от интегрального слагаемого.
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
        # НЕ тихая заметка, а ПРОВАЛ — точно так же, как в L6b для строк без порога.
        # Строка выгрузки, не попавшая ни в одну проверку, — это НЕСВЕРЕННЫЙ
        # результат. Фактически проверено: добавление в дампер задачи F2span давало
        # 84 несверенные строки, одну печатную строку и КОД ВОЗВРАТА 0 — новая задача
        # тихо оставалась без сверки. Теперь либо сверяется, либо роняет сверку.
        listing = ", ".join(f"{name}: {count}" for name, count in sorted(out_of_scope.items()))
        print(f"  [РАСХОЖДЕНИЕ] строки вне обрабатываемого набора слоя: {listing}")
        failures.append(
            f"L4/L5 / полнота обработки: в operator-images.tsv есть строки, не попавшие ни в одну "
            f"проверку ({listing}). Либо добавьте задачу/величину в FREDHOLM_PROBLEMS / "
            f"VOLTERRA_PROBLEMS и соответствующую ветку сверки, либо уберите из выгрузки: "
            f"выгруженный, но не сверенный результат создаёт видимость проверки"
        )
    for name in sorted(stats):
        entry = stats[name]
        report("L4/L5", name, entry["worst"], TOL_INTEGRAL,
               compared=int(entry["compared"]), skipped=int(entry["skipped"]))


# ---------------------------------------------------------------------------
# L6. Итоговые решения: эталонный Nystrom на квадратуре Гаусса--Лежандра
# ---------------------------------------------------------------------------
def reference_nystrom(kernel, rhs, lower: float, upper: float, node_count: int = 80):
    """Классический метод Nystrom (Atkinson 1997, гл. 4) на [lower, upper].

    Ни сплайнов, ни аппроксимационных функционалов: узлы квадратуры, матрица
    (I - w_j K(t_i,t_j)) и решение СЛАУ. На гладких задачах даёт точность порядка
    1e-14, поэтому служит внешним эталоном.

    Границы передаются ЯВНО (без значений по умолчанию): они берутся из
    метаданных выгрузки, а зашитый отрезок делал бы согласие с дампером
    случайным совпадением.
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
    """L6a. ПРИГОДНОСТЬ ЭТАЛОНА — самопроверка оракула, НЕ сверка проекта.

    Здесь не читается НИ ОДИН артефакт проекта, и это намеренно: слой отвечает на
    вопрос «годится ли эталон в качестве эталона». Если независимый Nystrom не
    воспроизводит точное решение, сверка им (L6b) лишена смысла.

    Раньше эта логика называлась просто «L6. Итоговые решения» и создавала
    впечатление, будто решения проекта сверены. Название разделено именно
    потому, что вводящее в заблуждение имя само по себе создаёт ложную гарантию.
    """
    print("\nL6a. Пригодность эталона: Nystrom (80 узлов leggauss) против точного решения")
    lower, upper = interval()
    sample = np.linspace(lower, upper, 51)
    for name, spec in FREDHOLM_PROBLEMS.items():
        kernel, exact = spec["kernel"], spec["exact"]

        def rhs(t: float) -> float:
            integral, _ = quad(lambda s: kernel(t, s) * exact(s), lower, upper, limit=200)
            return exact(t) - integral

        approximate = reference_nystrom(kernel, rhs, lower, upper)
        deviation = worst_over(abs(approximate(t) - exact(t)) for t in sample)
        # Эталон обязан воспроизводить точное решение: иначе сверять им нельзя.
        report("L6a", f"{name}: эталонный Nystrom против точного решения",
               deviation, TOL_REFERENCE_FITNESS, compared=int(len(sample)), skipped=0)


# ПОРОГИ ДЛЯ E_h ПРОЕКТА (слой L6b), ключ (problem, system, scheme) -> (n=8, n=16, n=32).
#
# Откуда взяты. Из ФАКТИЧЕСКОГО прогона (`solution-errors.tsv`, выгрузка
# `VerificationArtifacts.dumpSolutionErrors`), удвоением наблюдавшегося значения.
#
# Почему ИМЕННО двойной запас, а не «порядок величины». При запасе x10
# деградация на один порядок сходимости прошла бы незамеченной: при n = 8
# переход с O(h^3) на O(h^2) даёт рост всего в 8 раз. Множитель 2 ловит такое
# изменение на ЛЮБОЙ из трёх сеток и при этом с запасом перекрывает разброс
# между платформами (порядка ulp: величины детерминированы с точностью
# бит-в-бит, что подтверждается эталонным снимком `baseline-eh.tsv`).
#
# ОСОБЫЙ СЛУЧАЙ F2exp/H. Там E_h порядка 1e-13...1e-11 и РАСТЁТ с n — это не
# дефект: решение u = e^t лежит В ПОРОЖДАЮЩЕМ ПРОСТРАНСТВЕ гиперболической
# системы H, поэтому погрешность аппроксимации тождественно нулевая, и остаётся
# только округление, растущее с размером СЛАУ. Поэтому строка исключена из проверки
# порядка сходимости (порядка там нет и быть не может), но ПОРОГ взят ПО ТОМУ ЖЕ
# ПРАВИЛУ — удвоением факта на каждой сетке отдельно. Раньше здесь стоял единый
# потолок 1e-9 — запас в 60...600 раз, то есть порог почти ничего не ограничивал:
# утрата свойства «решение в span» с выходом на 1e-10 прошла бы незамеченной.
# Опасение «порог на шум округления ломкий» оказалось необоснованным: величины
# детерминированы бит-в-бит (это ежепрогонно подтверждает `baseline-eh.tsv`),
# поэтому запаса x2 достаточно и здесь.
EH_LIMITS: dict[tuple[str, str, str], tuple[float, float, float]] = {
    ("F2", "B", "base"): (2.029e-04, 2.493e-05, 3.042e-06),
    ("F2", "B", "sloan"): (9.615e-06, 4.993e-07, 2.503e-08),
    ("F2", "H", "base"): (1.694e-04, 2.086e-05, 2.543e-06),
    ("F2", "H", "sloan"): (8.619e-06, 4.539e-07, 2.289e-08),
    ("F2", "T", "base"): (2.364e-04, 2.899e-05, 3.541e-06),
    ("F2", "T", "sloan"): (1.061e-05, 5.447e-07, 2.716e-08),
    ("F2exp", "B", "base"): (9.815e-05, 1.131e-05, 1.368e-06),
    ("F2exp", "B", "sloan"): (1.282e-05, 6.386e-07, 3.499e-08),
    # Решение в span системы H: остаётся только округление (см. выше).
    ("F2exp", "H", "base"): (1.452e-12, 1.443e-11, 3.417e-11),
    ("F2exp", "H", "sloan"): (2.234e-13, 2.229e-12, 3.144e-12),
    ("F2exp", "T", "base"): (1.964e-04, 2.262e-05, 2.735e-06),
    ("F2exp", "T", "sloan"): (2.564e-05, 1.277e-06, 6.998e-08),
}

# Сетки, в порядке колонок [EH_LIMITS].
EH_GRID_SIZES = (8, 16, 32)

# Минимальный наблюдаемый порядок сходимости p = log2(E_h(n) / E_h(2n)).
#
# Теория даёт p = 3 для базовой схемы и p = 4 для итерации Слоана;
# фактически наблюдается 3.02...3.12 и 4.19...4.33. Порог 2.5 пропускает штатный
# разброс, но заведомо ловит падение порядка до 2 — главный симптом утраты
# свойств минимальных сплайнов. Строки, где погрешность определяется округлением
# (F2exp/H), из проверки порядка исключены: порядка там нет и быть не может.
MIN_OBSERVED_ORDER = 2.5

# Порог, ниже которого E_h считается определяемым округлением, а не аппроксимацией.
EH_ROUNDOFF_FLOOR = 1e-9


def verify_project_solutions() -> None:
    """L6b. СВЕРКА РЕШЕНИЙ ПРОЕКТА: `solution-errors.tsv` против независимого Nystrom.

    До этого слоя выгруженный `solution-errors.tsv` не читался НИГДЕ: слой L6
    сравнивал оракул с точным решением и ни одного числа проекта не трогал.

    Проверяемое утверждение (три части, каждая осмысленна ОТДЕЛЬНО):

      (1) СОГЛАСИЕ С ЭТАЛОНОМ. Независимый Nystrom воспроизводит точное решение
          на уровне 1e-15 (это установлено слоем L6a), поэтому отклонение схемы
          проекта ОТ ЭТАЛОНА практически равно её отклонению от точного решения.
          Значит, выгруженное E_h обязано укладываться в [EH_LIMITS].
      (2) СХОДИМОСТЬ. E_h обязано убывать с ростом n с порядком не ниже
          [MIN_OBSERVED_ORDER]. Одних порогов недостаточно: схема, застрявшая на
          уровне точности n = 8, всё ещё прошла бы порог для n = 8.
      (3) ПОЛНОТА. Каждая строка выгрузки обязана иметь порог: неопознанная строка
          — это либо новая схема без сверки, либо опечатка; молча пропускать нельзя.
    """
    print("\nL6b. Решения ПРОЕКТА: E_h из solution-errors.tsv против независимого Nystrom")
    rows = load_tsv("solution-errors.tsv")
    if not rows:
        return
    lower, upper = interval()

    # Отклонение самого эталона от точного решения — поправка, которую НЕЛЬЗЯ
    # считать нулём без измерения: если она вдруг станет сравнима с E_h проекта,
    # сверка потеряет смысл, и это обнаружится здесь, а не останется незамеченным.
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
        # Отклонение выражается ОТНОСИТЕЛЬНО порога: так одна проверка
        # объединяет три сетки с разными масштабами E_h, а допуск остаётся 1.0.
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
            # Поправка на погрешность самого эталона (см. выше).
            # worst_of, а не встроенный max: NaN в выгрузке иначе был бы отброшен
            # и пороговая проверка осталась бы зелёной (проверено мутацией).
            worst_ratio = worst_of(worst_ratio, eh / (limit + reference_error[problem]))
            compared += 1
            details.append(f"n={n}: {eh:.3e} <= {limit:.3e}")
        report("L6b", f"{name}: E_h проекта против порогов эталона", worst_ratio, 1.0,
               compared=compared, skipped=skipped, kind="отн.")

        # НЕХВАТКА СЕТКИ — ПРОВАЛ, А НЕ ПРОПУСК. Без этого исчезновение строк
        # n = 32 из выгрузки оставляло проверку ЗЕЛЁНОЙ с compared=2, skipped=1: точность
        # на самой мелкой сетке (где требования строже всего) не проверялась вовсе.
        if compared != len(EH_GRID_SIZES):
            missing = [n for n in EH_GRID_SIZES if n not in values]
            failures.append(
                f"L6b / {name}: в solution-errors.tsv нет сеток {missing} — сверено {compared} из "
                f"{len(EH_GRID_SIZES)}. Неполная сверка НЕ есть сверка: отсутствие самой мелкой "
                f"сетки скрывает именно те отклонения, к которым требования строже всего"
            )
            print(f"  [РАСХОЖДЕНИЕ] {name}: не выгружены сетки {missing}")

        # Часть (2): наблюдаемый порядок сходимости. Осмыслена только там, где
        # погрешность определяется аппроксимацией, а не округлением (см. F2exp/H).
        roundoff_dominated = bool(values) and min(values.values()) <= EH_ROUNDOFF_FLOOR
        if roundoff_dominated:
            continue
        if len(values) != len(EH_GRID_SIZES):
            # ОТСУТСТВИЕ ПРОВЕРКИ ПОРЯДКА — ТОЖЕ ПРОВАЛ. Раньше проверка просто
            # НЕ СОЗДАВАЛАСЬ, и отсутствие доказательства сходимости выглядело как его
            # наличие: в отчёте её просто не было, а суммарный статус оставался зелёным.
            failures.append(
                f"L6b / {name}: проверка порядка сходимости НЕ СОЗДАНА: требуются все сетки "
                f"{list(EH_GRID_SIZES)}, выгружены {sorted(values)}. Погрешность здесь не шум "
                f"округления (min E_h > {EH_ROUNDOFF_FLOOR:.0e}), значит сходимость ОБЯЗАНА быть "
                f"проверена, а одни пороги пропустили бы схему, застрявшую на точности n = 8"
            )
            print(f"  [РАСХОЖДЕНИЕ] {name}: проверка порядка невозможна, выгружены только {sorted(values)}")
            continue
        orders = []
        for coarse, fine in zip(EH_GRID_SIZES, EH_GRID_SIZES[1:]):
            ratio = values[coarse] / values[fine]
            # NaN/0 в выгрузке — нефинитный порядок; он передаётся в report КАК ЕСТЬ.
            orders.append(math.log2(ratio) if ratio > 0.0 else float("nan"))
        # Минимум с ОБРАТНЫМ приоритетом NaN: встроенный `min` теряет NaN, если тот
        # пришёл вторым аргументом, и нефинитность проскользнула бы мимо проверки.
        worst_order = float("nan") if any(not math.isfinite(p) for p in orders) else min(orders)
        # Отклонение = насколько порядок НЕ ДОТЯНУЛ до требуемого (0 = в норме).
        # Нефинитный порядок (E_h = 0 или NaN в выгрузке) проходит в report как есть и
        # превращается в отказ там: глотать его здесь значило бы скрыть причину.
        deficit = MIN_OBSERVED_ORDER - worst_order
        report("L6b", f"{name}: порядок сходимости (наблюдаемый min p = {worst_order:.2f})",
               deficit if not math.isfinite(deficit) else max(0.0, deficit), 0.0,
               compared=len(orders), skipped=0, kind="дефицит p,")

    if unknown:
        # НЕ тихий пропуск: неопознанная строка — это несверенный результат.
        failures.append(
            f"L6b / полнота таблицы порогов: в solution-errors.tsv есть строки без порога "
            f"({len(unknown)}): {sorted(set(unknown))}. Добавьте их в EH_LIMITS — иначе результат "
            f"выгружается, но не сверяется"
        )
        print(f"  [РАСХОЖДЕНИЕ] строки без порога: {sorted(set(unknown))}")


def main() -> int:
    global ARTIFACT_DIR
    parser = argparse.ArgumentParser(description="Внешняя сверка со SciPy/NumPy")
    parser.add_argument("--artifacts", default=ARTIFACT_DIR,
                        help="каталог с выгруженными артефактами")
    parser.add_argument("--json", default=None,
                        help="файл для машиночитаемого результата (для Kotlin-теста)")
    args = parser.parse_args()
    ARTIFACT_DIR = args.artifacts

    print("=" * 78)
    print("ВНЕШНЯЯ СВЕРКА СО SciPy/NumPy")
    print("=" * 78)
    print(f"NumPy {np.__version__}")
    import scipy
    print(f"SciPy {scipy.__version__}")
    print(f"Python {platform.python_version()}")
    print(f"Каталог артефактов: {os.path.abspath(ARTIFACT_DIR)}")

    # Метаданные выгрузки читаются ДО первой проверки и без значений по
    # умолчанию: без них неизвестен отрезок, а значит, и все интегралы.
    try:
        META.update(load_meta())
    except MetaError as exc:
        print(f"ОШИБКА: {exc}")
        return 4
    print(
        f"Конфигурация выгрузки: отрезок [{META['a']:.17g}, {META['b']:.17g}], "
        f"сетка n = {int(META['dumpGridSize'])}, узлов квадратуры {int(META['quadratureNodes'])}"
    )

    verify_quadrature()
    verify_linear_algebra()
    verify_spline_basis()
    verify_operator_images()
    verify_reference_fitness()
    verify_project_solutions()

    print("\n" + "=" * 78)
    if notes:
        print("ЗАМЕЧАНИЯ:")
        for note in notes:
            print(f"  - {note}")

    exit_code = 1 if failures else 0
    if failures:
        print(f"ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        print("\nРасхождение НЕ следует устранять ослаблением допуска: сначала причина.")
    else:
        print("ВСЕ СЛОИ СОШЛИСЬ: расхождений со SciPy/NumPy не обнаружено.")

    # Машиночитаемый итог. Пишется ВСЕГДА, включая случай расхождений:
    # именно тогда он нужнее всего — чтобы тест назвал конкретные слои и числа.
    if args.json:
        summary = {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            # Окружение целиком — чтобы расхождение можно было соотнести с версиями
            # библиотек по одному отчёту, не восстанавливая обстановку прогона.
            # Порядок полей важен: скалярные "numpy"/"scipy" выше читает Kotlin-тест
            # построчным разбором, и они обязаны встречаться в тексте первыми.
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
        # Сначала сериализуем ЦЕЛИКОМ в памяти, потом пишем: при ошибке
        # сериализации запись прямо в файл оставила бы обрезанный JSON, который
        # неотличим от «слой не выполнялся» и вводит в заблуждение.
        try:
            serialized = json.dumps(summary, ensure_ascii=False, indent=2)
        except TypeError as exc:
            print(f"ОШИБКА: отчёт не сериализуется ({exc}).")
            return 3
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        print(f"Машиночитаемый результат: {os.path.abspath(args.json)}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
