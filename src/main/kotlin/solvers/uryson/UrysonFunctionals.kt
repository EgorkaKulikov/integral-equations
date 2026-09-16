package solvers.uryson

import splines.functionals.ApproxFunctional
import splines.functionals.ProjFunctionals
import splines.functionals.ValueFunctional

/**
 * Применяет функционал к функции значений, явно передавая нулевые производные.
 *
 * Проекционные функционалы `theta_j` производных не используют, но общий интерфейс
 * [ApproxFunctional] их принимает. Отдельное имя (вместо `apply`) выбрано, чтобы
 * исключить неоднозначность с одноимённой функцией-областью видимости из стандартной
 * библиотеки Kotlin.
 */
internal fun ApproxFunctional.applyTo(f: (Double) -> Double): Double = apply(f, { 0.0 }, { 0.0 })

/**
 * Возвращает функционал `theta_j` в виде линейной комбинации значений.
 *
 * Схемы Урысона (сборка `Xi`, якобиана и метод Nyström) работают напрямую с опорными
 * точками и коэффициентами функционала, поэтому нужен именно [ValueFunctional].
 * Семейство `theta` состоит из таких функционалов по построению.
 */
internal fun ProjFunctionals.valueFunctional(j: Int): ValueFunctional =
    chi(j) as? ValueFunctional
        ?: error("Функционал theta_$j не является линейной комбинацией значений (ValueFunctional).")
