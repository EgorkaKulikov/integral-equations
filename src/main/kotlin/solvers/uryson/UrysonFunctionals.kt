package solvers.uryson

import splines.functionals.ApproxFunctional
import splines.functionals.ProjFunctionals
import splines.functionals.ValueFunctional

/**
 * Applies a functional to a value function, passing zero derivatives explicitly.
 *
 * The projection functionals `theta_j` do not use derivatives, but the generic interface
 * [ApproxFunctional] accepts them. A dedicated name (instead of `apply`) is chosen to rule
 * out ambiguity with the scope function of the same name from the Kotlin standard
 * library.
 */
internal fun ApproxFunctional.applyTo(f: (Double) -> Double): Double = apply(f, { 0.0 }, { 0.0 })

/**
 * Returns the functional `theta_j` as a linear combination of values.
 *
 * The Uryson schemes (assembly of `Xi`, of the Jacobian and the Nyström method) work directly
 * with the support points and coefficients of the functional, so exactly a [ValueFunctional]
 * is required. The `theta` family consists of such functionals by construction.
 */
internal fun ProjFunctionals.valueFunctional(j: Int): ValueFunctional =
    chi(j) as? ValueFunctional
        ?: error("Functional theta_$j is not a linear combination of values (ValueFunctional).")
