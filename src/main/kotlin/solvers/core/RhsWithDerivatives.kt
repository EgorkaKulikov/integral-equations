package solvers.core

/**
 * The right-hand side `f` of the equation together with its two derivatives.
 *
 * The triple `f`, `f'`, `f''` NEVER travels apart: the functional families
 * `chi_j` accept exactly it (`ApproxFunctional.apply(f, fD, fDD)`), and which
 * derivatives are actually read is decided by the FAMILY, not by the solver — `theta`
 * reads none, `xi` reads the first one, `xi^<0>` reads both. Passing them as
 * three independent parameters would therefore allow a state that does not
 * exist: the derivative of ONE function next to the value of ANOTHER. Such a
 * desynchronization is not diagnosed in any way — it simply yields a wrong system
 * (exactly this defect has already happened with `f''`, which was forgotten on the way through, and the
 * family `xi^<0>` silently received zero).
 *
 * ### Why a plain class and not a `data class`
 *
 * The fields are lambdas, whose `equals` is a reference comparison, so the automatic
 * `equals`/`hashCode` would carry no meaning, while `toString` would print the names
 * of synthetic classes. None of these members is needed.
 *
 * ### Hot path
 *
 * The object is created ONCE per solver, and [SecondKindSolverCore] immediately unpacks
 * it into three of its own fields. Hence the matrix assembly loops and the iterations keep
 * the same single field dereference followed by an `invoke` as before: no allocations
 * and no extra level of indirection are added.
 *
 * @param value the right-hand side `f(t)` itself.
 * @param deriv the first derivative `f'(t)`.
 * @param deriv2 the second derivative `f''(t)`. The default value `{ 0.0 }`
 *        is kept from the previous signature: it is used by callers working with
 *        families that do not need the second derivative.
 */
public class RhsWithDerivatives(
    public val value: (Double) -> Double,
    public val deriv: (Double) -> Double,
    public val deriv2: (Double) -> Double = { 0.0 },
)
