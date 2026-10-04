package problems.wsie

import numerics.SpecialFunctions
import solvers.wsie.KernelWS
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

/** Kind of a weakly singular equation: upper limit of integration `t` (Volterra) or `b` (Fredholm). */
enum class WeaklySingularType { VOLTERRA, FREDHOLM }

/**
 * Model problem for the linear weakly singular equation of the second kind on `[0, 1]`
 *
 *     u(t) − cL · ∫ |t − s|^(−alpha) k(t, s) u(s) ds = f(t),
 *
 * where the integral runs over `[0, t]` for [WeaklySingularType.VOLTERRA] and over `[0, 1]` for
 * [WeaklySingularType.FREDHOLM].
 *
 * Unlike [problems.volterra.VolterraProblem], the right-hand side is given analytically (closed form or a
 * convergent series): the exact solutions are non-smooth at the ends of the interval, and a right-hand side built
 * by the quadrature of the solver would hide exactly the quadrature error that these problems are meant to expose.
 *
 * @param name short problem name used in tables and test messages; serves as the problem id.
 * @param alpha exponent of the singularity, `0 < alpha < 1`.
 * @param k smooth factor of the kernel.
 * @param cL coefficient λ in front of the integral operator.
 * @param rhs right-hand side `f(t)`.
 * @param exact exact solution `u*(t)` — the baseline for computing the error.
 * @param type Volterra (upper limit `t`) or Fredholm (upper limit `1`).
 * @param description one-line statement of the problem.
 */
class WeaklySingularProblem(
    val name: String,
    val alpha: Double,
    val k: (Double, Double) -> Double,
    val cL: Double,
    val rhs: (Double) -> Double,
    val exact: (Double) -> Double,
    val type: WeaklySingularType,
    val description: String,
) {
    /** The kernel `|t − s|^(−alpha) k(t, s)` in the form accepted by the operators of `solvers.wsie`. */
    val kernel: KernelWS get() = KernelWS(alpha, k)

    companion object {
        private val ONE: (Double, Double) -> Double = { _, _ -> 1.0 }

        /** Smooth factor `k(t, s) = 1 + t s` of [V_D] and [F_B]. */
        private val LINEAR_TS: (Double, Double) -> Double = { t, s -> 1.0 + t * s }

        /**
         * Abel equation with `alpha = 1/2` and unit right-hand side. By the Laplace transform,
         * `U(p) = p^(−1) / (1 − Γ(1/2) p^(−1/2))`, hence `u(t) = Σ_k (π t)^(k/2) / Γ(k/2 + 1) = E_{1/2}(√(π t))`.
         * The solution behaves like `1 + 2√t` at `t = 0`.
         */
        val V_A = WeaklySingularProblem(
            name = "V-a",
            alpha = 0.5,
            k = ONE,
            cL = 1.0,
            rhs = { 1.0 },
            exact = { t -> SpecialFunctions.mittagLeffler(0.5, sqrt(PI * max(t, 0.0))) },
            type = WeaklySingularType.VOLTERRA,
            description = "u − ∫_0^t (t − s)^(−1/2) u(s) ds = 1, u = E_{1/2}(√(π t))",
        )

        /** Smooth solution `u = cos t` under the Abel kernel; the right-hand side `cos t − (𝓛 cos)(t)` is non-smooth. */
        val V_B = WeaklySingularProblem(
            name = "V-b",
            alpha = 0.5,
            k = ONE,
            cL = 1.0,
            rhs = { t -> cos(t) - abelCosine(t) },
            exact = { t -> cos(t) },
            type = WeaklySingularType.VOLTERRA,
            description = "u − ∫_0^t (t − s)^(−1/2) u(s) ds = f, u = cos t",
        )

        /**
         * Equation with `alpha = 1/3` and unit right-hand side. As for [V_A], `U(p) = p^(−1) / (1 − Γ(2/3) p^(−2/3))`,
         * hence `u(t) = Σ_k (Γ(2/3) t^(2/3))^k / Γ(2k/3 + 1) = E_{2/3}(Γ(2/3) t^(2/3))`.
         */
        val V_C = WeaklySingularProblem(
            name = "V-c",
            alpha = 1.0 / 3.0,
            k = ONE,
            cL = 1.0,
            rhs = { 1.0 },
            exact = { t ->
                SpecialFunctions.mittagLeffler(2.0 / 3.0, SpecialFunctions.gamma(2.0 / 3.0) * max(t, 0.0).pow(2.0 / 3.0))
            },
            type = WeaklySingularType.VOLTERRA,
            description = "u − ∫_0^t (t − s)^(−1/3) u(s) ds = 1, u = E_{2/3}(Γ(2/3) t^(2/3))",
        )

        /**
         * Fredholm equation with `alpha = 1/2`, `λ = 0.2` and a solution with square-root singularities at both ends.
         * The image `(𝓛u)(t) = ∫_0^1 |t − s|^(−1/2) u(s) ds` is taken in closed form term by term:
         * `∫_0^1 |t − s|^(−1/2) ds = 2(√t + √(1 − t))`, the √s term is [fredholmSqrtImage] at `t`, and the √(1 − s) term
         * is the same function at `1 − t` (substitution `s → 1 − s`).
         */
        val F_A = WeaklySingularProblem(
            name = "F-a",
            alpha = 0.5,
            k = ONE,
            cL = 0.2,
            rhs = { t -> faExact(t) - 0.2 * faImage(t) },
            exact = ::faExact,
            type = WeaklySingularType.FREDHOLM,
            description = "u − 0.2 ∫_0^1 |t − s|^(−1/2) u(s) ds = f, u = 1 + √t + √(1 − t)",
        )

        /**
         * Volterra equation with `alpha = 1/2`, the non-constant smooth factor `k(t, s) = 1 + t s` and the solution
         * `u = 1 + √t` with a square-root singularity at `t = 0`. Expanding `(1 + t s)(1 + √s) = 1 + s^(1/2) + t s +
         * t s^(3/2)` and integrating term by term with `∫_0^t (t − s)^(−1/2) s^β ds = t^(β+1/2) B(1/2, β + 1)`,
         *
         *     (𝓥u)(t) = t^(1/2) B(1/2, 1) + t B(1/2, 3/2) + t^(5/2) B(1/2, 2) + t^3 B(1/2, 5/2)
         *             = 2√t + (π/2) t + (4/3) t^(5/2) + (3π/8) t^3,
         *
         * and `f = u − 𝓥u` ([vdImage]).
         */
        val V_D = WeaklySingularProblem(
            name = "V-d",
            alpha = 0.5,
            k = LINEAR_TS,
            cL = 1.0,
            rhs = { t -> vdExact(t) - vdImage(t) },
            exact = ::vdExact,
            type = WeaklySingularType.VOLTERRA,
            description = "u − ∫_0^t (t − s)^(−1/2) (1 + t s) u(s) ds = f, u = 1 + √t",
        )

        /**
         * Fredholm equation with `alpha = 1/2`, `λ = 0.2`, the non-constant smooth factor `k(t, s) = 1 + t s` and the
         * solution `u = 1 + √t + √(1 − t)` of [F_A]. With `J_β(t) = ∫_0^1 |t − s|^(−1/2) s^β ds`,
         *
         *     (𝓛u)(t) = ∫_0^1 |t − s|^(−1/2) u(s) ds + t ∫_0^1 |t − s|^(−1/2) s u(s) ds
         *             = [faImage](t) + t (J_1(t) + J_{3/2}(t) + J_{1/2}(1 − t) − J_{3/2}(1 − t)),
         *
         * where `s u(s) = s + s^(3/2) + s √(1 − s)`, and the last two terms come from `s √(1 − s) = √(1 − s) −
         * (1 − s)^(3/2)` and the substitution `s → 1 − s`. `J_{1/2}`, `J_1`, `J_{3/2}` are [fredholmSqrtImage],
         * [fredholmLinearImage], [fredholmThreeHalvesImage]. The maximum over `t` of `∫_0^1 |t − s|^(−1/2) (1 + t s) ds
         * = 2(√t + √(1 − t)) + t J_1(t)` is ≈ 4.10, so `‖λ𝓛‖ ≈ 0.82 < 1` in `C[0, 1]`.
         */
        val F_B = WeaklySingularProblem(
            name = "F-b",
            alpha = 0.5,
            k = LINEAR_TS,
            cL = 0.2,
            rhs = { t -> faExact(t) - 0.2 * fbImage(t) },
            exact = ::faExact,
            type = WeaklySingularType.FREDHOLM,
            description = "u − 0.2 ∫_0^1 |t − s|^(−1/2) (1 + t s) u(s) ds = f, u = 1 + √t + √(1 − t)",
        )

        /** All weakly singular model problems, in the order V-a, V-b, V-c, V-d, F-a, F-b. */
        val ALL = listOf(V_A, V_B, V_C, V_D, F_A, F_B)

        /**
         * Erfc form of the solution of [V_A]: `E_{1/2}(z) = e^(z²) erfc(−z)` with `z = √(π t)`. Kept as an
         * independent cross-check of the Mittag-Leffler series used in [V_A].
         */
        fun abelHalfErfcForm(t: Double): Double {
            val tt = max(t, 0.0)
            return exp(PI * tt) * SpecialFunctions.erfc(-sqrt(PI * tt))
        }

        /**
         * `∫_0^t (t − s)^(−1/2) cos s ds = Σ_k (−1)^k t^(2k+1/2) B(1/2, 2k + 1) / (2k)!`, obtained by integrating the
         * cosine series term by term with `∫_0^t (t − s)^(−1/2) s^n ds = t^(n+1/2) B(1/2, n + 1)`.
         *
         * Summed until the modulus of the term drops below 1e-17 of the modulus of the partial sum. For `0 ≤ t ≤ 1`
         * the series is alternating with decreasing terms and the sum lies between `2√t` and `2√t − (8/15) t^(5/2)`,
         * so there is no cancellation; outside `[0, 1]` the function is still correct but loses relative accuracy as
         * `t` grows. Returns 0 for `t ≤ 0`.
         */
        private fun abelCosine(t: Double): Double {
            if (t <= 0.0) return 0.0
            val t2 = t * t
            var power = sqrt(t)
            var factorial = 1.0
            var sum = 0.0
            var k = 0
            while (true) {
                val term = power * SpecialFunctions.beta(0.5, 2.0 * k + 1.0) / factorial
                sum += if (k % 2 == 0) term else -term
                if (term < 1e-17 * abs(sum) || k >= MAX_SERIES_TERMS) return sum
                k++
                power *= t2
                factorial *= (2.0 * k - 1.0) * (2.0 * k)
            }
        }

        private const val MAX_SERIES_TERMS = 200

        private fun faExact(t: Double): Double = 1.0 + sqrt(max(t, 0.0)) + sqrt(max(1.0 - t, 0.0))

        /**
         * `∫_0^1 |t − s|^(−1/2) √s ds = π t/2 + √(1 − t) + t ln((1 + √(1 − t))/√t)` for `0 < t ≤ 1`; the last term
         * tends to 0 as `t → 0`, which gives the value 1 at `t = 0`.
         */
        private fun fredholmSqrtImage(t: Double): Double {
            val c = sqrt(max(1.0 - t, 0.0))
            return PI * t / 2 + c + if (t > 0.0) t * ln((1.0 + c) / sqrt(t)) else 0.0
        }

        private fun faImage(t: Double): Double =
            2.0 * (sqrt(max(t, 0.0)) + sqrt(max(1.0 - t, 0.0))) + fredholmSqrtImage(t) + fredholmSqrtImage(1.0 - t)

        private fun vdExact(t: Double): Double = 1.0 + sqrt(max(t, 0.0))

        /** `∫_0^t (t − s)^(−1/2) (1 + t s)(1 + √s) ds` in the Beta form of [V_D]; 0 for `t ≤ 0`. */
        private fun vdImage(t: Double): Double {
            if (t <= 0.0) return 0.0
            val r = sqrt(t)
            return r * SpecialFunctions.beta(0.5, 1.0) + t * SpecialFunctions.beta(0.5, 1.5) +
                t * t * r * SpecialFunctions.beta(0.5, 2.0) + t * t * t * SpecialFunctions.beta(0.5, 2.5)
        }

        /**
         * `J_1(t) = ∫_0^1 |t − s|^(−1/2) s ds = 2t(√t + √(1 − t)) + (2/3)((1 − t)^(3/2) − t^(3/2))`, from
         * `s = t + (s − t)`.
         */
        private fun fredholmLinearImage(t: Double): Double {
            val a = sqrt(max(t, 0.0))
            val c = sqrt(max(1.0 - t, 0.0))
            return 2.0 * t * (a + c) + 2.0 / 3.0 * (c * c * c - a * a * a)
        }

        /**
         * `J_{3/2}(t) = ∫_0^1 |t − s|^(−1/2) s^(3/2) ds = t J_{1/2}(t) + ((2 − t)/4) √(1 − t)
         * − (t²/4) ln((1 + √(1 − t))/√t) − π t²/8`, from `s^(3/2) = t √s + (s − t) √s`:
         * `∫_t^1 √(s(s − t)) ds = ((2 − t)/4) √(1 − t) − (t²/4) ln((1 + √(1 − t))/√t)` and
         * `∫_0^t √(s(t − s)) ds = t² B(3/2, 3/2) = π t²/8`. The logarithmic term tends to 0 as `t → 0`.
         */
        private fun fredholmThreeHalvesImage(t: Double): Double {
            val c = sqrt(max(1.0 - t, 0.0))
            val log = if (t > 0.0) t * t * ln((1.0 + c) / sqrt(t)) else 0.0
            return t * fredholmSqrtImage(t) + (2.0 - t) / 4.0 * c - log / 4.0 - PI * t * t / 8.0
        }

        private fun fbImage(t: Double): Double =
            faImage(t) + t * (
                fredholmLinearImage(t) + fredholmThreeHalvesImage(t) +
                    fredholmSqrtImage(1.0 - t) - fredholmThreeHalvesImage(1.0 - t)
                )
    }
}
