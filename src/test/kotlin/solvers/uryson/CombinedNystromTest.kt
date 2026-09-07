package solvers.uryson

import numerics.GaussLegendre
import splines.GeneratingSystem
import splines.Grid
import splines.MinimalSplineBasis
import splines.functionals.ProjFunctionals
import splines.metrics.errorEh
import org.junit.jupiter.api.Tag
import problems.uryson.NoiseNorm
import problems.uryson.UrysonProblem
import problems.uryson.noisyRightHandSide
import problems.uryson.secondKindSolver
import solvers.core.RhsWithDerivatives
import solvers.fredholm.FredholmOperator
import solvers.fredholm.FredholmSecondKindSolver
import solvers.fredholm.KernelF
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Проверки комбинированного метода Nyström для уравнения Урысона и нормировки шума.
 *
 *  1. Для ядра, ЛИНЕЙНОГО по `u` (`K(t,s,u) = k(t,s) u`), нелинейный комбинированный
 *     Nyström обязан совпасть с линейным [FredholmSecondKindSolver.combinedNystrom]
 *     на той же сетке и базисе: обе реализации решают одно и то же конечномерное
 *     уравнение `u = f + L_n u`, различаясь лишь способом решения (Ньютон против
 *     простой итерации). Ядро выбрано сжимающим, чтобы простая итерация линейного
 *     решателя сходилась.
 *  2. На задаче B (`u* = e^t`) с гиперболическим базисом решение лежит в сплайновом
 *     пространстве, и комбинированный Nyström воспроизводит его с точностью округления.
 *  3. Нормировка шума `SUP` даёт `max |xi| = delta` точно.
 */
@Tag("fast")
class CombinedNystromTest {

    private val quad = GaussLegendre(8)

    @Test
    fun linearKernelMatchesFredholmCombinedNystrom() {
        // k(t,s) = 0.5 / (1 + t + s): ||L|| <= 0.5 * ln 3 < 1, простая итерация Фредгольма сходится.
        val kLin = { t: Double, s: Double -> 0.5 / (1.0 + t + s) }
        val exact = { t: Double -> 1.0 / (t + 1.0) }
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
            val funcs = ProjFunctionals(basis)

            // Линейный решатель: u - L u = f, cL = 1.
            val opF = FredholmOperator(KernelF(k = kLin), grid, quad)
            val rhsF = { t: Double -> exact(t) - opF.apply(t) { s -> exact(s) } }
            val fredholm = FredholmSecondKindSolver(
                basis, funcs, opF, cL = 1.0,
                rhs = RhsWithDerivatives(value = rhsF, deriv = { 0.0 }),
            )
            val uF = fredholm.combinedNystrom()

            // Нелинейный решатель с тем же ядром, записанным как K(t,s,u) = k(t,s) u, lambda = 1.
            val kernel = object : Kernel {
                override fun k(t: Double, s: Double, u: Double) = kLin(t, s) * u
                override fun dkdu(t: Double, s: Double, u: Double) = kLin(t, s)
            }
            val problem = UrysonProblem("lin", kernel, lambda = 1.0, exact = exact, secondKind = true)
            val opU = UrysohnOperator(kernel, grid, quad)
            val space = SplineSpace(basis, quad)
            val uryson = secondKindSolver(problem, basis, funcs, space, opU)
            val uU = uryson.combinedNystrom()
            assertTrue(uU.converged, "Ньютон комбинированного Nyström не сошёлся, n=$n")

            var diff = 0.0
            val m = 100 * n
            for (i in 0..m) {
                val t = i.toDouble() / m
                diff = maxOf(diff, abs(uU.eval(t) - uF.eval(t)))
            }
            assertTrue(diff < 1e-10, "n=$n: расхождение с FredholmSolver.combinedNystrom = $diff")
        }
    }

    @Test
    fun problemBInHyperbolicSpanIsReproducedToRoundoff() {
        for (n in listOf(8, 16)) {
            val grid = Grid.uniform(n)
            val basis = MinimalSplineBasis(GeneratingSystem.H, grid)
            val funcs = ProjFunctionals(basis)
            val space = SplineSpace(basis, quad)
            val op = UrysohnOperator(UrysonProblem.B.kernel, grid, quad)
            val solver = secondKindSolver(UrysonProblem.B, basis, funcs, space, op)
            val exact = { t: Double -> UrysonProblem.B.exact(t) }
            val eComb = errorEh(exact, solver.combinedNystrom().eval, grid)
            val eIter = errorEh(exact, solver.iteratedNystrom().eval, grid)
            assertTrue(eComb < 1e-9, "n=$n: E_h(комб. Nyström) = $eComb")
            assertTrue(eIter < 1e-9, "n=$n: E_h(итер. Nyström) = $eIter")
        }
    }

    @Test
    fun iteratedKulkarniIsNotWorseThanKulkarniOnProblemA() {
        val grid = Grid.uniform(16)
        val basis = MinimalSplineBasis(GeneratingSystem.B, grid)
        val funcs = ProjFunctionals(basis)
        val space = SplineSpace(basis, quad)
        val op = UrysohnOperator(UrysonProblem.A.kernel, grid, quad)
        val solver = secondKindSolver(UrysonProblem.A, basis, funcs, space, op)
        val exact = { t: Double -> UrysonProblem.A.exact(t) }
        val eK = errorEh(exact, solver.kulkarni().eval, grid)
        val eIK = errorEh(exact, solver.iteratedKulkarni().eval, grid)
        assertTrue(eIK <= eK, "итер. Кулкарни $eIK хуже Кулкарни $eK")
    }

    @Test
    fun supNoiseHasExactSupNorm() {
        val grid = Grid.uniform(16)
        val delta = 1e-3
        val zero = { _: Double -> 0.0 }
        val noisy = noisyRightHandSide(zero, grid, quad, delta, seed = 20240517L, norm = NoiseNorm.SUP)
        // Профиль кусочно-линейный с узлами на сетке 4n: контрольная сетка 100n содержит их.
        val m = 100 * grid.n
        var maxAbs = 0.0
        for (i in 0..m) maxAbs = maxOf(maxAbs, abs(noisy(i.toDouble() / m)))
        assertTrue(abs(maxAbs - delta) < 1e-12, "max|xi| = $maxAbs, ожидалось $delta")
        // L2-вариант при том же seed отличается лишь множителем: профили пропорциональны.
        val noisyL2 = noisyRightHandSide(zero, grid, quad, delta, seed = 20240517L, norm = NoiseNorm.L2)
        val t1 = 0.31; val t2 = 0.77
        val ratio1 = noisy(t1) / noisyL2(t1)
        val ratio2 = noisy(t2) / noisyL2(t2)
        assertTrue(abs(ratio1 - ratio2) < 1e-12 * abs(ratio1), "профили SUP и L2 не пропорциональны")
    }
}
