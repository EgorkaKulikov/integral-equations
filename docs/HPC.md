# Computational efficiency and HPC readiness

The document describes how the code base is adapted to the requirement of computational
efficiency and of suitability for high-performance computing environments.

## Target regime

Dense matrices, system size `dim = n + 2` (n being the number of grid nodes). The practical
limit on a desktop machine is of the order N ~ 10³: an O(N³) complexity gives about 0.13 s
per full solution at N = 258, a few seconds at N ~ 10³, and already hours at N ~ 10⁴. The
bottlenecks are: (1) assembly of the matrices of the projection-collocation schemes,
(2) dense linear algebra of O(N³).

## Acceleration architecture

Two levels of acceleration — native BLAS for dense linear algebra and multi-core matrix
assembly — are implemented in the library `numerical-core` (`numerics.LinearAlgebra` on top of
multik/OpenBLAS with the reference implementation `numerics.ReferenceLinearAlgebra`;
`numerics.ParallelAssembly` through `IntStream.parallel()`; the backend SPI
`numerics.backend.LinAlgBackend`/`Backends` with an automatic fallback to
`ReferenceBackend` and an extension point for a GPU). Their design, the semantic guarantees and
the selection of a backend by the property `-Dnumerics.backend=multik|reference` are described
in
[`numerical-core/docs/PERFORMANCE.md`](https://github.com/EgorkaKulikov/numerical-core/blob/main/docs/PERFORMANCE.md).

Here, in the solvers, parallel assembly is enabled in the hot spots:
`matrixM`/`matrixM2` (Fredholm, Volterra), the B matrix and the Newton/Kulkarni Jacobians
(Uryson). Deliberately left sequential (as documented in the code): the symmetric assembly of
the Gram matrix (cross writes), the scatter-add accumulation and the finite-difference
Newton–Nyström Jacobian (a shared mutable vector `x`). The parallelism flag arrives through
`NumericsContext.parallel` and does not affect the result: the assembly is bitwise identical
(`solvers.core.NumericsContextWiringTest`).

All observable outputs of the solvers are protected by a characterization test that records
1366 numerical values with a relative tolerance of 1e-9
(`characterization.EhCharacterizationTest`); the baseline was captured on the multik backend
and is bound to it (see `docs/TESTING.md`).

## Benchmark

```
./gradlew runBenchmark
```

It prints two tables:
- (A) scaling: the time of a full solution (construction of the solver, assembly of the matrix,
  solution of the linear system) as a function of N, with the median, the minimum, the maximum
  and the spread;
- (B) scalability of the matrix assembly over 1, 2, 4, 8, ... threads, with the speedup,
  the efficiency and an estimate of the serial fraction by the Karp–Flatt formula.

The number of threads is set EXPLICITLY through a dedicated `ForkJoinPool` rather than taken
from the common pool: otherwise it would not be a controllable parameter of the experiment.
A warm-up is performed before every measurement series. The benchmark prints the configuration
of the environment, including the active linear-algebra backend, and is able to export the
results to CSV.

The current measurements and the caveats about their limitations are given in the README,
section "Performance". An important clarification on the methodology: previously the
construction of the solver was performed before the timer was started, so the precomputation
phase (which dominates at large N) did not enter the measurement and the time of a "full
solution" turned out to be substantially understated.
