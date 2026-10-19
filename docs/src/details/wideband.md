# Wideband Admissibility

## Motivation

The standard low-frequency admissibility condition breaks down when the wavenumber $k$ is large relative to the cluster diameter: blocks that are geometrically far apart are no longer well-approximated by a single low-rank factor because the oscillatory kernel does not decay on the scale of the cluster.

In the wideband / high-frequency regime one instead uses a *directional* admissibility condition [[1, 2]](@ref refs): far interactions are grouped not only by cluster pair but also by the direction of wave propagation, and each directional block remains low-rank.

## Mixed LF / HF admissibility

`isnearwideband` provides a mixed low-frequency / high-frequency admissibility condition.
Small clusters (diameter $\lesssim \lambda$) use the standard LF condition; large clusters (diameter $\gtrsim \lambda$) use the directional HF condition:

```julia
isnear = isnearwideband(k; ηhf=5.0, γ=1.0)
```

- `ηhf` controls the HF admissibility threshold (analogous to $\eta$ in the LF case).
- `γ` controls the transition between LF and HF regimes.

## Directional data and BottomUp

In the HF regime the admissible interactions are stored in a `DirectionalData` structure, which groups far nodes by direction index.
The BottomUp compressor handles `DirectionalData` natively, running a separate iACA for each (cluster, direction) pair.

The cost model in the HF regime is: each active level contributes $|\text{leaves}(T)| \times \bar{d}$ work, where $\bar{d}$ is the average number of directions per cluster at that level.
This is the same $\mathcal{O}(N \log N)$ scaling as a standard wideband fast multipole method.

## References

- [1] Engquist, Björn, and Lexing Ying. *Fast Directional Multilevel Algorithms for Oscillatory Kernels.* SIAM Journal on Scientific Computing 29, no. 4 (2007): 1710–1737. <https://doi.org/10.1137/07068583X>.
- [2] Börm, Steffen. *Directional $\mathcal{H}^2$-Matrix Compression for High-Frequency Problems.* Numerical Linear Algebra with Applications 24, no. 6 (2017). <https://doi.org/10.1002/nla.2112>.

See also the [References](@ref refs) section on the Introduction page.
