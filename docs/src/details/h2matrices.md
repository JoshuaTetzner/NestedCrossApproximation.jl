# H2-Matrices

## Background

Kernel matrices arising in boundary element methods (BEM) have the form

```math
A_{ij} = \int_{\Gamma} \int_{\Gamma} k(x, y)\, \varphi_j(y)\, \psi_i(x)\, \mathrm{d}s(y)\, \mathrm{d}s(x),
```

where $k(x, y)$ is a kernel function (e.g. the Helmholtz Green's function) and $\{\varphi_j\}$, $\{\psi_i\}$ are BEM basis and test functions.
These matrices are dense and require $\mathcal{O}(N^2)$ storage and $\mathcal{O}(N^2)$ work per matrix-vector product — prohibitive for large $N$.

## Admissibility and low-rank approximation

When the supports of $\psi_i$ and $\varphi_j$ are far apart relative to their diameter, $k(x, y)$ is smooth and the corresponding block $A|_{\sigma \times \tau}$ (a sub-matrix of $A$ indexed by test cluster $\sigma$ and trial cluster $\tau$) admits a low-rank approximation:

```math
A|_{\sigma \times \tau} \approx U_\sigma V_\tau^\top, \qquad U_\sigma \in \mathbb{R}^{|\sigma| \times r},\ V_\tau \in \mathbb{R}^{|\tau| \times r}.
```

The condition under which this is exploited is called *admissibility*. A standard (low-frequency) condition is

```math
\max(\mathrm{diam}(\sigma), \mathrm{diam}(\tau)) \leq \eta\, \mathrm{dist}(\sigma, \tau),
```

for a fixed $\eta > 0$.

## Nested structure

In an $\mathcal{H}$-matrix the low-rank factors $U_\sigma$, $V_\tau$ are allowed to differ for every admissible block. $\mathcal{H}^2$-matrices impose a *nested* constraint: the column space of $U_\sigma$ at a cluster must be a subspace of the column space of its parent's factor. This means the bases are shared across levels, reducing storage from $\mathcal{O}(N k \log N)$ to $\mathcal{O}(N k)$, and matrix-vector product complexity from $\mathcal{O}(N k \log N)$ to $\mathcal{O}(N k)$, where $k$ is the local rank.

Concretely, the $\mathcal{H}^2$-matrix format stores:

- **Leaf bases** $V_\ell$ for each leaf cluster $\ell$.
- **Transfer matrices** $T_\sigma$ for each non-leaf cluster $\sigma$, relating the cluster's basis to its children's bases: $U_\sigma = [T_{\sigma,1} U_{\sigma_1}\ |\ T_{\sigma,2} U_{\sigma_2}\ |\cdots]$.
- **Coupling matrices** $S_{\sigma\tau}$ (small, $r \times r$) for each admissible pair $(\sigma, \tau)$.

The matrix-vector product then proceeds as an upward pass (leaf → root), a coupling pass (admissible blocks), and a downward pass (root → leaf).

## References

See the [References](@ref refs) section on the Introduction page.
