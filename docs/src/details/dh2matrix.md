# [Directional $\mathcal{H}^2$-Matrices](@id wideband_theory)

## Oscillatory kernels

Consider again a dense matrix

```math
\boldsymbol{\mathsf A}\in\mathbb{K}^{M\times N},
\qquad
\mathcal I=\{1,\ldots,M\},
\qquad
\mathcal J=\{1,\ldots,N\},
```

whose row and column indices are organized in the cluster trees
$\mathcal T_{\mathcal I}$ and $\mathcal T_{\mathcal J}$. For the Helmholtz
Green's function,

```math
g(\boldsymbol x,\boldsymbol y)
=
\frac{\exp\!\left(-\mathrm i\kappa
\lVert\boldsymbol x-\boldsymbol y\rVert_2\right)}
{4\pi\lVert\boldsymbol x-\boldsymbol y\rVert_2},
```

the rank of a conventionally admissible block grows with the electrical size
$\kappa d$ of its clusters. A single basis per cluster is therefore efficient
only in the low-frequency regime.

Directional $\mathcal H^2$-matrices recover bounded ranks at high frequency by
combining stronger separation with cone-shaped far fields [[2, 3, 5]](@ref refs).
Inside a cone about a unit direction $\boldsymbol e$, the kernel is written as

```math
g(\boldsymbol x,\boldsymbol y)
=
\exp\!\left(-\mathrm i\kappa
\langle\boldsymbol x-\boldsymbol y,\boldsymbol e\rangle\right)
\frac{
\exp\!\left[-\mathrm i\kappa
\left(
\lVert\boldsymbol x-\boldsymbol y\rVert_2
-\langle\boldsymbol x-\boldsymbol y,\boldsymbol e\rangle
\right)
\right]}
{4\pi\lVert\boldsymbol x-\boldsymbol y\rVert_2}.
```

The first factor is a plane wave. Under a suitable angular and parabolic
separation condition, the remaining factor is smooth and can be approximated
with a rank that is essentially independent of $\kappa$. Each electrically
large cluster consequently uses several bases, one for each relevant direction.

## Geometry of a cluster

Let $\mathcal X_t$ and $\mathcal Y_s$ be the geometric supports of a test
cluster $t$ and a trial cluster $s$. Their diameter and distance are

```math
\operatorname{diam}(\mathcal X)
=
\sup_{\boldsymbol x,\boldsymbol y\in\mathcal X}
\lVert\boldsymbol x-\boldsymbol y\rVert_2,
\qquad
\operatorname{dist}(\mathcal X,\mathcal Y)
=
\inf_{\substack{\boldsymbol x\in\mathcal X\\
                 \boldsymbol y\in\mathcal Y}}
\lVert\boldsymbol x-\boldsymbol y\rVert_2.
```

For a bounding-ball or k-means tree, cluster $t$ is represented by a ball
$\mathcal B_t$ with center $\boldsymbol c_t$ and radius $r_t$. The distance
used during block-tree construction is

```math
\operatorname{dist}(\mathcal B_t,\mathcal B_s)
=
\max\!\left\{
\lVert\boldsymbol c_t-\boldsymbol c_s\rVert_2-r_t-r_s,0
\right\}.
```

For a k-means tree, $\boldsymbol c_t$ is the cluster center and $r_t$ is chosen
large enough to contain the complete support. For a $2^N$-tree, the ball
encloses the corresponding axis-aligned box.

## Low- and high-frequency admissibility

An interaction is classified as low frequency if its smaller cluster is
electrically small,

```math
\kappa\min\!\left\{
\operatorname{diam}(\mathcal X_t),
\operatorname{diam}(\mathcal Y_s)
\right\}
\leq\gamma.
```

Such a pair is admissible when it satisfies the usual separation condition

```math
\max\!\left\{
\operatorname{diam}(\mathcal X_t),
\operatorname{diam}(\mathcal Y_s)
\right\}
\leq
\eta_{\ell}\operatorname{dist}(\mathcal X_t,\mathcal Y_s).
```

If both clusters are electrically large, the pair is a high-frequency
interaction. It must satisfy the parabolic condition

```math
\kappa\max\!\left\{
\operatorname{diam}(\mathcal X_t)^2,
\operatorname{diam}(\mathcal Y_s)^2
\right\}
\leq
\eta_{\mathrm h}\operatorname{dist}(\mathcal X_t,\mathcal Y_s).
```

The terminal block partition is therefore

```math
\mathcal P
=
\mathcal P_{\mathrm{adm}}
\mathbin{\dot\cup}
\mathcal P_{\mathrm{nonadm}},
\qquad
\mathcal P_{\mathrm{adm}}
=
\mathcal P_{\mathrm{low}}
\mathbin{\dot\cup}
\mathcal P_{\mathrm{high}}.
```

Blocks in $\mathcal P_{\mathrm{nonadm}}$ are stored densely. The parabolic
condition becomes stricter for large clusters, so most high-frequency
interactions are compressed farther down the trees.

The frequency label of an interaction and the basis regime of a cluster serve
different purposes. The former determines admissibility; the latter determines
whether that cluster owns one basis or a family of directional bases. Every
cluster receives a single, partner-independent regime so that all blocks can
share its bases. In a bounding-ball tree, siblings are classified together from
their average diameter. This prevents adjacent siblings from changing regime at
different depths. A mixed interaction can consequently use an ordinary basis on
its low-frequency side and a directional basis on its high-frequency side.

## Direction sets

Let $\mathcal E(t)\subset\mathbb S^{N-1}$ be the direction set of a
high-frequency cluster $t$. The cone represented by
$\boldsymbol e\in\mathcal E(t)$ should satisfy an angular condition of the form

```math
\sin\angle\!\left(
\boldsymbol c_t-\boldsymbol y,\boldsymbol e
\right)
\leq
\frac{\gamma}{
\kappa\operatorname{diam}(\mathcal B_t)},
\qquad \boldsymbol y\in\mathcal Y_s.
```

Each admissible partner is assigned to its closest direction,

```math
\boldsymbol e(t,s)
=
\underset{\boldsymbol e\in\mathcal E(t)}{\operatorname{argmin}}
\;\angle\!\left(
\boldsymbol c_t-\boldsymbol c_s,\boldsymbol e
\right).
```

The required number of directions grows with the surface area of the unit sphere
measured at angular resolution $\mathcal O((\kappa d)^{-1})$. The package uses
two constructions adapted to its two tree families.

### Bounding-ball and k-means trees

In two dimensions, the angular condition gives

```math
n_t
=
\max\!\left\{
4,
\left\lceil
\frac{\pi}{
\arcsin\!\left(
\min\!\left\{1,
\frac{\gamma}{\kappa\operatorname{diam}(\mathcal B_t)}
\right\}
\right)}
\right\rceil
\right\}.
```

In three dimensions, let
$\alpha_0=\arccos(1/\sqrt 3)$ be the covering angle of the six coordinate-axis
directions. The geometry-adaptive estimate used in [[5]](@ref refs) is

```math
n_t
=
\max\!\left\{
6,
\left\lceil
6\left[
\frac{\alpha_0}{
\arcsin\!\left(
\min\!\left\{1,
\frac{\gamma}{\kappa\operatorname{diam}(\mathcal B_t)}
\right\}
\right)}
\right]^2
\right\rceil
\right\}.
```

The directions are placed by spherical Fibonacci points. For
$i=0,\ldots,n_t-1$,

```math
z_i=1-2\frac{i+\tfrac12}{n_t},
\qquad
\rho_i=\sqrt{1-z_i^2},
\qquad
\phi=\pi(3-\sqrt5),
```

and

```math
\boldsymbol e_i
=
\begin{bmatrix}
\rho_i\cos(i\phi) &
\rho_i\sin(i\phi) &
z_i
\end{bmatrix}^{\mathsf T}.
```

Thus each cluster receives the number of directions required by its own
electrical diameter. This is particularly useful for a k-means tree whose
clusters on the same level can have different sizes.

### $2^N$-trees

For a $2^N$-tree, the direction hierarchy starts with the $2N$ coordinate-axis
directions at the finest high-frequency level. Moving to a coarser level, every
direction is subdivided into $2^{N-1}$ directions. Larger clusters therefore
receive the finer angular subdivision required by their electrical size.

## Low-frequency and directional far fields

The directly admissible field of any cluster is

```math
\mathcal F_{\mathrm w}(t)
=
\mathbin{\dot\bigcup}
\left\{
s\in\mathcal T_{\mathcal J}:
(t,s)\in\mathcal P_{\mathrm{adm}}
\right\}.
```

If $t$ is a low-frequency cluster, its complete far field is

```math
\mathcal F(t)
=
\begin{cases}
\mathcal F_{\mathrm w}(t)\cup\mathcal F(t_{\mathrm p}),
& t_{\mathrm p}\text{ is low frequency},\\[2mm]
\mathcal F_{\mathrm w}(t),
& t\text{ is a root or }t_{\mathrm p}\text{ is high frequency}.
\end{cases}
```

Thus an ordinary low-frequency basis inherits interactions only through the
low-frequency part of the tree.

The directly admissible part of the far field belonging to
$(t,\boldsymbol e)$ is

```math
\mathcal F_{\boldsymbol e,\mathrm w}(t)
=
\mathbin{\dot\bigcup}
\left\{
s\in\mathcal T_{\mathcal J}:
(t,s)\in\mathcal P_{\mathrm{adm}},\;
\boldsymbol e(t,s)=\boldsymbol e
\right\}.
```

Nestedness requires a child basis to reproduce the interactions already
compressed at its parent. Let $t'$ be a high-frequency child of a
high-frequency cluster $t$. A parent direction is mapped to the nearest child
direction,

```math
\pi_{t'}(\boldsymbol e)
=
\underset{\widetilde{\boldsymbol e}\in\mathcal E(t')}
{\operatorname{argmin}}
\;\angle(\boldsymbol e,\widetilde{\boldsymbol e}).
```

Since a child is electrically smaller, several parent directions may map to the
same child direction. The inherited far field is

```math
\mathcal F_{\boldsymbol e'}(t')
=
\mathcal F_{\boldsymbol e',\mathrm w}(t')
\cup
\mathbin{\dot\bigcup}_{\substack{
\boldsymbol e\in\mathcal E(t)\\
\pi_{t'}(\boldsymbol e)=\boldsymbol e'}}
\mathcal F_{\boldsymbol e}(t).
```

At a high-frequency root, the complete and directly admissible directional far
fields coincide. When a child changes from the high-frequency to the
low-frequency regime, directional inheritance stops and the child starts with
its own direct far field. The high-frequency parent stores an explicit basis;
the low-frequency subtree constructs its own nested bases and transfers. There
is no transfer matrix across the regime boundary.

This construction is more than a bookkeeping detail. It ensures that every
directional basis represents all interactions for which it will be used during
the matrix-vector product, including interactions introduced at ancestors.

## Directional $\mathcal H^2$ representation

For a high-frequency admissible pair, the block approximation is

```math
\boldsymbol{\mathsf A}_{ts}
\approx
\boldsymbol{\mathsf U}_t^{\boldsymbol e(t,s)}
\boldsymbol{\mathsf S}_{ts}
\boldsymbol{\mathsf V}_s^{\boldsymbol e(s,t)}.
```

The bases on both sides point towards the respective partner. A low-frequency
cluster has a single basis and its directional superscript is omitted. The same
formula therefore also covers low-frequency and mixed interactions.

For a parent direction $\boldsymbol e\in\mathcal E(t)$ and a child $t'$, let
$\boldsymbol e'=\pi_{t'}(\boldsymbol e)$. Within one frequency regime, the
directional bases satisfy

```math
\left.\boldsymbol{\mathsf U}_t^{\boldsymbol e}\right|_{t'}
\approx
\boldsymbol{\mathsf U}_{t'}^{\boldsymbol e'}
\boldsymbol{\mathsf E}_{t',t}^{\boldsymbol e}.
```

For $\mathcal S(t)=\{t_1,\ldots,t_q\}$ this gives

```math
\boldsymbol{\mathsf U}_t^{\boldsymbol e}
\approx
\begin{bmatrix}
\boldsymbol{\mathsf U}_{t_1}^{\pi_{t_1}(\boldsymbol e)}
\boldsymbol{\mathsf E}_{t_1,t}^{\boldsymbol e}\\
\vdots\\
\boldsymbol{\mathsf U}_{t_q}^{\pi_{t_q}(\boldsymbol e)}
\boldsymbol{\mathsf E}_{t_q,t}^{\boldsymbol e}
\end{bmatrix}.
```

Low-frequency bases obey the same equation without directional superscripts.

## Wideband nested cross approximation

For a low-frequency cluster, the construction is the ordinary NCA applied to
the regime-restricted far field above. Pivots $\tau_t\subset t$ and
$\sigma_{\mathcal F(t)}\subset\mathcal F(t)$ give

```math
\boldsymbol{\mathsf A}_{t,\mathcal F(t)}
\approx
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}
\boldsymbol{\mathsf A}_{\tau_t,\mathcal F(t)},
```

and the interpolatory basis is

```math
\boldsymbol{\mathsf U}_t
=
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}.
```

At a non-leaf low-frequency cluster, the candidates are the pivots of its
low-frequency children and the transfer is

```math
\boldsymbol{\mathsf E}_{t',t}
=
\boldsymbol{\mathsf A}_{\tau_{t'},\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}.
```

The same interpolation principle is now applied separately to each directional
far field of a high-frequency cluster.

For every cluster-direction pair with a nonempty far field, the bottom-up
compressor selects row pivots
$\tau_t^{\boldsymbol e}\subset t$ and far-field column pivots
$\sigma_{\mathcal F_{\boldsymbol e}(t)}
\subset\mathcal F_{\boldsymbol e}(t)$. The directional interaction matrix has
the cross approximation

```math
\boldsymbol{\mathsf A}_{t,\mathcal F_{\boldsymbol e}(t)}
\approx
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F_{\boldsymbol e}(t)}}
\boldsymbol{\mathsf A}_{
\tau_t^{\boldsymbol e},
\sigma_{\mathcal F_{\boldsymbol e}(t)}}^{-1}
\boldsymbol{\mathsf A}_{
\tau_t^{\boldsymbol e},
\mathcal F_{\boldsymbol e}(t)}.
```

At a leaf, or at a high-frequency cluster whose children are low frequency, the
explicit basis is

```math
\boldsymbol{\mathsf U}_t^{\boldsymbol e}
=
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F_{\boldsymbol e}(t)}}
\boldsymbol{\mathsf A}_{
\tau_t^{\boldsymbol e},
\sigma_{\mathcal F_{\boldsymbol e}(t)}}^{-1}.
```

Otherwise, only pivots and transfers are needed. For a child $t'$ and
$\boldsymbol e'=\pi_{t'}(\boldsymbol e)$, the pivot candidates of the parent are

```math
\widetilde t^{\boldsymbol e}
=
\mathbin{\dot\bigcup}_{t'\in\mathcal S(t)}
\tau_{t'}^{\pi_{t'}(\boldsymbol e)}.
```

The selected pivots satisfy
$\tau_t^{\boldsymbol e}\subset\widetilde t^{\boldsymbol e}$, and the transfer
matrix is

```math
\boldsymbol{\mathsf E}_{t',t}^{\boldsymbol e}
=
\boldsymbol{\mathsf A}_{
\tau_{t'}^{\boldsymbol e'},
\sigma_{\mathcal F_{\boldsymbol e}(t)}}
\boldsymbol{\mathsf A}_{
\tau_t^{\boldsymbol e},
\sigma_{\mathcal F_{\boldsymbol e}(t)}}^{-1}.
```

The trial tree is compressed analogously. If $(t,s)$ is admissible, its coupling
matrix is the matrix evaluated at the two selected pivot sets,

```math
\boldsymbol{\mathsf S}_{ts}
=
\boldsymbol{\mathsf A}_{
\tau_t^{\boldsymbol e(t,s)},
\sigma_s^{\boldsymbol e(s,t)}}.
```

For a low-frequency side, the superscript is omitted. These formulas repeat the
ordinary $\mathcal H^2$ construction deliberately: the wideband method changes
the far fields and the nesting relation, while retaining the same interpolatory
cross approximation.

IACA avoids assembling complete rows over the large inherited far field. It
selects far-field pivots geometrically and obtains the opposite pivots from the
evaluated matrix entries. Its convergence estimator uses the sampled cross
terms, so neither the full directional far field nor its residual has to be
assembled [[4, 5]](@ref refs).

## Orientation-aware pivoting for the EFIE

The electric field integral equation contains a vector-potential contribution
and a scalar-potential contribution. For RWG basis functions, the former depends
on edge direction and surface orientation, whereas the latter is essentially
insensitive to the edge direction. A purely geometric pivot sequence can
therefore keep sampling almost parallel edges or one nearly planar part of a
cluster and declare convergence before all vector components have been tested.

The EFIE filter described in [[5]](@ref refs) augments IACA with two geometric
labels for every basis function:

- a canonicalized unit edge direction $\boldsymbol\ell_\beta$;
- a representative normal $\widehat{\boldsymbol n}_\beta$, obtained from the
  adjacent element normals.

Edge directions are grouped without distinguishing their signs. New groups are
preferred until the represented orientations have been sampled; afterwards a
group that has already supplied $m_\ell$ pivots receives the weight
$1/(1+m_\ell)$. For the normal-dependent part, up to three dominant, mutually
orthogonal normal groups are selected so that they cover most of the cluster.
The groups and the remaining functions are processed cyclically, and every
active group must satisfy the phase-convergence criterion.

The `PhaseExtrapolator` uses one common approximation scale but maintains a
separate residual and trend history for each active normal group. Consequently,
convergence in one planar component cannot hide a missing component. When the
normal-group conditions are not met, the normal restriction and grouped
convergence test reduce to the ordinary IACA criterion; edge weighting remains
available when edge labels are present. This is the role of
`EFIEDirectionalFilter` in the BEAST construction shown in the examples.

## Matrix-vector multiplication

The directional matrix-vector product has the same five stages as the ordinary
$\mathcal H^2$ product, but expansion coefficients are indexed by both cluster
and direction:

1. Trial leaves are projected into every active low-frequency or directional
   basis.
2. During the upward pass, child coefficients are mapped through the transfer
   associated with the parent direction. Several parent directions may read
   from the same child direction.
3. Every admissible block applies
   $\boldsymbol{\mathsf S}_{ts}$ between the coefficients belonging to
   $\boldsymbol e(s,t)$ and $\boldsymbol e(t,s)$.
4. The downward pass applies the row transfers in reverse tree order.
5. Explicit row bases at leaves and frequency boundaries expand the coefficients
   into the result vector.

Products with the blocks in $\mathcal P_{\mathrm{nonadm}}$ are added directly.
Low-frequency clusters take part with their single direction-independent basis.

## Accuracy and complexity

As in the ordinary construction, every local IACA is controlled by the requested
relative tolerance, while transfers propagate local errors through the tree.
The complete matrix error is therefore normally a small multiple of the local
tolerance rather than being bounded by it exactly.

In the low-frequency regime, bounded ranks and bounded interaction lists give
linear storage, setup, and matrix-vector complexity. In three dimensions, the
direction count of an electrically large cluster satisfies

```math
|\mathcal E(t)|
=
\mathcal O\!\left(
[\kappa\operatorname{diam}(\mathcal B_t)]^2
\right).
```

For a balanced, wavelength-resolving tree, summing this count over all clusters
and levels gives $\mathcal O(N\log N)$. Under the usual sparsity and bounded-rank
assumptions, wideband construction, storage, and matrix-vector multiplication
therefore have $\mathcal O(N\log N)$ complexity. The geometry-adaptive direction
count, nearest-direction maps, and orientation-aware EFIE filter change the
constants but not this asymptotic estimate.
