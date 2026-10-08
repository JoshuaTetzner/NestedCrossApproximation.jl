# [$\mathcal{H}^2$-Matrices](@id nca_theory)

## Dense matrices and cluster trees

Let

```math
\boldsymbol{\mathsf A}\in\mathbb{K}^{M\times N},
\qquad
\mathcal I=\{1,\ldots,M\},
\qquad
\mathcal J=\{1,\ldots,N\},
```

be the dense matrix obtained, for example, by discretizing a boundary integral
operator. With test functions $\psi_i$, trial functions $\varphi_j$, and a kernel
$g$, its entries are

```math
A_{ij}
=
\int_\Gamma \psi_i(\boldsymbol x)
\int_\Gamma g(\boldsymbol x,\boldsymbol y)\,
             \varphi_j(\boldsymbol y)\,\mathrm dS(\boldsymbol y)\,
\mathrm dS(\boldsymbol x).
```

Storing $\boldsymbol{\mathsf A}$ and multiplying it by a vector both require
$\mathcal O(MN)$ work or memory. An $\mathcal H^2$ representation avoids this
quadratic cost by exploiting the smoothness of $g$ for geometrically separated
test and trial functions.

The test indices are organized in a cluster tree $\mathcal T_{\mathcal I}$ and
the trial indices in a cluster tree $\mathcal T_{\mathcal J}$. For a cluster
$t$, let $\mathcal S(t)$ denote its children. Every non-leaf cluster is the
disjoint union of its children,

```math
t=\mathbin{\dot\bigcup}_{t'\in\mathcal S(t)}t'.
```

The geometric supports belonging to test and trial clusters are denoted by
$\mathcal X_t$ and $\mathcal Y_s$. A usual admissibility condition is

```math
\max\!\left\{
    \operatorname{diam}(\mathcal X_t),
    \operatorname{diam}(\mathcal Y_s)
\right\}
\leq
\eta\,\operatorname{dist}(\mathcal X_t,\mathcal Y_s).
```

Simultaneous traversal of the two cluster trees produces a block partition

```math
\mathcal P
=
\mathcal P_{\mathrm{adm}}
\mathbin{\dot\cup}
\mathcal P_{\mathrm{nonadm}}.
```

An admissible pair $(t,s)\in\mathcal P_{\mathrm{adm}}$ is stored in compressed
form. A non-admissible terminal pair
$(t,s)\in\mathcal P_{\mathrm{nonadm}}$ is stored as the dense block
$\boldsymbol{\mathsf A}_{ts}$.

## The $\mathcal H^2$ representation

An $\mathcal H$-matrix gives every admissible block its own low-rank
factorization. An $\mathcal H^2$-matrix [[1]](@ref refs) instead shares one
cluster basis between all admissible blocks involving that cluster. For
$(t,s)\in\mathcal P_{\mathrm{adm}}$,

```math
\boldsymbol{\mathsf A}_{ts}
\approx
\boldsymbol{\mathsf U}_t\,
\boldsymbol{\mathsf S}_{ts}\,
\boldsymbol{\mathsf V}_s,
```

where

```math
\boldsymbol{\mathsf U}_t\in\mathbb{K}^{|t|\times k_t},
\qquad
\boldsymbol{\mathsf S}_{ts}\in\mathbb{K}^{k_t\times k_s},
\qquad
\boldsymbol{\mathsf V}_s\in\mathbb{K}^{k_s\times|s|}.
```

The row bases are nested. If $t'$ is a child of $t$, a transfer matrix
$\boldsymbol{\mathsf E}_{t',t}$ satisfies

```math
\left.\boldsymbol{\mathsf U}_t\right|_{t'}
\approx
\boldsymbol{\mathsf U}_{t'}\boldsymbol{\mathsf E}_{t',t}.
```

Consequently, for $\mathcal S(t)=\{t_1,\ldots,t_q\}$,

```math
\boldsymbol{\mathsf U}_t
\approx
\begin{bmatrix}
\boldsymbol{\mathsf U}_{t_1}\boldsymbol{\mathsf E}_{t_1,t}\\
\vdots\\
\boldsymbol{\mathsf U}_{t_q}\boldsymbol{\mathsf E}_{t_q,t}
\end{bmatrix}.
```

The trial bases satisfy the corresponding relation

```math
\left.\boldsymbol{\mathsf V}_s\right|_{s'}
\approx
\boldsymbol{\mathsf E}_{s',s}^{\mathsf T}
\boldsymbol{\mathsf V}_{s'}.
```

Only bases at the leaves, transfer matrices between adjacent levels, coupling
matrices, and dense near-field blocks have to be stored. If the ranks and the
number of interacting clusters are bounded, the storage is linear in $M+N$.

## Far fields and interpolatory bases

For a test cluster $t$, define its directly admissible or well-separated field
by

```math
\mathcal F_{\mathrm w}(t)
=
\mathbin{\dot\bigcup}
\left\{
s\in\mathcal T_{\mathcal J}:(t,s)\in\mathcal P_{\mathrm{adm}}
\right\}.
```

The basis at $t$ must also reproduce interactions compressed at its ancestors.
Its complete far field is therefore defined recursively by

```math
\mathcal F(t)
=
\begin{cases}
\mathcal F_{\mathrm w}(t), & t\text{ is the root},\\[2mm]
\mathcal F_{\mathrm w}(t)\cup\mathcal F(t_{\mathrm p}),
& t_{\mathrm p}\text{ is the parent of }t.
\end{cases}
```

Nested cross approximation constructs the cluster bases directly from entries
of $\boldsymbol{\mathsf A}$ [[1, 4]](@ref refs). It selects row pivots
$\tau_t\subset t$ and far-field column pivots
$\sigma_{\mathcal F(t)}\subset\mathcal F(t)$, with equal cardinality, such that

```math
\boldsymbol{\mathsf A}_{t,\mathcal F(t)}
\approx
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}
\boldsymbol{\mathsf A}_{\tau_t,\mathcal F(t)}.
```

The row cluster basis is the interpolation operator

```math
\boldsymbol{\mathsf U}_t
=
\boldsymbol{\mathsf A}_{t,\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}.
```

It is interpolatory because
$\left.\boldsymbol{\mathsf U}_t\right|_{\tau_t}
=\boldsymbol{\mathsf I}$. On the trial side, pivots
$\sigma_s\subset s$ and
$\tau_{\mathcal F(s)}\subset\mathcal F(s)$ give

```math
\boldsymbol{\mathsf V}_s
=
\boldsymbol{\mathsf A}_{\tau_{\mathcal F(s)},\sigma_s}^{-1}
\boldsymbol{\mathsf A}_{\tau_{\mathcal F(s)},s}.
```

The coupling matrix of an admissible pair is consequently a small submatrix of
the original matrix,

```math
\boldsymbol{\mathsf S}_{ts}
=
\boldsymbol{\mathsf A}_{\tau_t,\sigma_s}.
```

This identity is one of the central advantages of an interpolatory
$\mathcal H^2$ representation: no separate projection of every admissible block
is necessary.

## Nested pivot selection

For a leaf, all local indices are possible row-pivot candidates. For a
non-leaf cluster, the candidate set is reduced to the pivots of its children,

```math
\widetilde t
=
\mathbin{\dot\bigcup}_{t'\in\mathcal S(t)}\tau_{t'}.
```

Cross approximation of
$\boldsymbol{\mathsf A}_{\widetilde t,\mathcal F(t)}$ selects
$\tau_t\subset\widetilde t$. Thus every pivot of the parent is a pivot of one
of its children. Restricting the parent interpolation operator to the pivots of
a child yields the transfer matrix

```math
\boldsymbol{\mathsf E}_{t',t}
=
\boldsymbol{\mathsf A}_{\tau_{t'},\sigma_{\mathcal F(t)}}
\boldsymbol{\mathsf A}_{\tau_t,\sigma_{\mathcal F(t)}}^{-1}.
```

Together with the interpolation performed by the child, this gives

```math
\left.\boldsymbol{\mathsf U}_t\right|_{t'}
\approx
\boldsymbol{\mathsf U}_{t'}\boldsymbol{\mathsf E}_{t',t}.
```

The corresponding construction on the trial tree produces the trial transfer
matrices. In the implementation, inverses in these formulas are applied through
factorizations of the cross matrices. A numerically singular cross matrix leads
to a `SingularCrossMatrixError`.

## [Compressors](@id compressors_theory)

The compressors use the same interpolatory representation but differ in the
order in which clusters are processed and in the part of the far field exposed
to the cross approximation.

### Top-down construction

The top-down construction follows the original nested cross approximation
[[1]](@ref refs). It visits a parent before its children and passes a compressed
representation of the inherited far field down the tree. If $t$ has parent
$t_{\mathrm p}$, the local factorization sees

```math
\mathcal F^\downarrow(t)
=
\mathcal F_{\mathrm w}(t)
\cup
\sigma_{\mathcal F^\downarrow(t_{\mathrm p})}.
```

The pivots of the parent represent interactions introduced above the current
level. The row pivots at $t$ are selected from all indices in $t$, so nestedness
is obtained to the accuracy of the cross approximations rather than by a strict
subset relation between parent and child pivots.

### Bottom-up construction

The bottom-up construction visits the leaves first. A leaf is compressed against
its complete inherited far field $\mathcal F(t)$. At an inner cluster, only the
union of the child pivots is retained as the row candidate set. This produces
strictly nested pivot sets and the transfer matrices above.

The complete far field can contain a fixed fraction of all unknowns. A standard
ACA would evaluate complete rows of
$\boldsymbol{\mathsf A}_{\widetilde t,\mathcal F(t)}$. The incomplete ACA
(IACA) [[4]](@ref refs) instead chooses candidates in $\mathcal F(t)$ from their
geometry and evaluates only selected columns. Tree-mimicry pivoting descends the
far-field cluster tree, so selecting a pivot does not require scanning every
far-field index.

### Batched bottom-up construction

The batched variant separates one level into three phases:

1. For each cluster, geometric pivoting selects at most `maxrank` far-field
   columns without evaluating matrix entries.
2. The small candidate blocks of all clusters on the level are assembled
   together.
3. Fully pivoted ACA factorizes every candidate block and determines its row and
   column pivots.

Selection and factorization are not interleaved, so the selected candidate count
does not adapt to early convergence. The benefit is that matrix-entry evaluation
can be organized as a batch of small dense blocks.

### Representor construction

A representor replaces the complete far field by a smaller set
$\mathcal R(t)\subset\mathcal F(t)$ before ACA is applied. The package can pick
this set randomly, near a Chebyshev grid fitted to the principal axes of the far
field, or with tree-mimicry pivoting. The ACA tolerance then controls the error
on this representative block.

### Geometric pivoting

The mimicry pivoting supplied by AdaptiveCrossApproximation.jl estimates which
far-field pivot a fully pivoted ACA would choose. A candidate
$\boldsymbol y\in\mathcal F(t)$ is weighted by its separation from pivots already
chosen, its fill distance, and its distance from the cluster. The first two
terms distribute pivots through the directional content of the far field; the
last favors nearby interactions, where the kernel is usually strongest. The
IACA documentation in AdaptiveCrossApproximation.jl gives the full selection and
convergence algorithm.

## Matrix-vector multiplication

For $\boldsymbol y=\boldsymbol{\mathsf A}\boldsymbol x$, the compressed part is
applied in five stages:

1. **Leaf projection.** At every trial leaf,
   $\widehat{\boldsymbol x}_s=\boldsymbol{\mathsf V}_s\boldsymbol x_s$.
2. **Upward aggregation.** From leaves to root,
   $\widehat{\boldsymbol x}_s
   =\sum_{s'\in\mathcal S(s)}
   \boldsymbol{\mathsf E}_{s',s}^{\mathsf T}
   \widehat{\boldsymbol x}_{s'}$.
3. **Coupling.** For every admissible pair,
   $\widehat{\boldsymbol y}_t
   \mathrel{+}=\boldsymbol{\mathsf S}_{ts}
   \widehat{\boldsymbol x}_s$.
4. **Downward disaggregation.** From root to leaves,
   $\widehat{\boldsymbol y}_{t'}
   \mathrel{+}=\boldsymbol{\mathsf E}_{t',t}
   \widehat{\boldsymbol y}_t$.
5. **Leaf expansion.** At every test leaf,
   $\boldsymbol y_t
   \mathrel{+}=\boldsymbol{\mathsf U}_t
   \widehat{\boldsymbol y}_t$.

The products with the dense blocks in $\mathcal P_{\mathrm{nonadm}}$ are then
added. Transposed and adjoint multiplication exchange the test and trial sides
and reverse the corresponding operators.

## Accuracy and complexity

Every local cross approximation is stopped at the prescribed relative
tolerance. Transfers propagate the approximation errors of descendants, so the
global matrix error is generally a small multiple of the local tolerance; the
tolerance itself is not a strict bound for the complete matrix error.

For bounded rank and bounded block-tree sparsity, the numbers of leaf-basis,
transfer, coupling, and near-field entries all grow linearly with the problem
size. Construction, storage, and matrix-vector multiplication therefore have
$\mathcal O(M+N)$ complexity. The rank, admissibility parameters, leaf size, and
depth of the trees determine the constants.
