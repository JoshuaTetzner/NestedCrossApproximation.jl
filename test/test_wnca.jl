using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using Random
using LinearAlgebra
using OhMyThreads
using Test
using StaticArrays

##
Γ = meshcuboid(1.0, 1.0, 1.0, 0.0275)#meshicosphere(28, 1.0)#meshsphere(1.0, 0.05)#
space = raviartthomas(Γ)
println("Size RT ", length(space))
h = edgeinfo(Γ)[3]
λ = 10h
k = 2 * pi / λ
tRT = space#raviartthomas(Γ1)
sRT = space#raviartthomas(Γ2)

op = Maxwell3D.singlelayer(; wavenumber=k)
Random.seed!(1)
testtree = KMeansTree(
    tRT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
Random.seed!(1)
trialtree = KMeansTree(
    sRT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
#testtree = TwoNTree(tRT.pos, 2 / 2^10; minvalues=200)
#trialtree = TwoNTree(sRT.pos, 2 / 2^10; minvalues=200)

tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);
##

x = rand(ComplexF64, length(sRT))

A = assemble(op, tRT, sRT);

##
tol = 1e-3
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        #factorization=ACA(; tol=tol),#
        factorization=iACA(
            MaximumValue(),
            #MimicryPivoting(tRT.pos, sRT.pos),
            TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),##
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
            #MimicryPivoting(sRT.pos, tRT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
        #factorization=ACA(; tol=tol),
    ),
    scheduler=DynamicScheduler(),
);
##
hmat = AdaptiveCrossApproximation.HMatrix(
    op,
    tRT,
    sRT,
    tree;
    isnear=isnear,
    maxrank=100,
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    tol=1e-2 * tol,
    scheduler=DynamicScheduler(),
)
##
norm(h2mat * x - A * x) / norm(A * x)
xt = rand(ComplexF64, length(tRT))
norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt)
norm(adjoint(h2mat) * xt - adjoint(A) * xt) / norm(adjoint(A) * xt)
##
farh2mat = NestedCrossApproximation.farmatrix(h2mat)
farhmat = AdaptiveCrossApproximation.farmatrix(hmat)

norm(farhmat * x - farh2mat * x) / norm(farhmat * x)

##

estimate_reldifference(h2mat, A; tol=1e-4)
estimate_reldifference(hmat, A; tol=1e-4)
estimate_reldifference(h2mat, hmat; tol=1e-4)
estimate_reldifference(farh2mat, farhmat; tol=1e-4)

##

function testbases(h2mat, tree)
    bases = Vector{Matrix{ComplexF64}}(undef, length(h2mat.disaggregationplan.ptr) - 1)
    basesidcs = Vector{Vector{Int}}(undef, length(h2mat.disaggregationplan.ptr) - 1)
    b = h2mat.nestedtestbases

    for i in eachindex(b.plan.nodes)
        node = b.plan.nodes[i]
        idcs = H2Trees.values(tree, node)
        for ldiridx in b.plan.diridxptr[i]:(b.plan.diridxptr[i + 1] - 1)
            diridx = b.plan.diridx[ldiridx]
            bases[diridx] = b.blocks[ldiridx]
            basesidcs[diridx] = idcs
        end
    end
    return bases, basesidcs
end

function trialbases(h2mat, tree)
    bases = Vector{Matrix{ComplexF64}}(undef, length(h2mat.aggregationplan.ptr) - 1)
    basesidcs = Vector{Vector{Int}}(undef, length(h2mat.aggregationplan.ptr) - 1)
    b = h2mat.nestedtrialbases

    for i in eachindex(b.plan.nodes)
        node = b.plan.nodes[i]
        idcs = H2Trees.values(tree, node)
        for ldiridx in b.plan.diridxptr[i]:(b.plan.diridxptr[i + 1] - 1)
            diridx = b.plan.diridx[ldiridx]
            bases[diridx] = b.blocks[ldiridx]
            basesidcs[diridx] = idcs
        end
    end
    return bases, basesidcs
end

cps = h2mat.couplingmatrices
tb, tbidcs = testbases(h2mat, H2Trees.testtree(tree))
sb, sbidcs = trialbases(h2mat, H2Trees.trialtree(tree))
for (idx, cp) in enumerate(h2mat.couplingmatrices.blocks)
    tidx = cps.plan.tidcs[idx]
    sidx = cps.plan.sidcs[idx]
    if isassigned(tb, tidx) && isassigned(sb, sidx)
        t = tbidcs[tidx]
        s = sbidcs[sidx]

        blk = tb[tidx] * cp * sb[sidx]

        if norm(blk - A[t, s]) / norm(A[t, s]) > 1e-3
            println("t: $tidx, s: $sidx, relerr: $(norm(blk - A[t, s])/norm(A[t, s]))")
        end
    end
end
##
t = 624
s = [453, 454, 478, 481, 580, 137, 470]
tdata, sdata = NestedCrossApproximation.fardata(tree, isnear)
tidcs = H2Trees.values(H2Trees.testtree(tree), t);
sidcs = H2Trees.values(H2Trees.trialtree(tree), s);

##
using Plots
plotlyjs()

spos = space.pos[sidcs]
tpos = space.pos[tidcs]

p = scatter(getindex.(spos, 1), getindex.(spos, 2), getindex.(spos, 3));
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));

display(p)
##
factorization = iACA(
    MaximumValue(),
    TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
    FNormExtrapolator(iFNormEstimator(tol)),
)

iacafctr = factorization(tidcs, s, 40)

rowbuffer = zeros(ComplexF64, 40, 40)
colbuffer = zeros(ComplexF64, length(tidcs), 40)
row = zeros(Int, 40)
cols = zeros(Int, 40)
blk = A[tidcs, sidcs]
iacafctr.convergence.estimator.tol = 0.00025#1e-3
n, r, c = iacafctr(A, colbuffer, rowbuffer, row, cols, tidcs, s, 40;)
norm(blk - A[tidcs, c] * inv(A[r, c]) * A[r, sidcs]) / norm(blk)
##
norm(blk - A[tidcs, sidcs])
for i in eachindex(r)
    residual = blk - A[tidcs, c[1:i]] * inv(A[r[1:i], c[1:i]]) * A[r[1:i], sidcs]
    println(norm(residual) / norm(blk))
end
##
U, V = AdaptiveCrossApproximation.aca(blk; tol=1e-3);
norm(blk - U * V) / norm(blk)
size(U)
for i in 1:size(U, 2)
    residual = blk - U[:, 1:i] * V[1:i, :]
    println(norm(residual) / norm(blk))
end
##
#r = tidcs[[1, 87, 102, 2, 12, 86, 85, 97, 57, 30, 74]]
#c = sidcs[[27, 63, 9, 11, 49, 6, 13, 60, 48, 18, 17]]
#r = tidcs[nr]
#c = sidcs[nc]
spos = space.pos[sidcs]
spivs = space.pos[c]
tpos = space.pos[tidcs]
tpivs = space.pos[r]
p = scatter(getindex.(spos, 1), getindex.(spos, 2), getindex.(spos, 3));
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));
scatter!([getindex.(spivs, 1)], [getindex.(spivs, 2)], [getindex.(spivs, 3)])
scatter!([getindex.(tpivs, 1)], [getindex.(tpivs, 2)], [getindex.(tpivs, 3)])
scatter!(
    [getindex(iacafctr.columnpivoting.refcentroid, 1)],
    [getindex(iacafctr.columnpivoting.refcentroid, 2)],
    [getindex(iacafctr.columnpivoting.refcentroid, 3)],
)
display(p)
