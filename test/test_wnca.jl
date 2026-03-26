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
Random.seed!(1)
##
h = 0.05
λ = 10h
k = 2 * pi / λ

Γ1 = meshsphere(1.0, h)#meshrectangle(1.0, 2.0, 0.08)
Γ2 = meshsphere(1.0, 0.045)#translate(Γ1, SVector(0.0, 0.0, 3.85))
tRT = raviartthomas(Γ1)
sRT = raviartthomas(Γ2)
println("Size RT ", length(tRT))

op = Maxwell3D.singlelayer(; wavenumber=k)
#testtree = KMeansTree(tRT.pos, 2; minvalues=100)
#trialtree = KMeansTree(sRT.pos, 2; minvalues=100)
testtree = TwoNTree(tRT.pos, 2 / 2^10; minvalues=200)
trialtree = TwoNTree(sRT.pos, 2 / 2^10; minvalues=200)

tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5, γ=1.0)

for level in H2Trees.levels(tree.testcluster)
    println("Level ", level, ", islf = ", isnear.islf(tree.testcluster, level))
end

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
        factorization=iACA(
            MaximumValue(),
            MimicryPivoting(tRT.pos, sRT.pos),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MimicryPivoting(sRT.pos, tRT.pos),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator(tol)),
        ),
    ),
    scheduler=DynamicScheduler(),
);

norm(h2mat * x - A * x) / norm(A * x)
xt = rand(ComplexF64, length(tRT))
norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt)
norm(adjoint(h2mat) * xt - adjoint(A) * xt) / norm(adjoint(A) * xt)
size(A)
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

for (idx, cp) in enumerate(h2mat.couplingmatrices.blocks)
    tidx = cps.plan.tidcs[idx]
    sidx = cps.plan.sidcs[idx]
    println(tidx, "; ", sidx)
    if isassigned(tb, tidx) && isassigned(sb, sidx)
        t = tbidcs[tidx]
        s = sbidcs[sidx]

        blk = tb[tidx] * cp * sb[sidx]

        println("t: $tidx, s: $sidx, relerr: $(norm(blk - A[t, s])/norm(A[t, s]))")
    end
end
