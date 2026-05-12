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

function getedges(m::Mesh)
    edgemesh = skeleton(m, 1)
    edges = Vector{SVector{3,Float64}}(undef, length(edgemesh.faces))

    for (i, face) in enumerate(edgemesh.faces)
        edgevector = edgemesh.vertices[face[2]] - edgemesh.vertices[face[1]]
        edges[i] = edgevector / norm(edgevector)
    end
    return edges
end

function plane_normal(points::AbstractVector{<:SVector{3,T}}) where {T}
    c = sum(points) / length(points)
    C = @MMatrix zeros(T, 3, 3)
    for p in points
        q = p - c
        @inbounds begin
            C[1, 1] += q[1] * q[1]
            C[1, 2] += q[1] * q[2]
            C[1, 3] += q[1] * q[3]
            C[2, 2] += q[2] * q[2]
            C[2, 3] += q[2] * q[3]
            C[3, 3] += q[3] * q[3]
        end
    end
    C[2, 1] = C[1, 2]
    C[3, 1] = C[1, 3]
    C[3, 2] = C[2, 3]
    E = eigen(Symmetric(SMatrix(C)))
    n = SVector{3,T}(E.vectors[:, 1])
    return n / norm(n)
end

@inline function normal_axis(n::SVector{3,T}) where {T}
    ax = abs(n[1])
    ay = abs(n[2])
    az = abs(n[3])

    if ax >= ay && ax >= az
        return SVector{3,T}(one(T), zero(T), zero(T))
    elseif ay >= az
        return SVector{3,T}(zero(T), one(T), zero(T))
    else
        return SVector{3,T}(zero(T), zero(T), one(T))
    end
end

function nodeorientation(pos::Vector{SVector{3,Float64}}, tree)
    normals = Vector{SVector{3,Float64}}(undef, length(tree.nodes))
    for node in eachindex(tree.nodes)
        idcs = H2Trees.values(tree, node)
        normal = plane_normal(pos[idcs])
        normals[node] = normal
    end
    return normals
end

function edgeinfo(m)
    edges = skeleton(m, 1)

    _edgelength = Float64[]
    for (_, e) in enumerate(edges)
        push!(_edgelength, norm(diff(vertices(chart(edges, e)))))
    end
    _edgelength
    return minimum(_edgelength),
    sum(_edgelength) / length(_edgelength),
    maximum(_edgelength)
end

##
Γ = meshcuboid(1.0, 1.0, 1.0, 0.02)
#Γ = meshicosphere(40, 1.0)#meshsphere(1.0, 0.05)#
edges = getedges(Γ)
space = raviartthomas(Γ)
println("Size RT ", length(space))
h = edgeinfo(Γ)[3]
λ = 10h
k = 2 * pi / λ
gamma = im * k
alpha = -gamma
beta = -1 / gamma
tRT = space#raviartthomas(Γ1)
sRT = space#raviartthomas(Γ2)

op = Maxwell3D.singlelayer(; wavenumber=k)#
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
@time testorientations = nodeorientation(tRT.pos, testtree);
@time trialorientations = nodeorientation(sRT.pos, trialtree);
##

x = rand(ComplexF64, length(sRT))

A = assemble(op, tRT, sRT; threading=:cellcoloring);

##

tol = 1e-3
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    tRT,
    sRT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        #representor=NestedCrossApproximation.MimicryRep(
        #    AdaptiveCrossApproximation.TreeMimicryPivoting2(tRT.pos, sRT.pos, edges, H2Trees.trialtree(tree))
        #),
        #factorization=ACA(;
        #    convergence=CombindeConvCrit(
        #        FNormEstimator(; tol=tol),
        #        AdaptiveCrossApproximation.RandomSampling(; tol=tol),
        #    ),
        #),#
        factorization=iACA(
            MaximumValue(),
            #TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                tRT.pos, sRT.pos, edges, trialorientations, H2Trees.trialtree(tree)
            ),
            FNormExtrapolator(iFNormEstimator2(tol)),
        ),##
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        #representor=NestedCrossApproximation.MimicryRep(
        #    TreeMimicryPivoting2(sRT.pos, tRT.pos, edges, H2Trees.testtree(tree))
        #),
        #factorization=ACA(;
        #    convergence=CombinedConvCrit(
        #        FNormEstimator(; tol=tol),
        #        AdaptiveCrossApproximation.RandomSampling(; tol=tol),
        #    ),
        #),
        factorization=iACA(
            #TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                sRT.pos, tRT.pos, edges, testorientations, H2Trees.testtree(tree)
            ),
            MaximumValue(),
            FNormExtrapolator(iFNormEstimator2(tol)),
        ),

        #factorization=ACA(; tol=tol),
    ),
    scheduler=SerialScheduler(),
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

        if norm(blk - A[t, s]) / norm(A[t, s]) > 1e-1
            println("t: $tidx, s: $sidx, relerr: $(norm(blk - A[t, s])/norm(A[t, s]))")
        end
    end
end
##

tdata, sdata = NestedCrossApproximation.fardata(tree, isnear)
dir = NestedCrossApproximation.dirs(tdata, 45)[10]
dirfarfield = NestedCrossApproximation.dirfarfield(testtree, tdata, 45, dir)
#H2Trees.values(H2Trees.testtree(tree), t);
t = 285
tidcs = H2Trees.values(H2Trees.testtree(tree), t);
Ft = [
    244,
    245,
    247,
    248,
    259,
    260,
    255,
    262,
    263,
    41,
    42,
    45,
    46,
    48,
    49,
    52,
    53,
    56,
    57,
    59,
    60,
    124,
    125,
    126,
    130,
    131,
    133,
    134,
    136,
    137,
]
sidcs = H2Trees.values(H2Trees.trialtree(tree), Ft);

##
using Plots
plotlyjs()

spos = space.pos[sidcs]
tpos = space.pos[tidcs]

p = scatter(
    getindex.(spos, 1),
    getindex.(spos, 2),
    getindex.(spos, 3);
    aspect_ratio=:equal,
    xlims=(0, 1),
    ylims=(0, 1),
    zlims=(0, 1),
);
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));

display(p)
##
factorization = iACA(
    MaximumValue(),
    #AdaptiveCrossApproximation.TreeMimicryPivoting(
    #    tRT.pos, sRT.pos, H2Trees.trialtree(tree)
    #),
    AdaptiveCrossApproximation.TreeMimicryPivoting2(
        tRT.pos, sRT.pos, edges, trialorientations, H2Trees.trialtree(tree)
    ),
    FNormExtrapolator(iFNormEstimator2(tol)),
)

iacafctr = factorization(tidcs, Ft, 40)

rowbuffer = zeros(ComplexF64, 40, 40)
colbuffer = zeros(ComplexF64, length(tidcs), 40)
row = zeros(Int, 40)
cols = zeros(Int, 40)
blk = A[tidcs, sidcs]
iacafctr.convergence.estimator.tol = 0.001#1e-3
n, r, c = iacafctr(A, colbuffer, rowbuffer, row, cols, tidcs, Ft, 40;)
norm(blk - A[tidcs, c] * inv(A[r, c]) * A[r, sidcs]) / norm(blk)
##
using Polynomials
f2 = fit(collect(1:n), log10.(sort(iacafctr.convergence.lastnorms[1:n]; rev=true)), 2)

scatter(
    1:n, log10.(sort(iacafctr.convergence.lastnorms[1:n]; rev=true)); label="log10(norms)"
)
plot!(1:n, f2.(collect(1:n)); label="fit", linestyle=:dash)
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
#r = tidcs[nr]
#c = sidcs[nc]
##
spos = space.pos[sidcs]
spivs = space.pos[c]
tpos = space.pos[tidcs]
tpivs = space.pos[r]
p = scatter(getindex.(spos, 1), getindex.(spos, 2), getindex.(spos, 3); aspect_ratio=:equal);
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));
scatter!([getindex.(spivs, 1)], [getindex.(spivs, 2)], [getindex.(spivs, 3)])
scatter!([getindex.(tpivs, 1)], [getindex.(tpivs, 2)], [getindex.(tpivs, 3)])
#scatter!(
#    [getindex(iacafctr.columnpivoting.refcentroid, 1)],
#    [getindex(iacafctr.columnpivoting.refcentroid, 2)],
#    [getindex(iacafctr.columnpivoting.refcentroid, 3)],
#)
display(p)
##
k = 2 * π / 10
gamma = im * k
alpha = -gamma
beta = -1 / gamma

m1 = CompScienceMeshes.rotate(meshrectangle(1.0, 1.0, 1.0), SVector(0, 0, pi / 2))
m2 = CompScienceMeshes.rotate(meshrectangle(1.0, 1.0, 1.0), SVector(0, 0, pi / 2))
m2 = CompScienceMeshes.translate(m2, SVector(0, -2, 0))
using BEAST
s = raviartthomas(Γ)
t = raviartthomas(m2)

op = Maxwell3D.singlelayer(; wavenumber=k)
ϕ = Maxwell3D.singlelayer(; gamma=gamma, alpha=im * 0.0, beta=beta)
A = Maxwell3D.singlelayer(; gamma=gamma, alpha=alpha, beta=0.0 * im)

T = assemble(op, t, s)
T_phi = assemble(ϕ, t, s)
T_A = assemble(A, t, s)
##

##
using CompScienceMeshes
using StaticArrays
using LinearAlgebra

Γ = meshrectangle(1.0, 1.0, 0.1)#meshsphere(1.0, 0.1)

function getedges(m::Mesh)
    edgemesh = skeleton(m, 1)
    edges = Vector{SVector{3,Float64}}(undef, length(edgemesh.faces))

    for (i, face) in enumerate(edgemesh.faces)
        edgevector = edgemesh.vertices[face[2]] - edgemesh.vertices[face[1]]
        edges[i] = edgevector / norm(edgevector)
    end
    return edges
end

canon(d) =
    (d[1] < 0 || (d[1] == 0 && d[2] < 0) || (d[1] == 0 && d[2] == 0 && d[3] < 0)) ? -d : d

function orientation_weights!(weights, keys, selected)
    counts = Dict{typeof(keys[1]),Int}()
    for i in selected
        k = keys[i]
        counts[k] = get(counts, k, 0) + 1
    end
    for i in eachindex(keys)
        weights[i] = 1.0 / (1 + get(counts, keys[i], 0))
    end
end
##

dirs = getedges(Γ)
keys = [Tuple(round.(canon(d); digits=2)) for d in dirs]

weights = zeros(length(dirs))

orientation_weights!(weights, keys, Int[1, 2, 3, 5, 4])

weights
