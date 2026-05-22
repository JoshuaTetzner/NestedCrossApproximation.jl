using CompScienceMeshes
using AdaptiveCrossApproximation
using NestedCrossApproximation
using LinearAlgebra
using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using OhMyThreads
using Random

@inline unit(v) = v / norm(v)
@inline cz(x) = iszero(x) ? zero(x) : x
@inline clean(v) = typeof(v)(cz(v[1]), cz(v[2]), cz(v[3]))
@inline roundvec(v; digits=2) = clean(
    typeof(v)(
        round(v[1]; digits=digits),
        round(v[2]; digits=digits),
        round(v[3]; digits=digits),
    ),
)

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

@inline function canon(v)
    v = clean(v)
    return if v[1] < 0 || (v[1] == 0 && v[2] < 0) || (v[1] == 0 && v[2] == 0 && v[3] < 0)
        clean(-v)
    else
        v
    end
end

@inline function edgeverts(cell, refid)
    refid == 1 && return cell[2], cell[3]
    refid == 2 && return cell[3], cell[1]
    return cell[1], cell[2]
end

function rwg_orientations(rwg)
    mesh  = rwg.geo
    verts = vertices(mesh)
    cells = collect(CompScienceMeshes.cells(mesh))
    nc    = length(cells)

    ℓ  = similar(rwg.pos)
    n  = similar(rwg.pos)
    cn = similar(rwg.pos, nc)

    @inbounds for c in 1:nc
        cn[c] = clean(unit(normal(chart(mesh, cells[c]))))
    end

    println(unique(cn))

    @inbounds for i in eachindex(rwg.fns)
        fn = rwg.fns[i]
        sh = fn[1]

        cell = cells[sh.cellid]
        a, b = edgeverts(cell, sh.refid)

        ℓ[i] = roundvec(canon(unit(verts[b] - verts[a])))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            ni += dot(ni, nj) < 0 ? -nj : nj
        end
        n[i] = roundvec(unit(ni))
    end

    return ℓ, n
end

@inline function dirkey(v; digits=1)
    s = 10.0^digits
    x = round(Int16, v[1] * s)
    y = round(Int16, v[2] * s)
    z = round(Int16, v[3] * s)

    return if x < 0 || (x == 0 && y < 0) || (x == 0 && y == 0 && z < 0)
        (-x, -y, -z)
    else
        (x, y, z)
    end
end

@inline orientation_key(ℓ, n; dell=1, dn=1) = (dirkey(ℓ; digits=dell), dirkey(n; digits=dn))

function orientation_keys(ℓ, n; dell=1, dn=1)
    K = typeof(orientation_key(ℓ[1], n[1]; dell=dell, dn=dn))
    q = Vector{K}(undef, length(ℓ))

    @inbounds for i in eachindex(ℓ)
        q[i] = orientation_key(ℓ[i], n[i]; dell=dell, dn=dn)
    end

    return q
end
##
Γ = meshcuboid(1.0, 1.0, 1.0, 0.0135)
RT = raviartthomas(Γ)
println("Size RT ", length(RT))
h = edgeinfo(Γ)[3]
λ = 10h
k = 2 * pi / λ
gamma = im * k
alpha = -gamma
beta = -1 / gamma

op = Maxwell3D.singlelayer(; wavenumber=k)#
Random.seed!(1)
testtree = KMeansTree(
    RT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
Random.seed!(1)
trialtree = KMeansTree(
    RT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
tree = H2Trees.BlockTree(testtree, trialtree)
isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);
##
edges, normals = rwg_orientations(RT)
unique(normals)
edgeids, normalids, nedgeids, nnormalids = NestedCrossApproximation.basisfunction_orientation_ids(
    edges, normals
)
trial_node_normal_sets, trialnormalids, ntrialnormalids = NestedCrossApproximation.node_normal_orientation_sets(
    normals, trialtree
)
test_node_normal_sets, testnormalids, ntestnormalids = NestedCrossApproximation.node_normal_orientation_sets(
    normals, testtree
)
edges
##

x = rand(ComplexF64, length(RT))

A = assemble(op, RT, RT; threading=:cellcoloring);

##

tol = 1e-3
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    RT,
    RT,
    tree;
    isnear=isnear,
    testcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            MaximumValue(),
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                RT.pos, RT.pos, edgeids, trialnormalids, trial_node_normal_sets, trialtree
            ),
            OversampIFNormEst(tol),
        ),
    ),
    trialcompressor=NestedCrossApproximation.BottomUp(;
        factorization=iACA(
            AdaptiveCrossApproximation.TreeMimicryPivoting2(
                RT.pos, RT.pos, edgeids, testnormalids, test_node_normal_sets, testtree
            ),
            MaximumValue(),
            OversampIFNormEst(tol),
        ),
    ),
    #scheduler=SerialScheduler(),
);
##
hmat = AdaptiveCrossApproximation.HMatrix(
    op,
    RT,
    RT,
    tree;
    isnear=isnear,
    maxrank=100,
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    tol=1e-4 * tol,
    scheduler=DynamicScheduler(),
)
##
norm(h2mat * x - A * x) / norm(A * x)
xt = rand(ComplexF64, length(RT))
norm(transpose(h2mat) * xt - transpose(A) * xt) / norm(transpose(A) * xt)
norm(adjoint(h2mat) * xt - adjoint(A) * xt) / norm(adjoint(A) * xt)
##
farh2mat = NestedCrossApproximation.farmatrix(h2mat)
farhmat = AdaptiveCrossApproximation.farmatrix(hmat)

norm(farhmat * x - farh2mat * x) / norm(farhmat * x)
estimate_reldifference(farh2mat, farhmat; tol=1e-4)
##

#t = 566
#tidcs = H2Trees.values(H2Trees.testtree(tree), t)
tidcs = [
    75390,
    23502,
    75352,
    75206,
    75303,
    75292,
    23479,
    75392,
    75258,
    75300,
    75254,
    75221,
    75405,
    75242,
    75208,
    75209,
    75233,
    75399,
    23484,
    75290,
    75307,
    82015,
    75310,
    16674,
    23408,
    81964,
    75275,
    23424,
    75328,
    75281,
    75295,
    82021,
    75320,
    81962,
    75280,
    75289,
    75313,
    82024,
]
Ft = [656, 1511]

sidcs = H2Trees.values(H2Trees.trialtree(tree), Ft);

##
using Plots
plotlyjs()

spos = RT.pos[sidcs]
tpos = RT.pos[tidcs]

p = scatter(
    getindex.(spos, 1),
    getindex.(spos, 2),
    getindex.(spos, 3);
    aspect_ratio=:equal,
    xlims=(-0.24, 1.25),
    ylims=(-0.25, 1.25),
    zlims=(-0.250, 1.25),
);
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));

display(p)
##
tol = 1e-3
cpiv = AdaptiveCrossApproximation.TreeMimicryPivoting2(
    RT.pos, RT.pos, edgeids, trialnormalids, trial_node_normal_sets, trialtree
)
factorization = iACA(MaximumValue(), cpiv, OversampIFNormEst(tol))

bdirs = cpiv.basisdirections

iacafctr = factorization(tidcs, Ft, 40)

rowbuffer = zeros(ComplexF64, 40, 40)
colbuffer = zeros(ComplexF64, length(tidcs), 40)
row = zeros(Int, 40)
cols = zeros(Int, 40)
blk = A[tidcs, sidcs]
n, r, c = iacafctr(A, colbuffer, rowbuffer, row, cols, tidcs, Ft, 40;)
norm(blk - A[tidcs, c] * inv(A[r, c]) * A[r, sidcs]) / norm(blk)
for i in 1:n
    println(norm(blk - A[tidcs, c[1:i]] * inv(A[r[1:i], c[1:i]]) * A[r[1:i], sidcs]))
end

##
U, V = AdaptiveCrossApproximation.aca(
    blk;
    convergence=AdaptiveCrossApproximation.CombinedConvCrit([
        FNormEstimator(tol * 1e-2),
        AdaptiveCrossApproximation.RandomSampling(; tol=tol * 1e-2, factor=1.0),
    ]),
);
norm(blk - U * V) / norm(blk)
##
size(U)
r = tidcs[[1, 31, 4, 65, 84, 6, 3, 72, 74, 80, 11, 21, 10, 89, 90, 40, 70, 58, 8, 35, 9, 7]]
c = sidcs[[
    447,
    504,
    525,
    404,
    704,
    221,
    6835,
    401,
    693,
    3983,
    473,
    382,
    519,
    1727,
    6483,
    2740,
    111,
    449,
    979,
    662,
    474,
    593,
]]
##

spos = RT.pos[sidcs]
spivs = RT.pos[c]
tpos = RT.pos[tidcs]
tpivs = RT.pos[r]
p = scatter(
    getindex.(spos, 1),
    getindex.(spos, 2),
    getindex.(spos, 3);
    aspect_ratio=:equal,
    xlims=(-0.24, 1.25),
    ylims=(-0.25, 1.25),
    zlims=(-0.250, 1.25),
);
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));
scatter!([getindex.(spivs, 1)], [getindex.(spivs, 2)], [getindex.(spivs, 3)])
scatter!([getindex(spivs[end], 1)], [getindex(spivs[end], 2)], [getindex(spivs[end], 3)])
scatter!([getindex.(tpivs, 1)], [getindex.(tpivs, 2)], [getindex.(tpivs, 3)])
#scatter!(
#    [getindex(iacafctr.columnpivoting.refcentroid, 1)],
#    [getindex(iacafctr.columnpivoting.refcentroid, 2)],
#    [getindex(iacafctr.columnpivoting.refcentroid, 3)],
#)
display(p)
##
sdirs = bdirs[sidcs]
spos1 = spos[findall(x -> x == 4, sdirs)]
spos2 = spos[findall(x -> x == 5, sdirs)]
spos3 = spos[findall(x -> x == 6, sdirs)]

p = scatter(
    getindex.(spos1, 1), getindex.(spos1, 2), getindex.(spos1, 3); aspect_ratio=:equal
);
scatter!(getindex.(spos2, 1), getindex.(spos2, 2), getindex.(spos2, 3));
scatter!(getindex.(spos3, 1), getindex.(spos3, 2), getindex.(spos3, 3));
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));
scatter!([getindex.(spivs, 1)], [getindex.(spivs, 2)], [getindex.(spivs, 3)])
scatter!([getindex(spivs[end], 1)], [getindex(spivs[end], 2)], [getindex(spivs[end], 3)])
scatter!([getindex.(tpivs, 1)], [getindex.(tpivs, 2)], [getindex.(tpivs, 3)])
