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
@inline roundvec(v; digits=1) = clean(
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

@inline function cellnormal(verts, cell)
    return cross(verts[cell[2]] - verts[cell[1]], verts[cell[3]] - verts[cell[1]])
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
        cn[c] = clean(unit(cellnormal(verts, cells[c])))
    end

    @inbounds for i in eachindex(rwg.fns)
        fn = rwg.fns[i]
        sh = fn[1]

        cell = cells[sh.cellid]
        a, b = edgeverts(cell, sh.refid)

        ℓ[i] = canon(unit(verts[b] - verts[a]))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            if abs(dot(ni, nj)) < 0.9
                n[i] = zero(ni)
                continue
            end
            ni += dot(ni, nj) < 0 ? -nj : nj
        end
        n[i] = clean(unit(ni))
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
Γ = meshsphere(1.0, 0.05)#meshcuboid(1.0, 1.0, 1.0, 0.0275)

RT = raviartthomas(Γ)
println("Size RT ", length(RT))
h = edgeinfo(Γ)[3]
λ = 10h
k = 2 * pi / λ

op = Maxwell3D.singlelayer(; wavenumber=k)#
Random.seed!(1)
testtree = KMeansTree(
    RT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
#testtree = TwoNTree(RT, 0.0; minvalues=200)
Random.seed!(1)
trialtree = KMeansTree(
    RT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
#trialtree = TwoNTree(RT, 0.0; minvalues=200)
tree = H2Trees.BlockTree(testtree, trialtree)

isnear = NestedCrossApproximation.isnearwideband(k; ηhf=5.0, γ=1.0);

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
);
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
function fullfarmat(A, h2mat)
    nears = h2mat.nearinteractions
    for i in eachindex(nears.blocks)
        A[nears.rowindices[i], nears.colindices[i]] .= 0.0
    end
    return A
end

A = fullfarmat(A, h2mat)
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
    3331,
    3332,
    3333,
    3334,
    3335,
    3336,
    3337,
    3338,
    3339,
    3340,
    3341,
    3342,
    3343,
    3344,
    3345,
    3346,
    3347,
    3348,
    3349,
    3350,
    3351,
    3352,
    3353,
    3354,
    3355,
    3356,
    3357,
    3358,
    3359,
    3360,
    3361,
    3362,
    3363,
    3364,
    3365,
    3366,
    3367,
    3368,
    3369,
    3370,
    3371,
    3372,
    3373,
    3374,
    3375,
    3376,
    3377,
    3378,
    3379,
    3380,
    3381,
    3382,
    3383,
    3384,
    3385,
    3386,
    3387,
    3388,
    3389,
    3390,
    3391,
]
Ft = [77]
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
    xlims=(-1, 1),
    ylims=(-1, 1),
    zlims=(-1, 1),
);
scatter!(getindex.(tpos, 1), getindex.(tpos, 2), getindex.(tpos, 3));

display(p)
##
tol = 1e-3
cpiv = AdaptiveCrossApproximation.TreeMimicryPivoting2(
    RT.pos, RT.pos, edgeids, trialnormalids, trial_node_normal_sets, trialtree
)
trial_node_normal_sets[77]
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
##
for i in 1:n
    println(
        norm(blk - A[tidcs, c[1:i]] * inv(A[r[1:i], c[1:i]]) * A[r[1:i], sidcs]) / norm(blk)
    )
end

##
U, V = AdaptiveCrossApproximation.aca(blk; tol=1e-3);
norm(blk - U * V) / norm(blk)
##
size(U)
r = tidcs[[1, 28, 10, 47, 52, 50, 3, 5, 36, 37, 26]]
c = sidcs[[120, 901, 52, 1702, 1581, 119, 54, 1748, 896, 1068, 50]]
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
    xlims=(-1, 1),
    ylims=(-1, 1),
    zlims=(-1, 1),
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
