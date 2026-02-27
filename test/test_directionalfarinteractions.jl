using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using CompScienceMeshes
using Test
using BEAST
using Random
using LinearAlgebra

#meshes = "/home/jt286/Documents/Geometries/typhoon_simple/typhoon_simple_gmsh_refined.msh"
Γ = meshsphere(1.0, 0.02)#CompScienceMeshes.read_gmsh_mesh(meshes;)
RT = raviartthomas(Γ)

_, _, h = edgeinfo(Γ)
numfcts = length(RT)
λ = 20h
k = 2 * pi / λ

ttree = KMeansTree(RT.pos, 2; minvalues=100)
stree = KMeansTree(RT.pos, 2; minvalues=100)

tree = BlockTree(ttree, stree)
isnear = NestedCrossApproximation.isnear(k)

values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
tfars = NestedCrossApproximation.farinteractions(ttree, stree; isnear=isnear)
sfars = NestedCrossApproximation.farinteractions(stree, ttree; isnear=isnear)
##
for (t, tfar) in enumerate(tfars)
    for s in tfar
        @test t ∈ sfars[s]
    end
end

for (s, sfar) in enumerate(sfars)
    for t in sfar
        @test s ∈ tfars[t]
    end
end

A = zeros(Bool, numfcts, numfcts)
for i in eachindex(values)
    @test !any(A[values[i], nearvalues[i]])
    A[values[i], nearvalues[i]] .= true
end

for (t, tfar) in enumerate(tfars)
    tvals = H2Trees.values(ttree, t)
    for s in tfar
        svals = H2Trees.values(stree, s)
        @test !any(A[tvals, svals])
        A[tvals, svals] .= true
    end
end

@test all(A)
##
tree = TwoNTree(RT, RT, 1 / 2^20; minvaluestest=100, minvaluestrial=100)
values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
tfars = NestedCrossApproximation.farinteractions(
    tree.testcluster, tree.trialcluster; isnear=isnear
);
sfars = NestedCrossApproximation.farinteractions(
    tree.trialcluster, tree.testcluster; isnear=isnear
);
##
sto = 0
for i in eachindex(nearvalues)
    sto += length(values[i]) * length(nearvalues[i])
end
sto * 8 * 10^-9
##
for (t, tfar) in enumerate(tfars)
    for s in tfar
        @test t ∈ sfars[s]
    end
end

for (s, sfar) in enumerate(sfars)
    for t in sfar
        @test s ∈ tfars[t]
    end
end

A = zeros(Bool, numfcts, numfcts)
for i in eachindex(values)
    @test !any(A[values[i], nearvalues[i]])
    A[values[i], nearvalues[i]] .= true
end

for (t, tfar) in enumerate(tfars)
    tvals = H2Trees.values(tree.testcluster, t)
    for s in tfar
        svals = H2Trees.values(tree.trialcluster, s)
        @test !any(A[tvals, svals])
        A[tvals, svals] .= true
    end
end
@test all(A)

##
using NestedCrossApproximation
using CompScienceMeshes
using StaticArrays
using LinearAlgebra

ref = NestedCrossApproximation.sphericalfibonaccipoints(10)
pts = NestedCrossApproximation.sphericalfibonaccipoints(50)

refidx = 4

ptsidcs = Int[]
for p in eachindex(pts)
    if argmin(norm.(ref .- Scalar(pts[p]))) == 4
        push!(ptsidcs, p)
    end
end
ptsidcs
##
ref1 = [re[1] for re in ref]
ref2 = [re[2] for re in ref]
ref3 = [re[3] for re in ref]

pts1 = [pt[1] for pt in pts]
pts2 = [pt[2] for pt in pts]
pts3 = [pt[3] for pt in pts]

using PlotlyJS
tbl = [0, 0.2902, 0.6] .* 255
tbr = [1.0, 0.6, 0.145] .* 255
tbg = [0.145, 0.6, 0.145] .* 255
talkblue = "rgb($(tbl[1]),$(tbl[2]),$(tbl[3]))"
talkred = "rgb($(tbr[1]),$(tbr[2]),$(tbr[3]))"
talkgreen = "rgb($(tbg[1]),$(tbg[2]),$(tbg[3]))"

trace1 = patch(meshsphere(0.975, 0.1); color=talkblue)
# Change sphere color (gray, semi-transparent)
# make it slightly transparent

# ----------------------------
# 3D points
# ----------------------------
trace = scatter3d(;
    x=pts1, y=pts2, z=pts3, mode="markers", marker=attr(;
        size=15,
        color=talkred,      # change to any color, or use a vector for gradient
    )
)

trace2 = scatter3d(;
    x=ref1, y=ref2, z=ref3, mode="markers", marker=attr(;
        size=15,
        color=talkgreen,      # change to any color, or use a vector for gradient
    )
)

# ----------------------------
# Layout: remove axes, background, grid
# ----------------------------
layout = Layout(;
    scene=attr(;
        xaxis=attr(; visible=false),
        yaxis=attr(; visible=false),
        zaxis=attr(; visible=false),
        aspectmode="data",
        camera=attr(; eye=attr(; x=1.3, y=1.3, z=1.3)),
    ),
    margin=attr(; l=0, r=0, b=0, t=0),   # remove extra margin
)
plot([trace1, trace2], layout)
