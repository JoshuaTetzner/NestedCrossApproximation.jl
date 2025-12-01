using BEAST
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes
##

#λ = 0.6
#k = 2 * pi / λ
#h = 0.04
#k = 0.15 / h
λ = 0.2
k = 2 * pi / λ
#path = "/home/jt286/.julia/dev/NestedCrossApproximation/test/geo/ellipsoid.geo"
#pathsave = "/home/jt286/.julia/dev/NestedCrossApproximation/test/geo/ellipsoid.msh"
#run(`gmsh $path -2 -clmax $h -format msh2 -o $pathsave`)
Γ = meshsphere(1.0, λ / 10)#CompScienceMeshes.read_gmsh_mesh(pathsave)

##
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
length(space)
##
tree = H2Trees.TwoNTree(space, space, 0.01; minvaluestest=50, minvaluestrial=50);
#ttree = KMeansTree(space.pos, 2; minvalues=100)
#tree = H2Trees.BlockTree(ttree, ttree)
##
dtree = NestedCrossApproximation.𝒟tree(
    H2Trees.halfsize(tree.testcluster),
    NestedCrossApproximation.maxlevel(tree.trialcluster, NestedCrossApproximation.islf(k)),
)
Ft, eₜ, Fs, eₛ = NestedCrossApproximation.directionalfarinteractions(
    tree, dtree; isnear=NestedCrossApproximation.isnear(k)
);

##
hfinteractions = Int[]
lfinteractions = Int[]
for level in H2Trees.levels(tree.testcluster)
    lfinteraction = 0
    hfinteraction = 0
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        interactions = length(Ft[t])
        if interactions > 0
            if eₜ[t][1] == 0
                lfinteraction += interactions
            else
                hfinteraction += interactions
            end
        end
    end
    push!(lfinteractions, lfinteraction)
    push!(hfinteractions, hfinteraction)
end

hfinteractions
lfinteractions

##
values, nearvalues = H2Trees.nearinteractions(
    tree; isnear=NestedCrossApproximation.isnear(k), extractselfvalues=false
)

nearstorage = 0
for i in eachindex(values)
    nearstorage += length(values[i]) * length(nearvalues[i])
end
println(nearstorage * 8 / 10^6, " MB")
##
function admissiblelevel(Ft, tree)
    lflevel = 0
    hflevel = 0
    for level in H2Trees.levels(tree.testcluster)
        for t in H2Trees.LevelIterator(tree.testcluster, level)
            if length(Ft[t]) > 0
                if eₜ[t][1] == 0
                    lflevel += 1
                    break
                else
                    hflevel += 1
                    break
                end
            end
        end
    end
    return max(lflevel, hflevel)
end

@time admissiblelevel(Ft, tree)

function settolerace(
    comp::TopDownCompressor{RP,LRF}
) where {RP<:iACA,LRF<:LowRankFactorization}
    return comp.lrf.convergence.estimator.tol = 1e-4
end
