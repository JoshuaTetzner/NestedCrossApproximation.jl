using BEAST
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes
##

λ = 0.6
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.03)
##
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, space, 0.01; minvaluestest=200, minvaluestrial=200)
length(space)
##
dtree = NestedCrossApproximation.𝒟tree(
    H2Trees.halfsize(tree.testcluster),
    NestedCrossApproximation.maxlevel(tree.trialcluster, NestedCrossApproximation.islf(k)),
)
Ft, eₜ, Fs, eₛ = NestedCrossApproximation.directionalfarinteractions(
    tree, dtree; isnear=NestedCrossApproximation.isnear(k)
)

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
