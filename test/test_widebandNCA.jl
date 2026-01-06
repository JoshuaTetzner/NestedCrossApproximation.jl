using BEAST
using FastBEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
using Random

h = 0.025
λ = 20h
k = 2π / λ
##
Γ = meshsphere(1.0, h)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(3)
ttree = KMeansTree(space.pos, 2; minvalues=100)
stree = ttree
tree = BlockTree(ttree, stree)

##

testcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(space.pos, space.pos),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MimicryPivoting(space.pos, space.pos),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
@time wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    space,
    space,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=50,
    #ntasks=,
);

##
@time A = assemble(op, space, space);
x = rand(ComplexF64, length(space))
##

estimate_reldifference(wnca, A)
##
norm(wnca * x - A * x) / norm(A * x)
norm(adjoint(wnca) * x - adjoint(A) * x) / norm(adjoint(A) * x)
norm(transpose(wnca) * x - transpose(A) * x) / norm(transpose(A) * x)

##
function informations(h2mat::NestedCrossApproximation.PetrovGalerkinWNCA)
    leveltransfers = Int[]
    leveledranks = Tuple{Int,Int}[]
    leveldbases = Int[]
    for level in H2Trees.levels(h2mat.tree.testcluster)
        lrank = (100, 0)
        ntransfers = 0
        nbases = 0
        for blk in H2Trees.LevelIterator(h2mat.tree.testcluster, level)
            if haskey(h2mat.testtransfermatrices, blk)
                ntransfers += 1

                for mats in values(h2mat.testtransfermatrices[blk])
                    for mat in mats
                        lrank = (
                            min(lrank[1], size(mat[2], 1)), max(lrank[2], size(mat[2], 1))
                        )
                    end
                end
            end
            if haskey(h2mat.nestedtestbases, blk)
                nbases += 1
                for mat in values(h2mat.nestedtestbases[blk])
                    lrank = (min(lrank[1], size(mat, 2)), max(lrank[2], size(mat, 2)))
                end
            end
        end
        push!(leveledranks, lrank)
        push!(leveldbases, nbases)
        push!(leveltransfers, ntransfers)
    end

    return leveltransfers, leveldbases, leveledranks
end

leveltransfers, leveldbases, leveledranks = informations(wnca)

leveltransfers
leveldbases
leveledranks

##
fullwnca = reconstruct(wnca)

##
norm(fullwnca - A) / norm(A)
estimate_reldifference(A, fullwnca; tol=1e-4)
##

for level in H2Trees.levels(wnca.tree.testcluster)
    for t in H2Trees.LevelIterator(wnca.tree.testcluster, level)
        for (idcs, coupling) in wnca.couplingmatrices[t]
            tidcs = H2Trees.values(wnca.tree.testcluster, t)
            sidcs = H2Trees.values(wnca.tree.trialcluster, idcs[2])
            err = norm(fullwnca[tidcs, sidcs] - A[tidcs, sidcs]) / norm(A[tidcs, sidcs])
            #println(err)
            if err > 1e-2
                println("level $level between clusters $t and $(idcs[2])., error = $err")
            end
        end
    end
end
##
wnca.trialtransfermatrices[466]

wnca.couplingmatrices[899][(899, 466)][1]
wnca.couplingmatrices[650][(650, 466)][1]
wnca.couplingmatrices[896][(896, 466)][1]

wnca.testtransfermatrices[899]
chds = collect(H2Trees.children(wnca.tree.testcluster, 899))
wnca.nestedtestbases[900]
wnca.nestedtestbases[901]

H2Trees.radius(wnca.tree.testcluster, 899)
H2Trees.radius(wnca.tree.testcluster, 900)
H2Trees.radius(wnca.tree.testcluster, 901)

wnca.couplingmatrices[899]
norm(wnca.testtransfermatrices[899][1][1][2])
norm(wnca.testtransfermatrices[899][1][2][2])
tidcs = H2Trees.values(wnca.tree.testcluster, 899)
sidcs = H2Trees.values(wnca.tree.trialcluster, 466)
blk = A[tidcs, sidcs] - fullwnca[tidcs, sidcs]
norm(blk - wncablk ./ 2) / norm(blk)

using AdaptiveCrossApproximation
U, V = AdaptiveCrossApproximation.aca(blk; tol=1e-3)
norm(U * V - blk) / norm(blk)

wnca.couplingmatrices[899][(899, 466)]

children()

##
tdata.F[899]
tdata.𝓔[899]
sdata.F[466]
sdata.𝓔[466]
collect(H2Trees.children(wnca.tree.testcluster, 899))
sdata.𝓔map[468]
sdata.𝓔[468]

wnca.nestedtestbases[900]

wnca.nestedtrialbases[467]
##
tc = H2Trees.center(wnca.tree.testcluster, 899)
sc1 = H2Trees.center(wnca.tree.trialcluster, 456)
sc2 = H2Trees.center(wnca.tree.trialcluster, 463)
sc3 = H2Trees.center(wnca.tree.trialcluster, 466)
r1 = H2Trees.radius(wnca.tree.trialcluster, 456)
r2 = H2Trees.radius(wnca.tree.trialcluster, 463)
r3 = H2Trees.radius(wnca.tree.trialcluster, 466)

norm(tc - sc1)
norm(tc - sc2)
norm(tc - sc3)
##
radien = Tuple{Float64,Float64}[]
for level in H2Trees.levels(wnca.tree.testcluster)
    println("Level: $level")
    radii = (100.0, 0.0)
    for t in H2Trees.LevelIterator(wnca.tree.testcluster, level)
        if H2Trees.radius(wnca.tree.testcluster, t) < radii[1]
            radii = (H2Trees.radius(wnca.tree.testcluster, t), radii[2])
        end
        if H2Trees.radius(wnca.tree.testcluster, t) > radii[2]
            radii = (radii[1], H2Trees.radius(wnca.tree.testcluster, t))
        end
    end
    push!(radien, radii)
end

radien

H2Trees.siblings(wnca.tree.testcluster, 900)
