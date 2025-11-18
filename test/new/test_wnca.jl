using BEAST
using FastBEAST
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using StaticArrays
using LinearAlgebra

##

λ = 0.2
k = 2 * pi / λ
k * h
##

h = 0.02
Γ1 = meshrectangle(1.0, 1.0, h)
Γ2 = translate(meshrectangle(1.0, 1.0, h), SVector(3.0, 0.0, 0.0))
X1 = raviartthomas(Γ1)
X2 = raviartthomas(Γ2)
op = Maxwell3D.singlelayer(; wavenumber=k)
println("N = ", length(X1))

##

tree = H2Trees.TwoNTree(X1, X2, 0.01; minvaluestest=100, minvaluestrial=100)
dtree = NestedCrossApproximation.𝒟tree(
    H2Trees.halfsize(tree.testcluster),
    NestedCrossApproximation.maxlevel(tree.trialcluster, NestedCrossApproximation.islf(k)),
)

println("Tree Level: ", length(collect(H2Trees.levels(tree.testcluster))))
println("DTree Level: ", dtree.level)

Ft, eₜ, Fs, eₛ = NestedCrossApproximation.directionalfarinteractions(
    tree, dtree; isnear=NestedCrossApproximation.isnear(k)
)
for level in H2Trees.levels(tree.testcluster)
    lfcount = 0
    hfcount = 0
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        interactions = length(Ft[t])
        if interactions > 0
            if eₜ[t][1] == 0
                lfcount += interactions
            else
                hfcount += interactions
            end
        end
    end
    println("Level $level: LF = $lfcount, HF = $hfcount")
end

##

testcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(X1.pos, X2.pos),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MimicryPivoting(X2.pos, X1.pos),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
h2mat = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    X1,
    X2,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=50,
    ntasks=1,
);
A = assemble(op, X1, X2);
##
estimate_reldifference(h2mat, A)
##
tbases, sbases = lbases(h2mat)
Fh2mat = zeros(ComplexF64, length(X1), length(X2))
lrbmat = zeros(ComplexF64, length(X1), length(X2))
for cp in h2mat.couplingmatrices
    Fh2mat[
        H2Trees.values(tree.testcluster, cp.row_basis),
        H2Trees.values(tree.trialcluster, cp.col_basis),
    ] = tbases[cp.row_basis][cp.dir] * cp.Z * sbases[cp.col_basis][cp.dir]
    lrbmat[H2Trees.values(tree.testcluster, cp.row_basis), H2Trees.values(tree.trialcluster, cp.col_basis)] = A[
        H2Trees.values(tree.testcluster, cp.row_basis),
        H2Trees.values(tree.trialcluster, cp.col_basis),
    ]
end
##
for level in H2Trees.levels(tree.testcluster)
    println("Level $level")
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        if Ft != []
            println("dirs", eₜ[t])
            tvals = H2Trees.values(tree.testcluster, t)
            for (i, s) in enumerate(Ft[t])
                svals = H2Trees.values(tree.trialcluster, s)
                Fblock = Fh2mat[tvals, svals]
                lrbblock = lrbmat[tvals, svals]
                errh2 = norm(Fblock - lrbblock) / norm(lrbblock)
                rpiv = tp[t][eₜ[t][i]][1]
                rb = A[tvals, tp[t][eₜ[t][i]][2]] * A[rpiv, tp[t][eₜ[t][i]][2]]^-1

                spiv = sp[s][eₜ[t][i]][2]
                sb = A[sp[s][eₜ[t][i]][1], spiv]^-1 * A[sp[s][eₜ[t][i]][1], svals]
                errpiv = norm(rb * A[rpiv, spiv] * sb - lrbblock) / norm(lrbblock)

                println(errh2, ", ", errpiv)
            end
        end
    end
end
##
norm(Fh2mat - lrbmat) / norm(lrbmat)
##
