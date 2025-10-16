using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using AdaptiveCrossApproximation
using StaticArrays
using LinearAlgebra

λ = 0.3
k = 2 * pi / λ
Γ = meshrectangle(1.0, 1.0, 0.01)
Γ2 = translate(Γ, SVector(1.0, 0.0, 0.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
spaceX = raviartthomas(Γ)
spaceY = raviartthomas(Γ2)
ttree = H2Trees.TwoNTree(spaceX, 0.02)#; minvalues=200)
stree = H2Trees.TwoNTree(spaceY, 0.02)#; minvalues=200)
tree = BlockTree(ttree, stree)
##
function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return false) : (return true)
    end
end
issnear(treea, treeb, nodea, nodeb) = isnear(k, treea, treeb, nodea, nodeb)

islf = NestedCrossApproximation.islf(NestedCrossApproximation.wavenumber(op))
dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), imag(op.gamma))
values, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, issnear
);
fars
##

tol = 1e-3
testcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.iACAPivoting(spaceX.pos, spaceY.pos),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(tol)
        ),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.iACAPivoting(spaceY.pos, spaceX.pos),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(tol)
        ),
    ),
    nothing,
)
wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    spaceX,
    spaceY,
    tree;
    isnear=issnear,
    lfcompressor=AdaptiveCrossApproximation.ACA(; convergence=FNormEstimator(tol)),
    #testcompressor=testcompressor,
    #trialcompressor=trialcompressor,
    maxrank=100,
    #ntasks=16,
);

##
x = rand(ComplexF64, size(wnca, 2))

A = assemble(op, spaceX, spaceY)
##
norm(A * x - wnca * x) / norm(A * x)
norm(A * x - multi(wnca, x)) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(adjoint(A) * x - adjoint(wnca) * x) / norm(adjoint(A) * x)
##
for (n, basis) in wnca.nestedtestbases
    #println(H2Trees.levelindex(tree.testcluster, n))
    if H2Trees.levelindex(tree.testcluster, n) == 3
        println(n)
        println(keys(basis))
    end
end
for (n, basis) in wnca.nestedtrialbases
    #println(H2Trees.levelindex(tree.testcluster, n))
    if H2Trees.levelindex(tree.trialcluster, n) == 3
        println(n)
    end
end

collect(H2Trees.children(tree.testcluster, 97))
NestedCrossApproximation.parent(dtree, 1457)

wnca.testtransfermatrices[5][106]

##
fh2 = reconstruct(wnca)
##
norm(A - fh2) / norm(A)
norm(A * x - fh2 * x) / norm(A * x)
##
wnca.couplingmatrices[1].row_basis
wnca.lowfrequencyinteractions

##
fullh2mat = zeros(ComplexF64, size(A, 1), size(A, 2));

for couple in wnca.couplingmatrices
    t = couple.row_basis
    s = couple.col_basis
    ridcs = H2Trees.values(tree.testcluster, t)
    cidcs = H2Trees.values(tree.trialcluster, s)

    if tree.testcluster.nodes[t].firstchild == 0 &&
        tree.trialcluster.nodes[s].firstchild == 0
        blk =
            wnca.nestedtestbases[t][couple.dir[1]].T *
            couple.Z *
            wnca.nestedtrialbases[s][couple.dir[2]].T
        fullh2mat[ridcs, cidcs] = blk
        err = norm(blk - A[ridcs, cidcs]) / norm(A[ridcs, cidcs])
        if err > 1e-3
            println(err)
        end
    end
end
##
for ts in tp
    println(tp)
end

##
wnca.testtransfermatrices
for couple in wnca.couplingmatrices
    t = couple.row_basis
    s = couple.col_basis
    ridcs = H2Trees.values(tree.testcluster, t)
    cidcs = H2Trees.values(tree.trialcluster, s)

    if tree.testcluster.nodes[t].firstchild != 0 &&
        tree.trialcluster.nodes[s].firstchild != 0
        if haskey(wnca.testtransfermatrices[5], t)
            tt = wnca.testtransfermatrices[5][t]
            ss = wnca.trialtransfermatrices[5][s]
            tc = collect(H2Trees.children(tree.testcluster, t))
            tb =
                wnca.nestedtestbases[tc[1]][NestedCrossApproximation.parent(
                    wnca.dtree, couple.dir[1]
                )].T * tt[couple.dir[1]].T[1]
            for i in 2:length(tc)
                tb = vcat(
                    tb,
                    wnca.nestedtestbases[tc[i]][NestedCrossApproximation.parent(
                        wnca.dtree, couple.dir[1]
                    )].T * tt[couple.dir[1]].T[i],
                )
            end

            sc = collect(H2Trees.children(tree.trialcluster, s))
            sb =
                ss[couple.dir[2]].T[1] *
                wnca.nestedtrialbases[sc[1]][NestedCrossApproximation.parent(
                    wnca.dtree, couple.dir[2]
                )].T
            for i in 2:length(sc)
                sb = hcat(
                    sb,
                    ss[couple.dir[2]].T[i] *
                    wnca.nestedtrialbases[sc[i]][NestedCrossApproximation.parent(
                        wnca.dtree, couple.dir[2]
                    )].T,
                )
            end

            blk = tb * couple.Z * sb
            fullh2mat[ridcs, cidcs] = blk
            err = norm(blk - A[ridcs, cidcs]) / norm(A[ridcs, cidcs])
            if err > 1e-3
                println("t", t, "s", s)
                println(err)
            end
        end
    end
end
##
tp[122][1367][1]

num = 0
rankk = 0
maxx = 0
minn = 100
for node in H2Trees.LevelIterator(tree.testcluster, 4)
    if isassigned(tp, node)
        for (dir, pivs) in tp[node]
            num += 1
            rankk += length(pivs[1])

            if length(pivs[1]) < minn
                minn = length(pivs[1])
            end
            if length(pivs[1]) > maxx
                println(node, ", ", dir)
                maxx = length(pivs[1])
            end
        end
    end
end
rankk / num
maxx
minn
#num
##
islf = NestedCrossApproximation.islf(k)
dirfars = NestedCrossApproximation.testfars(dtree, tree, fars, dirs, islf)

dirfars[4][113][1367]

##
t = H2Trees.values(tree.testcluster, 113)
ct = tree.testcluster.nodes[113].data.center
hs = tree.testcluster.nodes[113].data.halfsize
for s in dirfars[4][113][1367]
    println(s, ", ", norm(ct - tree.trialcluster.nodes[s].data.center))
    println(issnear(tree.testcluster, tree.testcluster, 113, s))
end
##

issnear(tree.testcluster, tree.trialcluster, 113, 20)
##
NestedCrossApproximation.parent(dtree, 694)

H2Trees.parent(tree.testcluster, 122)
###
for near in wnca.nearinteractions.blocks
    fullh2mat[near.rowindices, near.colindices] = near.matrix
end
##
norm(A - fullh2mat) / norm(A)
##
tree.testcluster.nodes[1].firstchild
##
islf = NestedCrossApproximation.islf(NestedCrossApproximation.wavenumber(op))
dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), imag(op.gamma))
values, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, issnear
)
lfvalues

##
dirs
@time tf = NestedCrossApproximation.testfars(dtree, tree, fars, dirs, islf)
fars
lfvalues
tf[2]
tf[3]
tf[4]

tf[3]
##
tf[3][55]
collect(H2Trees.children(tree.testcluster, 55))
tf[4][59]
NestedCrossApproximation.parent(dtree, 1711)
using StaticArrays
dir = NestedCrossApproximation.direction(
    SVector(0.0, 0.0, 0.0) - SVector(1.0, 1.0, 1.0), dtree, 4
)

dtree.level

##
γ = meshsphere(1.0, 0.012)
length(raviartthomas(γ))
