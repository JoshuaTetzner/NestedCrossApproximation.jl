using BEAST
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using ClusterTrees
using LinearAlgebra

λ = 0.25
k = 2 * pi / λ
ηₗ = 1.0
ηₕ = 4.0
Γ = meshsphere(1.0, 0.025)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
println("N = ", length(space.pos))
tree = create_tree(space.pos, BoxTreeOptions(; nmin=100))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, hffars, lffars = NestedCrossApproximation.computeinteractionshf(
    blktree, k; ηₗ=ηₗ, ηₕ=ηₕ
);
hffars
lffars[end]
##
#=
kmtree = create_tree(space.pos, BoxTreeOptions(; nmin=50))
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
blktree = ClusterTrees.BlockTrees.BlockTree(kmtree, kmtree)
nears, fars = computeinteractions(blktree; η=0.4)
fars

nca = NestedCrossApproximation.PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    maxrank=40,
    tol=1e-5,
    multithreading=true,
    η=1.0,
)
fnca = NestedCrossApproximation.fullmat(nca)

A = assemble(op, space, space)
norm(A - fnca) / norm(A)=#
##

@time wnca = NestedCrossApproximation.WidebandNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    ηₕ=5.0,
    #testcompressor=LRF.ACA(),
    #trialcompressor=LRF.ACA(),
)

x = rand(size(wnca, 2))
wnca * x

A = assemble(op, space, space)

norm(wnca * x - A * x) / norm(A * x)
##
nca = PetrovGalerkinNCA(op, space, space; testtree=tree, trialtree=tree)
##
lens = Int[]
for i in eachindex(wnca)
    if isassigned(wnca, i)
        if length(value(tree, i)) > 60
            ldirs = length(wnca[i])
            ran = 0
            for el in wnca[i]
                ran += length(el[2][1])
            end
            push!(lens, Int(round(ran / ldirs)))
        end
    end
end
maximum(lens)
minimum(lens)
##
lenss = 0
i = 0
length(value(tree, nca.couplingmatrices[1].row_basis))

for c in nca.couplingmatrices
    if length(value(tree, c.row_basis)) > 50 && length(value(tree, c.col_basis)) > 50
        i += 2
        lenss += size(c.Z, 1) + size(c.Z, 2)
    end
end
lenss / i
##
A = assemble(op, space, space)
##
x = rand(ComplexF64, size(wnca, 2))
wnca * x

norm(A * x - wnca * x) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(A * x - wnca * x) / norm(A * x)
##

##
function lbases(h2mat::NestedCrossApproximation.WidebandNCA{K}) where {K}
    trialbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.trial_cluster.nodes))
    testbases = Vector{Dict{Int,Matrix{K}}}(undef, length(h2mat.tree.test_cluster.nodes))
    for (ind, b) in h2mat.nestedtestbases
        testbases[ind] = b
    end
    for (ind, b) in h2mat.nestedtrialbases
        trialbases[ind] = b
    end

    for level in reverse(h2mat.trialtransfermatrices)
        for (node, transferdict) in level
            recbases = Matrix{K}[]
            for (dir, dirtransfers) in transferdict
                dirbases = Matrix{ComplexF64}[]
                pdir = ClusterTrees.parent(h2mat.dtree, dir)
                childs = collect(ClusterTrees.children(h2mat.tree.trial_cluster, node))
                for (i, transe) in enumerate(dirtransfers)
                    push!(dirbases, transe * trialbases[childs[i]][pdir])
                end
                push!(recbases, reduce(hcat, dirbases))
            end
            trialbases[node] = Dict(keys(transferdict) .=> recbases)
        end
    end

    for level in reverse(h2mat.testtransfermatrices)
        for (node, transferdict) in level
            recbases = Matrix{K}[]
            for (dir, dirtransfers) in transferdict
                dirbases = Matrix{ComplexF64}[]
                pdir = ClusterTrees.parent(h2mat.dtree, dir)
                childs = collect(ClusterTrees.children(h2mat.tree.test_cluster, node))
                for (i, transe) in enumerate(dirtransfers)
                    push!(dirbases, testbases[childs[i]][pdir] * transe)
                end
                push!(recbases, reduce(vcat, dirbases))
            end
            testbases[node] = Dict(keys(transferdict) .=> recbases)
        end
    end

    return testbases, trialbases
end

function fullmat(h2mat::NestedCrossApproximation.WidebandNCA{K}) where {K}
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for M in h2mat.nearinteractions.blocks
        A_h2[M.rowindices, M.colindices] = M.matrix
    end
    for blk in h2mat.lowfrequencyinteractions
        A_h2[blk.τ, blk.σ] = blk.M.U * blk.M.V
    end

    testbases, trialbases = lbases(h2mat)

    for i2o in h2mat.couplingmatrices
        A_h2[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ] =
            testbases[i2o.row_basis][i2o.row_dir] *
            i2o.Z *
            trialbases[i2o.col_basis][i2o.col_dir]
    end
    return A_h2
end

function fullmat2(h2mat::NestedCrossApproximation.WidebandNCA{K}, A) where {K}
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    Are = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for M in h2mat.nearinteractions.blocks
        A_h2[M.rowindices, M.colindices] = M.matrix
        Are[M.rowindices, M.colindices] = A[M.rowindices, M.colindices]
    end
    for blk in h2mat.lowfrequencyinteractions
        A_h2[blk.τ, blk.σ] = blk.M.U * blk.M.V
        Are[blk.τ, blk.σ] = A[blk.τ, blk.σ]
    end

    testbases, trialbases = lbases(h2mat)

    for (i, i2o) in enumerate(h2mat.couplingmatrices)
        τ = value(h2mat.tree.test_cluster, i2o.row_basis)
        σ = value(h2mat.tree.trial_cluster, i2o.col_basis)
        if ClusterTrees.haschildren(h2mat.tree.test_cluster, i2o.row_basis)
            τ = value(
                tree, collect(ClusterTrees.children(h2mat.tree.test_cluster, i2o.row_basis))
            )
        end
        if ClusterTrees.haschildren(h2mat.tree.trial_cluster, i2o.col_basis)
            σ = value(
                tree,
                collect(ClusterTrees.children(h2mat.tree.trial_cluster, i2o.col_basis)),
            )
        end

        A_h2[τ, σ] =
            testbases[i2o.row_basis][i2o.row_dir] *
            i2o.Z *
            trialbases[i2o.col_basis][i2o.col_dir]
        Are[τ, σ] = A[τ, σ]
        err = norm(Are[τ, σ] - A_h2[τ, σ]) / norm(Are[τ, σ])
        if err > 1.0 &&
            !ClusterTrees.haschildren(h2mat.tree.test_cluster, i2o.row_basis) &&
            !ClusterTrees.haschildren(h2mat.tree.trial_cluster, i2o.col_basis)
            println(i, ": ", i2o.row_basis, ", ", i2o.col_basis)
        end
    end
    return A_h2, Are
end

##
Ah2, Are = fullmat2(wnca, A)
norm(Ah2 - Are) / norm(Ah2)
##
ClusterTrees.haschildren(wnca.tree.test_cluster, 29)
for far in hffars[4]
    if far == inter
        println("found")
    end
end

wnca.nestedtestbases[29]

wnca.testtransfermatrices[1][132]

testbases, trialbases = lbases(wnca)
##
interaction = (132, 29)
S = wnca.couplingmatrices[30481]
##
NestedCrossApproximation.direction()

U = testbases[132][S.row_dir]
V = trialbases[29][S.col_dir]

S.Z

blk = A[
    value(wnca.tree.test_cluster, interaction[1]),
    value(wnca.tree.trial_cluster, interaction[2]),
]

U * S.Z * V - blk

for inter in wnca.testtransfermatrices[1][132][S.row_dir]
    println(size(inter))
end

ClusterTrees.children(ztz)

hffars
