using Test
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes
##
λ = 1.0
k = 2π / λ

Γ = meshicosphere(20, 1.0)
pts = vertices(Γ)
ttree = H2Trees.TwoNTree(pts, 1 / 2^8; minvalues=3)
stree = H2Trees.TwoNTree(pts, 1 / 2^8; minvalues=3)
tree = BlockTree(ttree, stree)
testfarptr, trialfarptr, testfars, trialfars = NestedCrossApproximation.farinteractions(
    tree; isnear=NestedCrossApproximation.isnear()
)

tdirdata = NestedCrossApproximation.build_twondirectionaldata(
    stree, ttree, testfarptr, testfars, NestedCrossApproximation.islf(k)
)

node  = 10
first = Int(tdirdata.farptr[node])
last  = Int(tdirdata.farptr[node + 1]) - 1
for i in first:last
    far_node = Int(tdirdata.fars[i])
    local_dir = Int(tdirdata.fardir[i])   # local direction id at `node`
    ref = NestedCrossApproximation.directionref(tdirdata, node, local_dir)
    @show i far_node local_dir ref
end
#-----------------------------------------------------
# BoundingBall detailed isbasis checks
# -----------------------------------------------------------------------------

Random.seed!(321)
pts3 = [@SVector rand(3) for _ in 1:512]
ttree3 = H2Trees.KMeansTree(pts3, 2; minvalues=8)
stree3 = H2Trees.KMeansTree(pts3, 2; minvalues=8)

lvls3 = collect(H2Trees.levels(ttree3))
@test length(lvls3) >= 4

pnode3 = let found = 0
    for node in H2Trees.LevelIterator(ttree3, lvls3[1])
        found = Int(node)
        break
    end
    found == 0 && error("No node found at level $(lvls3[1]).")
    found
end

cnode3 = let found = 0
    for child in H2Trees.children(ttree3, pnode3)
        ichild = Int(child)
        found = ichild
        break
    end
    found == 0 && error("No high-frequency child found for node $(pnode3).")
    found
end

gnode3 = let found = 0
    for child in H2Trees.children(ttree3, cnode3)
        ichild = Int(child)
        found = ichild
        break
    end
    found == 0 && error("No high-frequency child found for node $(cnode3).")
    found
end

sparent3 = let lvl = H2Trees.level(ttree3, pnode3), found = 0
    for node in H2Trees.LevelIterator(stree3, lvl)
        found = Int(node)
        break
    end
    found == 0 && error("No node found at level $(lvl).")
    found
end

sgrand3 = let lvl = H2Trees.level(ttree3, gnode3), found = 0
    for node in H2Trees.LevelIterator(stree3, lvl)
        found = Int(node)
        break
    end
    found == 0 && error("No node found at level $(lvl).")
    found
end

islf_gap3 = (tree, node) -> H2Trees.level(tree, node) >= (H2Trees.level(ttree3, gnode3) + 2)

farptr_gap3 = Vector{Int}(undef, H2Trees.numberofnodes(ttree3) + 1)
fars_gap3 = Int[]
farptr_gap3[1] = 1
for node in 1:H2Trees.numberofnodes(ttree3)
    if node == pnode3
        append!(fars_gap3, [sparent3])
    elseif node == gnode3
        append!(fars_gap3, [sgrand3])
    end
    farptr_gap3[node + 1] = length(fars_gap3) + 1
end
data_gap3 = build_boundingballdirectionaldata(
    ttree3, stree3, farptr_gap3, fars_gap3; k=4.0, islf=islf_gap3
)

# Hierarchy must bridge a node without local fars.
@test data_gap3.farptr[cnode3] == data_gap3.farptr[cnode3 + 1]
@test ndirections(data_gap3, cnode3) > 0

pfirst3 = Int(data_gap3.farptr[pnode3])
@test pfirst3 <= Int(data_gap3.farptr[pnode3 + 1] - 1)
pdir_far3 = Int(data_gap3.fardir[pfirst3])
@test pdir_far3 > 0
@test paternaldirection(data_gap3, cnode3, pdir_far3) > 0

@test _has_descendant_far_for_paternal_dir(
    data_gap3, ttree3, pnode3, pdir_far3; islf=islf_gap3
)
@test !isbasis(data_gap3, ttree3, pnode3, pdir_far3; islf=islf_gap3)

prunable3 = Int[]
continuing3 = Int[]
for dir in directions(data_gap3, pnode3)
    idir = Int(dir)
    if isbasis(data_gap3, ttree3, pnode3, idir; islf=islf_gap3)
        push!(prunable3, idir)
    else
        push!(continuing3, idir)
    end
end
@test !isempty(prunable3)
@test !isempty(continuing3)
for dir in prunable3
    @test !_has_descendant_far_for_paternal_dir(
        data_gap3, ttree3, pnode3, dir; islf=islf_gap3
    )
end
for dir in continuing3
    @test _has_descendant_far_for_paternal_dir(
        data_gap3, ttree3, pnode3, dir; islf=islf_gap3
    )
end

# Rule 2: HF node with LF children (no mixed children allowed in this setup).
rule2node3 = cnode3
islf_rule23 =
    (tree, node) -> H2Trees.level(tree, node) >= (H2Trees.level(ttree3, rule2node3) + 1)
srule23 = let lvl = H2Trees.level(ttree3, rule2node3), found = 0
    for node in H2Trees.LevelIterator(stree3, lvl)
        found = Int(node)
        break
    end
    found == 0 && error("No node found at level $(lvl).")
    found
end
farptr_r23 = Vector{Int}(undef, H2Trees.numberofnodes(ttree3) + 1)
fars_r23 = Int[]
farptr_r23[1] = 1
for node in 1:H2Trees.numberofnodes(ttree3)
    node == rule2node3 && append!(fars_r23, [srule23])
    farptr_r23[node + 1] = length(fars_r23) + 1
end
data_r23 = build_boundingballdirectionaldata(
    ttree3, stree3, farptr_r23, fars_r23; k=4.0, islf=islf_rule23
)

r23first = Int(data_r23.farptr[rule2node3])
@test r23first <= Int(data_r23.farptr[rule2node3 + 1] - 1)
r23dir = Int(data_r23.fardir[r23first])
@test !H2Trees.isleaf(ttree3, rule2node3)
@test !islf_rule23(ttree3, rule2node3)
kids3 = collect(H2Trees.children(ttree3, rule2node3))
@test !isempty(kids3)
@test all(c -> islf_rule23(ttree3, Int(c)), kids3)
@test isbasis(data_r23, ttree3, rule2node3, r23dir; islf=islf_rule23)

# Rule 1: leaves always require a basis.
leaf3 = let found = 0
    for node in 1:H2Trees.numberofnodes(ttree3)
        if H2Trees.isleaf(ttree3, node)
            found = node
            break
        end
    end
    found == 0 && error("No leaf node found.")
    found
end
@test isbasis(data_r23, ttree3, leaf3, 1; islf=islf_rule23)
