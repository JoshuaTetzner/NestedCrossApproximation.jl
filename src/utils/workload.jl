import H2Trees: testtree, trialtree, numberofnodes, levels, LevelIterator, isleaf

# Dry runs that size the assembly progress bars: each has to total exactly what its
# traversal increments.

# Number of clusters with a far field of their own or of an ancestor.
function _farworkload(tree, fardata::FarData)
    ptr = farptr(fardata)
    active = falses(numberofnodes(tree))
    workload = 0
    for level in levels(tree)
        for t in LevelIterator(tree, level)
            ownfar = ptr[t + 1] > ptr[t]
            parentactive = level == 1 ? false : active[H2Trees.parent(tree, t)]
            if ownfar || parentactive
                active[t] = true
                workload += 1
            end
        end
    end
    return workload
end

_unitdirweight(::Int, ::Int) = 1

# Directional analogue: sums `weight(level, dir)` over the (node, dir) pairs with a
# nonempty `dirfarfield`.
function _farworkload(tree, fardata::DirectionalData, weight::W=_unitdirweight) where {W}
    workload = 0
    for level in levels(tree)
        for t in LevelIterator(tree, level)
            for diridx in dirrange(fardata, t)
                dir = Int(fardata.dirs[diridx])
                isempty(dirfarfield(tree, fardata, t, dir)) && continue
                workload += weight(level, dir)
            end
        end
    end
    return workload
end

# One method per compressor and far data type, see `testbases`.
function _basesworkload end

# The work per level is roughly constant, so a cluster counts nleaves ÷ (clusters on its
# level), the `levelstep` of topdown.jl.
function _basesworkload(::TopDown, tree, fardata::FarData)
    ptr = farptr(fardata)
    active = falses(numberofnodes(tree))
    nleaves = count(n -> isleaf(tree, n), 1:numberofnodes(tree))
    workload = 0
    for level in levels(tree)
        leveliter = LevelIterator(tree, level)
        levelstep = max(1, nleaves ÷ length(leveliter))
        for t in leveliter
            ownfar = ptr[t + 1] > ptr[t]
            parentactive = level == 1 ? false : active[H2Trees.parent(tree, t)]
            if ownfar || parentactive
                active[t] = true
                workload += levelstep
            end
        end
    end
    return workload
end

# One unit per active cluster.
_basesworkload(::BottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)

_basesworkload(::BatchedBottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)

# One unit per LF pair and the per-level step of TopDown per HF pair, as in bottomup.jl.
function _basesworkload(::BottomUp, tree, fardata::DirectionalData)
    nleaves = count(n -> isleaf(tree, n), 1:numberofnodes(tree))
    hfsteps = Dict(
        level => max(1, nleaves ÷ length(LevelIterator(tree, level))) for
        level in levels(tree)
    )
    return _farworkload(tree, fardata, (level, dir) -> dir == 0 ? 1 : hfsteps[level])
end

# Bases and coupling matrices have unrelated costs per unit and get separate bars.
function _basesprogress(
    testcompressor, trialcompressor, tree, testfardata, trialfardata, verbose::Bool
)
    workload =
        _basesworkload(testcompressor, testtree(tree), testfardata) +
        _basesworkload(trialcompressor, trialtree(tree), trialfardata)
    return Progress(workload; desc="Assembling far bases: ", enabled=verbose)
end

function _couplingprogress(testfardata, verbose::Bool)
    return Progress(
        length(farnodes(testfardata)); desc="Assembling far coupling: ", enabled=verbose
    )
end
