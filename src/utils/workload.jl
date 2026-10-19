import H2Trees: testtree, trialtree, numberofnodes, levels, LevelIterator, isleaf

# Every dry run used to size an assembly progress bar, collected here because they only
# make sense against one another: each has to total exactly what its traversal
# increments, and `_basesworkload` picks a per-compressor cost model on top of the same
# two `_farworkload` passes.

# Counts clusters that will actually do work in testbases/trialbases: a node is
# "active" if it has its own raw far nodes, or any ancestor does (the latter gets
# inherited via the parent's pivots once the parent is processed). This is exactly
# equivalent to `!isempty(farfield(tree, fardata, node))` but O(1) per node instead
# of walking the ancestor chain, so it's cheap enough to run as a dry-run pass
# before the real (TopDown or BottomUp) traversal to size the progress bar — and it
# naturally excludes levels with no interactions instead of weighting every level
# by its raw cluster count.
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

# One unit of progress per active (node, dir) pair.
_unitdirweight(::Int, ::Int) = 1

# Per-direction analogue of `_farworkload(tree, fardata::FarData)`: counts
# (node, dir) pairs that will actually do work in the BottomUp DirectionalData
# testbases/trialbases traversal. The LF/HF inheritance rules folded into
# `dirfarfield` (climb-while-LF vs dirmap-intersection) are intricate enough that
# re-deriving a static propagation risks silently diverging from the real
# traversal, so this dry run just reuses `dirfarfield` itself — correct by
# construction, and its cost (bounded by tree depth, not by ACA/IACA pivoting) is
# negligible next to the real factorization work it's sizing a bar for.
#
# `weight(level, dir)` scores each active pair, so a compressor whose per-pair cost
# is not uniform (see `_basesworkload(::BottomUp, tree, ::DirectionalData)`) can reuse
# this traversal instead of restating it.
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

# The dry-run weight sizing the bases progress bar, one method per compressor and
# fardata type it supports: see the contract in compressors/abstractcompressor.jl.
function _basesworkload end

# TopDown descends root-to-leaves: each active level has more clusters than the
# last, but each one is correspondingly cheaper (less work per cluster as Ftvalues
# narrows), so wall time per active level is roughly constant — about nleaves(tree)
# coupling-matrix-equivalents — rather than scaling with that level's cluster count.
# Mirrors the `levelactive`/`next!(pbar; step=nleaves)` runtime logic in topdown.jl so
# the dry-run total matches exactly what gets incremented.
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

# BottomUp: each active cluster costs about one coupling-matrix-equivalent
# (computing its pivots is comparable work to evaluating one coupling block), so
# the per-active-cluster dry run is already the right cost model — no per-level
# correction needed here, unlike TopDown.
_basesworkload(::BottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)

# Same cost model as BottomUp: one active cluster is about one
# coupling-matrix-equivalent of work, whether it runs as a per-node task
# (BottomUp) or as a share of a level-wide batch (BatchedBottomUp).
_basesworkload(::BatchedBottomUp, tree, fardata::FarData) = _farworkload(tree, fardata)

# BottomUp + DirectionalData: LF pairs (dir==0) keep weight 1 each; HF levels
# (dir!=0) follow the same per-level-constant model as TopDown — roughly
# nleaves × avg-HF-directions-per-cluster per active HF level. Mirrors the
# `fardata.dirs[diridx]==0 && next!(pbar)` + post-loop HF batch logic in bottomup.jl.
function _basesworkload(::BottomUp, tree, fardata::DirectionalData)
    nleaves = count(n -> isleaf(tree, n), 1:numberofnodes(tree))
    hfsteps = Dict(
        level => max(1, nleaves ÷ length(LevelIterator(tree, level))) for
        level in levels(tree)
    )
    return _farworkload(tree, fardata, (level, dir) -> dir == 0 ? 1 : hfsteps[level])
end

# Bases (ACA/IACA pivoting) and coupling (one kernel-block evaluation per far pair) have
# unrelated per-unit costs, so they get separate bars rather than sharing one linear scale
# where neither end is weighted correctly. Both advance at per-cluster / per-coupling-block
# granularity (not per-level, not in one final batch), and because the dry runs above count
# only what will actually do work, levels with no interactions contribute nothing.
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
