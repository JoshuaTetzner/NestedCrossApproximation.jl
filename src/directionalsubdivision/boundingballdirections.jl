using LinearAlgebra: dot
using StaticArrays: SVector
import H2Trees: BoundingBallTree, BlockTree

function directions(::Val{2}, diam::F, k::F) where {F<:Real}
    kd = max(k * diam, eps(F))
    n = max(4, Int(ceil(pi / asin(min(one(F), one(F) / kd)))))
    return [SVector{2,F}(cos(2 * pi * j / n), sin(2 * pi * j / n)) for j in 0:(n - 1)]
end

function directions(::Val{3}, diam::F, k::F) where {F<:Real}
    kd = max(k * diam, eps(F))
    n = ceil(
        Int, 6 * 4^(log(2, acos(one(F) / sqrt(F(3))) / asin(min(one(F), one(F) / kd))))
    )
    n = max(6, n)
    pts = sphericalfibonaccipoints(n)
    return [SVector{3,F}(p) for p in pts]
end

function sphericalfibonaccipoints(n::Int)
    ga = pi * (3 - sqrt(5))
    pts = Vector{SVector{3,Float64}}(undef, n)
    @inbounds for i in 0:(n - 1)
        z = 1 - 2 * (i + 0.5) / n
        r = sqrt(max(0.0, 1 - z * z))
        pts[i + 1] = SVector(r * cos(i * ga), r * sin(i * ga), z)
    end
    return pts
end

# Find the index in `evecs` (unit direction vectors) whose direction best matches `vec`.
# Uses dot product instead of angle (equivalent for unit vectors, avoids acos).
@inline function _nearest_direction(evecs::AbstractVector, vec)
    best = 1
    bestscore = dot(evecs[1], vec)
    @inbounds for i in 2:length(evecs)
        score = dot(evecs[i], vec)
        if score > bestscore
            bestscore = score
            best = i
        end
    end
    return best
end

function directionalfardata(
    fardata::FarData, cltree::BoundingBallTree, reftree::BoundingBallTree, isnear
)
    farptr = NestedCrossApproximation.farptr(fardata)
    fars = NestedCrossApproximation.fars(fardata)
    nnodes = length(farptr) - 1
    N = size(eltype(cltree), 1)   # spatial dimension
    F = eltype(eltype(cltree))    # float type
    k = isnear.k
    # dirs[node]  = indices into that node's totalEvec that are actually used (unique)
    # farE[node]  = for each far, local direction index (1..ndirs) into dirs[node]
    # pdirmap[node] = for each parent direction (1..ndirs_parent), child's local dir index
    # Evec[node]  = actual direction vectors for the used directions (temp, for child mapping)
    dirs = Vector{Vector{Int}}(undef, nnodes)
    farE = Vector{Vector{Int}}(undef, nnodes)
    pdirmap = [Int[] for _ in 1:nnodes]
    Evec = Vector{Vector{SVector{N,F}}}(undef, nnodes)

    # --- Top-down pass ---
    for level in H2Trees.levels(cltree)
        for node in H2Trees.LevelIterator(cltree, level)
            pnode = H2Trees.parent(cltree, node)
            hasfars = farptr[node + 1] > farptr[node]
            hasparentvecs = pnode != 0 && isassigned(Evec, pnode)

            # LF nodes: store sentinel [0] when relevant, skip direction assignment.
            if isnear.islf(cltree, node)
                inheritlf =
                    hasparentvecs &&
                    pnode != 0 &&
                    isassigned(dirs, pnode) &&
                    !isempty(dirs[pnode]) &&
                    isnear.islf(cltree, pnode)
                dirs[node] = (hasfars || inheritlf) ? Int[0] : Int[]
                farE[node] = Int[]
                continue
            end

            if !hasfars && !hasparentvecs
                dirs[node] = Int[]
                farE[node] = Int[]
                continue
            end

            totalEvec = directions(Val(N), F(2) * H2Trees.radius(cltree, node), F(k))

            # Assign each far to the closest direction (index into totalEvec).
            nfars_node = Int(farptr[node + 1] - farptr[node])
            rawE = Vector{Int}(undef, nfars_node)
            for (i, faridx) in enumerate(farptr[node]:(farptr[node + 1] - 1))
                vec = interaction(node, fars[faridx], cltree, reftree)
                rawE[i] = _nearest_direction(totalEvec, vec)
            end

            # Map each parent direction vector to the closest direction in totalEvec.
            rawEmap = Int[]
            if hasparentvecs
                pvecs = Evec[pnode]
                sizehint!(rawEmap, length(pvecs))
                for pvec in pvecs
                    push!(rawEmap, _nearest_direction(totalEvec, pvec))
                end
            end

            uniqueE = union(rawE, rawEmap)
            Evec[node] = totalEvec[uniqueE]

            # Build local-index lookup: totalEvec index → position in uniqueE (1-based).
            localslot = Dict{Int,Int}(e => i for (i, e) in enumerate(uniqueE))

            farE[node] = [localslot[e] for e in rawE]
            pdirmap[node] = [localslot[e] for e in rawEmap]
            dirs[node] = uniqueE
        end
    end

    # --- Bottom-up pass: sort fars by direction, update pdirmap to sorted indices ---
    dircounter = [Int[] for _ in 1:nnodes]
    for level in reverse(H2Trees.levels(cltree))
        for node in H2Trees.LevelIterator(cltree, level)
            isnear.islf(cltree, node) && continue
            isempty(dirs[node]) && continue

            nfars = Int(farptr[node + 1] - farptr[node])
            ndirs = length(dirs[node])

            # Count fars per (old) local direction.
            localdircounter = zeros(Int, ndirs)
            for d in farE[node]
                localdircounter[d] += 1
            end

            # Sort dirs[node] by totalEvec index, giving canonical order.
            permunique = sortperm(dirs[node])
            permute!(dirs[node], permunique)
            permute!(localdircounter, permunique)

            # Remap old local indices to new (sorted) local indices.
            invpermunique = invperm(permunique)
            for i in eachindex(farE[node])
                farE[node][i] = invpermunique[farE[node][i]]
            end
            for j in eachindex(pdirmap[node])
                pdirmap[node][j] = invpermunique[pdirmap[node][j]]
            end

            # Sort fars so that all fars of the same direction are contiguous.
            if nfars > 1
                perm = sortperm(farE[node])
                farview = @view fars[farptr[node]:(farptr[node + 1] - 1)]
                permute!(farview, perm)
            end

            dircounter[node] = localdircounter
        end
    end

    # --- Convert pdirmap (parent local → child local) to parentdir (child local → parent local) ---
    parentdir = [zeros(Int, length(dirs[node])) for node in eachindex(dirs)]
    for node in eachindex(dirs)
        for (pidx, cidx) in enumerate(pdirmap[node])
            iszero(cidx) && continue
            cidx <= length(parentdir[node]) && (parentdir[node][cidx] = pidx)
        end
    end

    # Fill dircounter for LF sentinel nodes (dirs=[0], dircounter not set in second pass).
    for node in 1:nnodes
        if !isempty(dirs[node]) && isempty(dircounter[node])
            dircounter[node] = [max(0, Int(farptr[node + 1] - farptr[node]))]
        end
    end

    # --- Build linear direction storage ---
    dirptr = Vector{Int}(undef, nnodes + 1)
    dirptr[1] = 1
    for node in 1:nnodes
        dirptr[node + 1] = dirptr[node] + length(dirs[node])
    end
    linedirs = Vector{Int}(undef, dirptr[end] - 1)
    linedircounter = Vector{Int}(undef, dirptr[end] - 1)
    lineparentdir = Vector{Int}(undef, dirptr[end] - 1)
    for node in 1:nnodes
        first = dirptr[node]
        last = dirptr[node + 1] - 1
        first > last && continue
        linedirs[first:last] = dirs[node]
        linedircounter[first:last] = dircounter[node]
        lineparentdir[first:last] = parentdir[node]
    end

    linefardata = fardata

    return DirectionalData(
        linefardata, dirptr, linedirs, linedircounter, lineparentdir, isnear
    )
end
