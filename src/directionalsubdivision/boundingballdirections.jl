using LinearAlgebra: dot
using StaticArrays: SVector
import H2Trees: BoundingBallTree, BlockTree

function directions(::Val{2}, diam::F, k::F, γ::F) where {F<:Real}

    #missing gamma functionality here
    kd = max(k * diam, eps(F))
    n = max(4, Int(ceil(pi / asin(min(one(F), one(F) / kd)))))
    return [SVector{2,F}(cos(2 * pi * j / n), sin(2 * pi * j / n)) for j in 0:(n - 1)]
end

function directions(::Val{3}, diam::F, k::F, γ::F) where {F<:Real}
    kd = max(k * diam, eps(F))
    n = ceil(Int, 6 * 4^(log(2, acos(one(F) / sqrt(F(3))) / asin(min(one(F), γ / kd)))))
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
@inline function _nearest_direction(evecs::AbstractVector, vec)
    best = 1
    minangle = angle(evecs[1], vec)
    @inbounds for i in 2:length(evecs)
        newangle = angle(evecs[i], vec)
        if newangle < minangle
            minangle = copy(newangle)
            best = i
        end
    end
    return best
end

function directionalfardata(
    fardata::FarData, tree::BoundingBallTree, fartree::BoundingBallTree, isnear
)
    farptr = NestedCrossApproximation.farptr(fardata)
    fars = NestedCrossApproximation.fars(fardata)
    nnodes = length(farptr) - 1
    N = size(eltype(tree), 1)   # spatial dimension
    F = eltype(eltype(tree))    # float type
    k = isnear.k
    γ = isnear.islf.γ
    # dirs[node]  = indices into that node's totalEvec that are actually used (unique)
    # farE[node]  = for each far, local direction index (1..ndirs) into dirs[node]
    # pdirmap[node] = for each parent direction (1..ndirs_parent), child's local dir index
    # Evec[node]  = actual direction vectors for the used directions (temp, for child mapping)
    dirs = Vector{Vector{Int}}(undef, nnodes)
    dircounter = [Int[] for _ in 1:nnodes]
    pdirmap = [Int[] for _ in 1:nnodes]
    pdircounter = zeros(Int, nnodes)

    dirsvec = Vector{Vector{SVector{N,F}}}(undef, nnodes)

    # --- Top-down pass ---
    for level in H2Trees.levels(tree)
        for node in H2Trees.LevelIterator(tree, level)
            pnode = H2Trees.parent(tree, node)
            hasfars = farptr[node + 1] > farptr[node]
            hasparentvecs = pnode != 0 && isassigned(dirsvec, pnode)
            hasparentdirs = pnode != 0 && !isempty(dirs[pnode])
            if !isnear.islf(tree, node)
                @assert hasparentvecs == hasparentdirs "Parent dirs without parent vecs should not happen"
            end
            # LF nodes do not use directional subdivision.
            # Store [0] for direct LF fars, or when inheriting from an LF parent with dirs.
            if isnear.islf(tree, node)
                inheritlf =
                    hasparentdirs &&
                    level > 1 &&
                    isnear.islf(tree, H2Trees.parent(tree, node))
                dirs[node] = (hasfars || inheritlf) ? Int[0] : Int[]
                pdirmap[node] = inheritlf ? [1] : Int[]
                continue
            end

            (!hasfars && !hasparentvecs) && (dirs[node] = Int[]; continue)

            totaldirsvec = directions(
                Val(N), F(2) * H2Trees.radius(tree, node), F(k), F(γ)
            )
            # Assign each far to the closest direction (index into totalEvec).
            nfars_node = Int(farptr[node + 1] - farptr[node])
            nodedirs = Vector{Int}(undef, nfars_node)
            for (i, faridx) in enumerate(farptr[node]:(farptr[node + 1] - 1))
                vec = interaction(node, fars[faridx], tree, fartree)
                nodedirs[i] = _nearest_direction(totaldirsvec, vec)
            end

            perm = sortperm(nodedirs)
            permute!(view(fars, farptr[node]:(farptr[node + 1] - 1)), perm)
            uniquenodedirs = sort!(unique(nodedirs))

            # Map each parent direction to the closest direction in totalEvec,
            # using child-centered far vectors when the parent direction has fars.
            dirmap = iszero(pnode) ? Int[] : zeros(Int, length(dirs[pnode]))
            ctopc = H2Trees.center(tree, node) - H2Trees.center(tree, pnode)
            pdirmultiplicator =
                isnear.k * (2 * H2Trees.radius(tree, pnode)) / isnear.ηhf +
                2 * H2Trees.radius(tree, pnode)
            if !iszero(pnode)
                for didx in eachindex(dirs[pnode])
                    # This if clause might be completely wrong!!!
                    #if !isempty(dircounter[pnode]) && dircounter[pnode][didx] > 0
                    #    pstart = Int(farptr[pnode])
                    #    @inbounds for i in 1:(didx - 1)
                    #        pstart += Int(dircounter[pnode][i])
                    #    end
                    #    count = Int(dircounter[pnode][didx])
                    #    acc = zero(dirsvec[pnode][didx])
                    #    @inbounds for faridx in pstart:(pstart + count - 1)
                    #        acc += interaction(node, fars[faridx], tree, fartree)
                    #    end
                    #    dirmap[didx] = _nearest_direction(totaldirsvec, acc)
                    #else

                    dirmap[didx] = _nearest_direction(
                        totaldirsvec, ctopc + pdirmultiplicator .* dirsvec[pnode][didx]
                    )
                    #end
                end

                for (idx, dir) in enumerate(dirmap)
                    if dir in uniquenodedirs
                        dirmap[idx] = findfirst(==(dir), uniquenodedirs)
                    else
                        push!(uniquenodedirs, dir)
                        dirmap[idx] = length(uniquenodedirs)
                    end
                end
            end
            localdircounter = zeros(Int, length(uniquenodedirs))
            for (uidx, udir) in enumerate(uniquenodedirs)
                for dir in nodedirs
                    udir == dir && (localdircounter[uidx] += 1)
                end
            end
            dirsvec[node] = totaldirsvec[uniquenodedirs]

            pdirmap[node] = dirmap
            pdircounter[node] = count(!iszero, dirmap)
            dirs[node] = uniquenodedirs
            dircounter[node] = localdircounter
        end
    end

    # Fill dircounter for LF sentinel nodes (dirs=[0], dircounter not set in second pass).
    for node in eachindex(dirs)
        if !isempty(dirs[node]) && isempty(dircounter[node])
            dircounter[node] = [max(0, Int(farptr[node + 1] - farptr[node]))]
            pdircounter[node] = length(pdirmap[node])
        end
    end

    # Build linear direction storage.
    dirptr = Vector{Int}(undef, length(dirs) + 1)
    pdirptr = Vector{Int}(undef, length(pdirmap) + 1)
    dirptr[1] = 1
    pdirptr[1] = 1
    for node in eachindex(dirs)
        dirptr[node + 1] = dirptr[node] + length(dirs[node])
        pdirptr[node + 1] = pdirptr[node] + pdircounter[node]
    end
    linedirs = Vector{Int}(undef, dirptr[end] - 1)
    linedircounter = Vector{Int}(undef, dirptr[end] - 1)
    lineparentdir = Vector{Int}(undef, pdirptr[end] - 1)
    for node in eachindex(dirs)
        first = dirptr[node]
        last = dirptr[node + 1] - 1
        first > last && continue
        linedirs[first:last] = dirs[node]
        linedircounter[first:last] = dircounter[node]

        pfirst = pdirptr[node]
        plast = pdirptr[node + 1] - 1
        pfirst > plast && continue
        lineparentdir[pfirst:plast] = pdirmap[node]
    end

    linefardata = fardata

    return DirectionalData(
        linefardata, dirptr, linedirs, linedircounter, pdirptr, lineparentdir, isnear
    )
end
