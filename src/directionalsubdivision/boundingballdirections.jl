using LinearAlgebra: dot
using StaticArrays: SVector
import H2Trees: BoundingBallTree, BlockTree

function directions(::Val{2}, diam::F, k::F, γ::F) where {F<:Real}
    kd = max(k * diam, eps(F))
    n = max(4, Int(ceil(pi / asin(min(one(F), γ / kd)))))
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
# Minimizing angle(a,b) over fixed `vec` and unit-norm `evecs` is equivalent to maximizing
# dot(a,b), which skips the norm/acos work.
@inline function _nearest_direction(evecs::AbstractVector, vec)
    best = 1
    bestdot = dot(evecs[1], vec)
    @inbounds for i in 2:length(evecs)
        newdot = dot(evecs[i], vec)
        if newdot > bestdot
            bestdot = newdot
            best = i
        end
    end
    return best
end

function directionalfardata(
    fardata::FarData, tree::BoundingBallTree, fartree::BoundingBallTree, isnear
)
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    nnodes = length(ptr) - 1
    N = size(eltype(tree), 1)   # spatial dimension
    F = eltype(eltype(tree))    # float type
    k = isnear.k
    γ = isnear.islf.γ
    # dirs[node]  = indices into that node's totalEvec that are actually used (unique)
    # farE[node]  = for each far, local direction index (1..ndirs) into dirs[node]
    # partochddirmap[node] = for each parent direction (1..ndirs_parent), child's local dir index
    # Evec[node]  = actual direction vectors for the used directions (temp, for child mapping)
    dirs = Vector{Vector{Int}}(undef, nnodes)
    dircount = [Int[] for _ in 1:nnodes]
    partochddirmap = [Int[] for _ in 1:nnodes]
    partochddircount = zeros(Int, nnodes)

    dirsvec = Vector{Vector{SVector{N,F}}}(undef, nnodes)

    # --- Top-down pass ---
    for level in H2Trees.levels(tree)
        for node in H2Trees.LevelIterator(tree, level)
            pnode = H2Trees.parent(tree, node)
            hasfars = ptr[node + 1] > ptr[node]
            hasparentvecs = pnode != 0 && isassigned(dirsvec, pnode)
            hasparentdirs = pnode != 0 && !isempty(dirs[pnode])
            if !isnear.islf(tree, node)
                @assert hasparentvecs == hasparentdirs "Parent dirs without parent vecs should not happen"
            end
            # LF nodes do not use directional subdivision.
            # Store [0] for direct LF far nodes, or when inheriting from an LF parent with dirs.
            if isnear.islf(tree, node)
                inheritlf =
                    hasparentdirs &&
                    level > 1 &&
                    isnear.islf(tree, H2Trees.parent(tree, node))
                dirs[node] = (hasfars || inheritlf) ? Int[0] : Int[]
                partochddirmap[node] = inheritlf ? [1] : Int[]
                continue
            end

            (!hasfars && !hasparentvecs) && (dirs[node]=Int[]; continue)

            totaldirsvec = directions(Val(N), F(2) * H2Trees.radius(tree, node), F(k), F(γ))
            # Assign each far to the closest direction (index into totalEvec).
            nfar = Int(ptr[node + 1] - ptr[node])
            nodedirs = Vector{Int}(undef, nfar)
            for (i, faridx) in enumerate(ptr[node]:(ptr[node + 1] - 1))
                vec = interaction(node, nodes[faridx], tree, fartree)
                nodedirs[i] = _nearest_direction(totaldirsvec, vec)
            end

            perm = sortperm(nodedirs)
            permute!(view(nodes, ptr[node]:(ptr[node + 1] - 1)), perm)
            uniquenodedirs = sort!(unique(nodedirs))
            noriginal = length(uniquenodedirs)

            # Map each parent direction to the closest direction in totalEvec,
            # using child-centered far vectors when the parent direction has far nodes.
            partochddir = iszero(pnode) ? Int[] : zeros(Int, length(dirs[pnode]))
            if !iszero(pnode)
                ctopc = H2Trees.center(tree, node) - H2Trees.center(tree, pnode)
                pdirmultiplicator =
                    isnear.k * (2 * H2Trees.radius(tree, pnode)) / isnear.ηhf +
                    2 * H2Trees.radius(tree, pnode)
                for didx in eachindex(dirs[pnode])
                    partochddir[didx] = _nearest_direction(
                        totaldirsvec, ctopc + pdirmultiplicator .* dirsvec[pnode][didx]
                    )
                end

                for (idx, cdir) in enumerate(partochddir)
                    if cdir in uniquenodedirs
                        partochddir[idx] = findfirst(==(cdir), uniquenodedirs)
                    else
                        push!(uniquenodedirs, cdir)
                        partochddir[idx] = length(uniquenodedirs)
                    end
                end
            end
            # nodedirs[perm] is sorted to match uniquenodedirs[1:noriginal], so a
            # run-length count over that shared order gives the per-direction counts.
            # Directions appended beyond noriginal are inherited from the parent and
            # have no own far nodes, so they stay at 0.
            localdircount = zeros(Int, length(uniquenodedirs))
            sortednodedirs = view(nodedirs, perm)
            n = length(sortednodedirs)
            i = 1
            for uidx in 1:noriginal
                c = 0
                while i <= n && sortednodedirs[i] == uniquenodedirs[uidx]
                    c += 1
                    i += 1
                end
                localdircount[uidx] = c
            end
            dirsvec[node] = totaldirsvec[uniquenodedirs]

            partochddirmap[node] = partochddir
            partochddircount[node] = count(!iszero, partochddir)
            dirs[node] = uniquenodedirs
            dircount[node] = localdircount
        end
    end

    # Fill dircount for LF sentinel nodes (dirs=[0], dircount not set in second pass).
    for node in eachindex(dirs)
        if !isempty(dirs[node]) && isempty(dircount[node])
            dircount[node] = [max(0, Int(ptr[node + 1] - ptr[node]))]
            partochddircount[node] = length(partochddirmap[node])
        end
    end

    return _finalize_directionaldata(
        fardata, dirs, dircount, partochddirmap, partochddircount, isnear
    )
end
