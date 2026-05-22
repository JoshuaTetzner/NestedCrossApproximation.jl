# orientation_utils.jl
# Generic NCA-side utilities. No BEAST, no CompScienceMeshes.

const ACA = AdaptiveCrossApproximation

function orientation_ids(q)
    key_to_id = Dict{eltype(q),Int32}()
    ids = Vector{Int32}(undef, length(q))
    nextid = Int32(0)

    @inbounds for i in eachindex(q)
        id = get(key_to_id, q[i], Int32(0))
        if id == 0
            nextid += 1
            id = nextid
            key_to_id[q[i]] = id
        end
        ids[i] = id
    end

    return ids, Int(nextid)
end

@inline function _canonical_direction_key(v; digits=1)
    scale = 10.0^digits
    x = round(Int16, abs(v[1]) * scale)
    y = round(Int16, abs(v[2]) * scale)
    z = round(Int16, abs(v[3]) * scale)
    return (x, y, z)
end

@inline function _keyvector(key)
    v = (Float64(key[1]), Float64(key[2]), Float64(key[3]))
    nv = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    iszero(nv) && return (0.0, 0.0, 0.0)
    return (v[1] / nv, v[2] / nv, v[3] / nv)
end

@inline function _absdot(a, b)
    return abs(a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
end

function phase_orientation_ids(
    keys; max_phases::Int=3, min_probability::Float64=0.1, orth_tol::Float64=0.25
)
    isempty(keys) && return Int32[], 0
    ids, nids = orientation_ids(keys)
    counts = zeros(Int, nids)

    @inbounds for id in ids
        counts[id] += 1
    end

    representatives = Vector{eltype(keys)}(undef, nids)
    @inbounds for i in eachindex(keys)
        id = ids[i]
        counts[id] > 0 && (representatives[id] = keys[i])
    end

    order = sortperm(counts; rev=true)
    selected = Int[]
    selectedvecs = Tuple{Float64,Float64,Float64}[]
    total = length(keys)

    for id in order
        counts[id] / total < min_probability && break
        v = _keyvector(representatives[id])
        iszero(v[1]) && iszero(v[2]) && iszero(v[3]) && continue
        all(_absdot(v, u) <= orth_tol for u in selectedvecs) || continue
        push!(selected, id)
        push!(selectedvecs, v)
        length(selected) == max_phases && break
    end

    phaseof = zeros(Int32, nids)
    @inbounds for (phaseid, id) in enumerate(selected)
        phaseof[id] = Int32(phaseid)
    end

    phaseids = Vector{Int32}(undef, length(ids))
    @inbounds for i in eachindex(ids)
        phaseids[i] = phaseof[ids[i]]
    end

    return phaseids, length(selected)
end

function basisfunction_orientation_ids(
    edges,
    normals;
    edge_digits=1,
    normal_digits=2,
    max_normal_phases=3,
    min_normal_probability=0.1,
    normal_orth_tol=0.25,
)
    edge_keys = [
        _canonical_direction_key(edge; digits=edge_digits) for edge in edges
    ]
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]

    edgeids, nedgeids = orientation_ids(edge_keys)
    normalids, nnormalids = phase_orientation_ids(
        normal_keys;
        max_phases=max_normal_phases,
        min_probability=min_normal_probability,
        orth_tol=normal_orth_tol,
    )

    return edgeids, normalids, nedgeids, nnormalids
end

function node_orientation_sets(
    oriid::AbstractVector{Int32}, tree, nori::Integer; max_orientations::Int=6
)
    out = Vector{Vector{Int32}}(undef, length(tree.nodes))

    seen = falses(nori)
    touched = Int32[]

    for node in eachindex(tree.nodes)
        empty!(touched)
        idcs = H2Trees.values(tree, node)

        @inbounds for i in idcs
            id = oriid[i]
            iszero(id) && continue
            if !seen[id]
                seen[id] = true
                push!(touched, id)
            end
        end

        if length(touched) > max_orientations
            fill!(out, Int32[])
            return out
        end

        out[node] = copy(touched)

        @inbounds for id in touched
            seen[id] = false
        end
    end

    return out
end

function _has_dominant_node_orientation(oriid, tree; min_probability::Float64=0.8)
    counts = Dict{Int32,Int}()

    for node in eachindex(tree.nodes)
        empty!(counts)
        idcs = H2Trees.values(tree, node)

        @inbounds for i in idcs
            id = oriid[i]
            iszero(id) && continue
            counts[id] = get(counts, id, 0) + 1
        end

        for count in values(counts)
            count / length(idcs) >= min_probability && return true
        end
    end

    return false
end

function node_normal_orientation_sets(
    normals,
    tree;
    normal_digits=2,
    max_orientations=6,
    max_normal_phases=3,
    min_normal_probability=0.1,
    normal_orth_tol=0.25,
    min_node_probability=0.8,
)
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]
    normalids, nnormalids = phase_orientation_ids(
        normal_keys;
        max_phases=max_normal_phases,
        min_probability=min_normal_probability,
        orth_tol=normal_orth_tol,
    )

    if nnormalids > 0 &&
       !_has_dominant_node_orientation(normalids, tree; min_probability=min_node_probability)
        fill!(normalids, Int32(0))
        return [Int32[] for _ in eachindex(tree.nodes)], normalids, 0
    end

    nodesets = node_orientation_sets(
        normalids, tree, nnormalids; max_orientations=max_orientations
    )

    return nodesets, normalids, nnormalids
end

function nodeorientationids(oriid, tree, nori)
    out = Vector{ACA.NodeOriID}(undef, length(tree.nodes))

    counts = zeros(Int32, nori)
    touched = Int32[]

    for node in eachindex(tree.nodes)
        empty!(touched)
        idcs = H2Trees.values(tree, node)

        @inbounds for i in idcs
            id = oriid[i]
            if counts[id] == 0
                push!(touched, id)
            end
            counts[id] += 1
        end

        best_id = touched[1]
        best_c = counts[best_id]

        @inbounds for id in touched
            c = counts[id]
            if c > best_c
                best_id = id
                best_c = c
            end
        end

        out[node] = ACA.NodeOriID(best_id, best_c / length(idcs))

        @inbounds for id in touched
            counts[id] = 0
        end
    end

    return out
end
