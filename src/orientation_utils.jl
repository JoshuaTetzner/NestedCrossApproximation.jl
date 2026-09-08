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

@inline function _iszero_direction_key(key)
    return iszero(key[1]) && iszero(key[2]) && iszero(key[3])
end

function normal_orientation_ids(q)
    key_to_id = Dict{eltype(q),Int32}()
    ids = zeros(Int32, length(q))
    nextid = Int32(0)

    @inbounds for i in eachindex(q)
        _iszero_direction_key(q[i]) && continue
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

function basisfunction_orientation_ids(edges, normals; edge_digits=1, normal_digits=1)
    edge_keys = [_canonical_direction_key(edge; digits=edge_digits) for edge in edges]
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]

    edgeids, nedgeids = orientation_ids(edge_keys)
    normalids, nnormalids = normal_orientation_ids(normal_keys)

    return edgeids, normalids, nedgeids, nnormalids
end

function node_orientation_sets(
    oriid::AbstractVector{Int32}, tree, nori::Integer; max_orientations::Int=3
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
            out[node] = Int32[]
        else
            out[node] = copy(touched)
        end

        @inbounds for id in touched
            seen[id] = false
        end
    end

    return out
end

function _key_representatives(keys, ids, nids::Integer)
    representatives = Vector{eltype(keys)}(undef, nids)
    @inbounds for i in eachindex(keys)
        id = ids[i]
        iszero(id) && continue
        representatives[id] = keys[i]
    end
    return representatives
end

function _orthogonal_direction_set(
    ids::AbstractVector{Int32}, representatives; orth_tol::Float64=0.25
)
    @inbounds for i in 1:length(ids)
        vi = _keyvector(representatives[ids[i]])
        for j in (i + 1):length(ids)
            vj = _keyvector(representatives[ids[j]])
            _absdot(vi, vj) <= orth_tol || return false
        end
    end
    return true
end

function node_normal_orientation_sets(
    normalids::AbstractVector{Int32},
    tree,
    nnormalids::Integer,
    representatives;
    max_orientations::Int=3,
    primary_probability::Float64=0.2,
    secondary_probability::Float64=0.1,
    active_probability::Float64=0.7,
    orth_tol::Float64=0.25,
)
    out = Vector{Vector{Int32}}(undef, length(tree.nodes))

    counts = zeros(Int, nnormalids)
    touched = Int32[]
    selected = Int32[]
    inherited = Int32[]

    function visit!(node::Int, parentclasses::AbstractVector{Int32})
        empty!(touched)
        empty!(selected)
        empty!(inherited)
        idcs = H2Trees.values(tree, node)
        ntotal = length(idcs)

        @inbounds for i in idcs
            id = normalids[i]
            iszero(id) && continue
            if iszero(counts[id])
                push!(touched, id)
            end
            counts[id] += 1
        end

        sort!(touched; by=id -> counts[id], rev=true)
        selectedcount = 0
        if !isempty(touched) && counts[first(touched)] / ntotal >= primary_probability
            firstid = first(touched)
            push!(selected, firstid)
            selectedcount += counts[firstid]

            @inbounds for k in 2:length(touched)
                id = touched[k]
                counts[id] / ntotal < secondary_probability && break
                v = _keyvector(representatives[id])
                orthogonal = true
                for sid in selected
                    u = _keyvector(representatives[sid])
                    if _absdot(v, u) > orth_tol
                        orthogonal = false
                        break
                    end
                end
                if orthogonal
                    push!(selected, id)
                    selectedcount += counts[id]
                    length(selected) == max_orientations && break
                end
            end
        end

        if length(selected) >= 2 && selectedcount / ntotal >= active_probability
            out[node] = copy(selected)
        elseif !isempty(parentclasses)
            @inbounds for id in parentclasses
                id <= nnormalids && counts[id] > 0 && push!(inherited, id)
            end
            out[node] = copy(inherited)
        else
            out[node] = Int32[]
        end

        @inbounds for id in touched
            counts[id] = 0
        end

        for child in H2Trees.children(tree, node)
            visit!(Int(child), out[node])
        end
    end

    for node in eachindex(tree.nodes)
        H2Trees.parent(tree, node) == 0 && visit!(Int(node), Int32[])
    end

    return out
end

function node_normal_orientation_sets(
    normals,
    tree;
    normal_digits=1,
    max_orientations=3,
    primary_probability=0.2,
    secondary_probability=0.1,
    active_probability=0.7,
    orth_tol=0.25,
)
    normal_keys = [
        _canonical_direction_key(normal; digits=normal_digits) for normal in normals
    ]
    normalids, nnormalids = normal_orientation_ids(normal_keys)
    representatives = _key_representatives(normal_keys, normalids, nnormalids)

    nodesets = node_normal_orientation_sets(
        normalids,
        tree,
        nnormalids,
        representatives;
        max_orientations=max_orientations,
        primary_probability=primary_probability,
        secondary_probability=secondary_probability,
        active_probability=active_probability,
        orth_tol=orth_tol,
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
