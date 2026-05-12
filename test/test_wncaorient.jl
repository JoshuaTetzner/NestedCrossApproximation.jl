using LinearAlgebra
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using BEAST

@inline unit(v) = v / norm(v)
@inline cz(x) = iszero(x) ? zero(x) : x
@inline clean(v) = typeof(v)(cz(v[1]), cz(v[2]), cz(v[3]))

@inline function canon(v)
    v = clean(v)
    return if v[1] < 0 || (v[1] == 0 && v[2] < 0) || (v[1] == 0 && v[2] == 0 && v[3] < 0)
        clean(-v)
    else
        v
    end
end

@inline function edgeverts(cell, refid)
    refid == 1 && return cell[2], cell[3]
    refid == 2 && return cell[3], cell[1]
    return cell[1], cell[2]
end

function rwg_orientations(rwg)
    mesh  = rwg.geo
    verts = vertices(mesh)
    nc    = numcells(mesh)

    cells = [CompScienceMeshes.indices(mesh, c) for c in 1:nc]

    ℓ = similar(rwg.pos)
    n = similar(rwg.pos)
    cn = similar(rwg.pos, nc)

    @inbounds for c in 1:nc
        cell = cells[c]
        p1, p2, p3 = verts[cell[1]], verts[cell[2]], verts[cell[3]]
        cn[c] = clean(unit(cross(p2 - p1, p3 - p1)))
    end

    @inbounds for i in eachindex(rwg.fns)
        fn = rwg.fns[i]
        sh = fn[1]

        cell = cells[sh.cellid]
        a, b = edgeverts(cell, sh.refid)

        ℓ[i] = canon(unit(verts[b] - verts[a]))

        ni = cn[fn[1].cellid]
        if length(fn) > 1
            nj = cn[fn[2].cellid]
            ni += dot(ni, nj) < 0 ? -nj : nj
        end

        n[i] = clean(unit(ni))
    end

    return ℓ, n
end

@inline function dirkey(v; digits=1)
    s = 10.0^digits

    x = round(Int16, v[1] * s)
    y = round(Int16, v[2] * s)
    z = round(Int16, v[3] * s)

    if x < 0 || (x == 0 && y < 0) || (x == 0 && y == 0 && z < 0)
        return (-x, -y, -z)
    else
        return (x, y, z)
    end
end

@inline orientation_key(ℓ, n; dell=1, dn=1) = (dirkey(ℓ; digits=dell), dirkey(n; digits=dn))

function orientation_keys(ℓ, n; dell=1, dn=1)
    q = Vector{typeof(orientation_key(ℓ[1], n[1]; dell = dell, dn = dn))}(undef, length(ℓ))

    @inbounds for i in eachindex(ℓ)
        q[i] = orientation_key(ℓ[i], n[i]; dell=dell, dn=dn)
    end

    return q
end

struct NodeOri{K}
    key::K
    purity::Float64
end

function nodeorientationkeys(rwgkeys, tree)
    K = eltype(rwgkeys)
    out = Vector{NodeOri{K}}(undef, length(tree.nodes))
    counts = Dict{K,Int}()

    for node in eachindex(tree.nodes)
        empty!(counts)

        idcs = H2Trees.values(tree, node)

        @inbounds for i in idcs
            q = rwgkeys[i]
            counts[q] = get(counts, q, 0) + 1
        end

        best_q = first(keys(counts))
        best_c = 0

        for (q, c) in counts
            if c > best_c
                best_q = q
                best_c = c
            end
        end

        out[node] = NodeOri(best_q, best_c / length(idcs))
    end

    return out
end

##
Γ = meshcuboid(1.0, 1.0, 1.0, 0.01)
RT = raviartthomas(Γ)
tree = H2Trees.KMeansTree(
    RT.pos, 2; minvalues=100, updateradii=H2Trees.unsafemaxradiusboundingsphere
)
length(RT)
##
@time ℓ, n = rwg_orientations(RT)
@time q = orientation_keys(ℓ, n; dell=1, dn=1)
@time nodeori = nodeorientationkeys(q, tree);
