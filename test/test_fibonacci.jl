using NestedCrossApproximation
using LinearAlgebra
using StaticArrays
using DelimitedFiles

E_t = NestedCrossApproximation.sphericalfibonaccipoints(50)
E_t_prime = NestedCrossApproximation.sphericalfibonaccipoints(20)

writedlm("E_t.dat", reduce(hcat, E_t)')
writedlm("E_t_prime.dat", reduce(hcat, E_t_prime)')

function compute_direction_map(E_t, E_t_prime)
    n = length(E_t)
    map = Matrix{Int}(undef, n, 2)

    for i in 1:n
        e = E_t[i]

        best_j = 1
        best_dot = dot(e, E_t_prime[1])

        for j in 2:length(E_t_prime)
            d = dot(e, E_t_prime[j])
            if d > best_dot
                best_dot = d
                best_j = j
            end
        end

        # zero-based indices for TikZ/pgfplotstable loops
        map[i, 1] = i - 1
        map[i, 2] = best_j - 1
    end

    return map
end

map = compute_direction_map(E_t, E_t_prime)
writedlm("map.dat", map)
##

using NestedCrossApproximation
using LinearAlgebra
using StaticArrays
using DelimitedFiles

E_t = NestedCrossApproximation.sphericalfibonaccipoints(50)
E_t_prime = NestedCrossApproximation.sphericalfibonaccipoints(20)

# -------------------------------------------------
# view angles (same as in TikZ)
az = deg2rad(35.0)
el = deg2rad(20.0)

# orthographic projection + visibility
function project_point(e::SVector{3,Float64}, az, el)
    x, y, z = e
    px = -sin(az) * x + cos(az) * y
    py = -sin(el) * cos(az) * x - sin(el) * sin(az) * y + cos(el) * z
    pzview = cos(el) * cos(az) * x + cos(el) * sin(az) * y + sin(el) * z
    return px, py, pzview
end

# visible projected points: rows = i px py
function visible_projection_table(E, az, el)
    rows = Float64[]
    for (i, e) in enumerate(E)
        px, py, pz = project_point(e, az, el)
        if pz > 0
            append!(rows, (i - 1, px, py))
        end
    end
    return reshape(rows, 3, :)'
end

# nearest-parent map: rows = i j  (zero-based)
function compute_direction_map(E_t, E_t_prime)
    n = length(E_t)
    map = Matrix{Int}(undef, n, 2)

    for i in 1:n
        e = E_t[i]
        best_j = 1
        best_dot = dot(e, E_t_prime[1])

        for j in 2:length(E_t_prime)
            d = dot(e, E_t_prime[j])
            if d > best_dot
                best_dot = d
                best_j = j
            end
        end

        map[i, 1] = i - 1
        map[i, 2] = best_j - 1
    end
    return map
end

# build lookup for visible points
function visible_lookup(E, az, el)
    d = Dict{Int,Tuple{Float64,Float64}}()
    for (i, e) in enumerate(E)
        px, py, pz = project_point(e, az, el)
        if pz > 0
            d[i - 1] = (px, py)
        end
    end
    return d
end

# visible mapping segments: rows = x1 y1 x2 y2
function visible_mapping_segments(E_t, E_t_prime, map, az, el)
    A = visible_lookup(E_t, az, el)
    B = visible_lookup(E_t_prime, az, el)

    rows = Float64[]
    for k in 1:size(map, 1)
        i = map[k, 1]
        j = map[k, 2]
        if haskey(A, i) && haskey(B, j)
            x1, y1 = A[i]
            x2, y2 = B[j]
            append!(rows, (x1, y1, x2, y2))
        end
    end
    return isempty(rows) ? zeros(0, 4) : reshape(rows, 4, :)'
end

# -------------------------------------------------
# compute + export
map = compute_direction_map(E_t, E_t_prime)

Et_vis = visible_projection_table(E_t, az, el)               # i px py
Etp_vis = visible_projection_table(E_t_prime, az, el)        # i px py
map_vis = visible_mapping_segments(E_t, E_t_prime, map, az, el)  # x1 y1 x2 y2

writedlm("E_t_vis.dat", Et_vis)
writedlm("E_t_prime_vis.dat", Etp_vis)
writedlm("map_vis.dat", map_vis)
