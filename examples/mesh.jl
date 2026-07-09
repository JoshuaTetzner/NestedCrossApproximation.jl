
using CompScienceMeshes
using BEAST
using H2Trees
using Printf

outdir = "cluster_visualization"
mkpath(outdir)

m = meshicosphere(10, 1.0)
patch = lagrangecxd0(m)

kmtree = KMeansTree(patch.pos, 2; minvalues=10)
octree = TwoNTree(patch.pos, 0.25)

function cluster_ids(tree, level, n)
    ids = zeros(Int, n)
    for (cid, node) in enumerate(H2Trees.LevelIterator(tree, level))
        ids[H2Trees.values(tree, node)] .= cid
    end
    ids
end

function write_pgfplots_file(file, m, ids)
    verts = vertices(m)
    open(file, "w") do io
        println(io, "x y z cluster")
        for i in 1:numcells(m)
            for v in CompScienceMeshes.indices(m, i)
                p = verts[v]
                @printf(io, "%.16e %.16e %.16e %d\n",
                    Float64(p[1]), Float64(p[2]), Float64(p[3]), ids[i])
            end
            println(io)
        end
    end
end
function cluster_color_ids(m, cluster_ids, ncolors)
    edge2face = Dict{Tuple{Int,Int},Int}()
    adj = Dict{Int,Set{Int}}()

    for c in unique(cluster_ids)
        adj[c] = Set{Int}()
    end

    for f in 1:numcells(m)
        vs = collect(CompScienceMeshes.indices(m, f))
        edges = (
            (min(vs[1],vs[2]), max(vs[1],vs[2])),
            (min(vs[2],vs[3]), max(vs[2],vs[3])),
            (min(vs[3],vs[1]), max(vs[3],vs[1]))
        )

        for e in edges
            if haskey(edge2face, e)
                g = edge2face[e]
                c1 = cluster_ids[f]
                c2 = cluster_ids[g]
                if c1 != c2
                    push!(adj[c1], c2)
                    push!(adj[c2], c1)
                end
            else
                edge2face[e] = f
            end
        end
    end

    colors = Dict{Int,Int}()
    order = sort(collect(keys(adj)), by=c->-length(adj[c]))

    for c in order
        used = Set(get(colors, nb, 0) for nb in adj[c] if haskey(colors, nb))
        for col in 1:ncolors
            if !(col in used)
                colors[c] = col
                break
            end
        end
    end

    [colors[c] for c in cluster_ids]
end

octree_level = 3
kmeans_level = 7

kmeans_ids = cluster_ids(kmtree, kmeans_level, length(patch.pos))
octree_ids = cluster_ids(octree, octree_level, length(patch.pos))

kmeans_color_ids = cluster_color_ids(m, kmeans_ids, 8)
octree_color_ids = cluster_color_ids(m, octree_ids, 8)

write_pgfplots_file(joinpath(outdir, "sphere_kmeans.dat"), m, kmeans_color_ids)
write_pgfplots_file(joinpath(outdir, "sphere_octree.dat"), m, octree_color_ids)
function box_edges(io, xmin, xmax, ymin, ymax, zmin, zmax)
    p = [
        (xmin,ymin,zmin), (xmax,ymin,zmin), (xmax,ymax,zmin), (xmin,ymax,zmin),
        (xmin,ymin,zmax), (xmax,ymin,zmax), (xmax,ymax,zmax), (xmin,ymax,zmax)
    ]

    e = [
        (1,2), (2,3), (3,4), (4,1),
        (5,6), (6,7), (7,8), (8,5),
        (1,5), (2,6), (3,7), (4,8)
    ]

    for (i,j) in e
        println(io, "$(p[i][1]) $(p[i][2]) $(p[i][3])")
        println(io, "$(p[j][1]) $(p[j][2]) $(p[j][3])")
        println(io, "nan nan nan")
    end
end

function write_manual_octree_boxes(file, points; level=2)
    n = 2^level
    h = 2 / n
    occupied = Set{NTuple{3,Int}}()

    for p in points
        ix = clamp(floor(Int, (p[1] + 1) / h) + 1, 1, n)
        iy = clamp(floor(Int, (p[2] + 1) / h) + 1, 1, n)
        iz = clamp(floor(Int, (p[3] + 1) / h) + 1, 1, n)
        push!(occupied, (ix, iy, iz))
    end

    open(file, "w") do io
        println(io, "x y z")
        for (ix, iy, iz) in occupied
            xmin, xmax = -1 + (ix-1)*h, -1 + ix*h
            ymin, ymax = -1 + (iy-1)*h, -1 + iy*h
            zmin, zmax = -1 + (iz-1)*h, -1 + iz*h
            box_edges(io, xmin, xmax, ymin, ymax, zmin, zmax)
        end
    end
end

function write_outer_bbox(file)
    open(file, "w") do io
        println(io, "x y z")
        box_edges(io, -1, 1, -1, 1, -1, 1)
    end
end

write_manual_octree_boxes("sphere_octree_boxes.dat", patch.pos; level=2)
write_outer_bbox("sphere_octree_bbox.dat")
