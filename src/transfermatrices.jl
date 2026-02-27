# directional topdowncompressor
function testtransfermatrix(
    t::Int,
    dir::Int,
    pivs::Tuple{Vector{Int},Vector{Int}},
    buffer::Matrix{K},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData,
) where {I,K}
    tmats = Tuple{Int,Matrix{K}}[]
    for child in ChildIterator(tree, t)
        crows = pivots[child][paternaldirection(dirdata, child, dir)][1]
        rows = pivs[1]

        @assert length(crows) == length(unique(crows))
        @assert length(rows) == length(unique(rows))

        @views tmat = buffer[crows, 1:length(rows)] / buffer[rows, 1:length(rows)]
        push!(tmats, (paternaldirection(dirdata, child, dir), tmat))
    end
    return tmats
end

function testtransfermatrices!(
    tmats::Vector{Dict{Int,Vector{Tuple{Int,Matrix{K}}}}},
    level::Int,
    blocks::Vector{D},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {I,K,D<:Dict{Int,Matrix{K}}}
    @tasks for t in collect(H2Trees.LevelIterator(tree, level))
        @set ntasks = ntasks
        if !isdirectionalroot(dirdata, tree, t) && isassigned(pivots, t)
            ntmats = Vector{Tuple{Int,Matrix{K}}}[]
            for (dir, piv) in pivots[t]
                dntmats = Tuple{Int,Matrix{K}}[]
                #cdirs = Int[]
                for child in ChildIterator(tree, t)
                    crows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                        idx in pivots[child][paternaldirection(dirdata, child, dir)][1]
                    ]
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for idx in piv[1]
                    ]
                    tmat = blocks[t][dir][crows, :] / blocks[t][dir][rows, :]

                    push!(dntmats, (paternaldirection(dirdata, child, dir), tmat))
                end
                push!(ntmats, dntmats)
            end
            tmats[t] = Dict(keys(pivots[t]) .=> ntmats)
        end
    end
end

# directional topdowncompressor
function trialtransfermatrix(
    s::Int,
    dir::Int,
    pivs,
    buffer::Matrix{K},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData,
) where {I,K}
    tmats = Tuple{Int,Matrix{K}}[]
    for child in ChildIterator(tree, s)
        ccols = pivots[child][paternaldirection(dirdata, child, dir)][2]
        cols = pivs[2]
        @views tmat = buffer[1:length(cols), cols] \ buffer[1:length(cols), ccols]

        push!(tmats, (paternaldirection(dirdata, child, dir), tmat))
    end
    return tmats
end

function trialtransfermatrices!(
    tmats::Vector{Dict{Int,Vector{Tuple{Int,Matrix{K}}}}},
    level::Int,
    blocks::Vector{D},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {I,K,D<:Dict{Int,Matrix{K}}}
    @tasks for s in collect(H2Trees.LevelIterator(tree, level))
        @set ntasks = ntasks
        if !isdirectionalroot(dirdata, tree, s) && isassigned(pivots, s)
            ntmats = Vector{Tuple{Int,Matrix{K}}}[]
            for (dir, piv) in pivots[s]
                dntmats = Tuple{Int,Matrix{K}}[]
                for child in ChildIterator(tree, s)
                    ccols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                        idx in pivots[child][paternaldirection(dirdata, child, dir)][2]
                    ]
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for idx in piv[2]
                    ]
                    tmat = blocks[s][dir][:, cols] \ blocks[s][dir][:, ccols]

                    push!(dntmats, (paternaldirection(dirdata, child, dir), tmat))
                end
                push!(ntmats, dntmats)
            end
            tmats[s] = Dict(keys(pivots[s]) .=> ntmats)
        end
    end
end
