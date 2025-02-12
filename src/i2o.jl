function assemble_couplingmatrices(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{I,I}}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}};
    multithreading=true,
) where {I,K}
    fars = reduce(vcat, fars)
    lk = Threads.SpinLock()
    lowrankblocks = H2MatrixBlock{I,K}[]

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(fars) do far
        if far[1] > far[2]
            blk = zeros(K, length(pivots[far[1]][1]), length(pivots[far[2]][1]))
            matrixassembler(blk, pivots[far[1]][1], pivots[far[2]][1])
            lock(lk) do
                push!(lowrankblocks, H2MatrixBlock(blk, far[1], far[2]))
            end
        end
    end

    return lowrankblocks
end

function assemble_couplingmatrices(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{I,I}}},
    testpivots::Vector{Tuple{Vector{I},Vector{I}}},
    trialpivots::Vector{Tuple{Vector{I},Vector{I}}};
    multithreading=true,
) where {I,K}
    fars = reduce(vcat, fars)
    lk = Threads.SpinLock()
    lowrankblocks = H2MatrixBlock{I,K}[]

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(fars) do far
        blk = zeros(K, length(testpivots[far[1]][1]), length(trialpivots[far[2]][2]))
        matrixassembler(blk, testpivots[far[1]][1], trialpivots[far[2]][2])
        lock(lk) do
            push!(lowrankblocks, H2MatrixBlock(blk, far[1], far[2]))
        end
    end

    return lowrankblocks
end
