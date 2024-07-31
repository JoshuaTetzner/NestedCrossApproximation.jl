function assemble_couplingmatrices(
    matrixassembler, 
    ::Type{K},
    fars::Vector{Vector{Tuple{I, I}}}, 
    test_fars::Vector{PivotBlocks{I, K}},
    trial_fars::Vector{PivotBlocks{I, K}},
    compressor::FastBEAST.ACAOptions{B, I, F}; 
    multithreading=true, 
    verbose=true
) where {B, I, F, K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{I, K}}(undef, length(fars))
    if verbose
        p = Progress(length(fars), desc="Assemble column bases: ")
    end
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        i = findfirst(x -> x == far[2], test_fars[far[1]].interactionmap[2])
        range = test_fars[far[1]].interactionmap[1][i]+1:test_fars[far[1]].interactionmap[1][i+1]
        
        blk = test_fars[far[1]].M.M.V[:, range][:, trial_fars[far[2]].M.M.σ]

        lowrankblocks[idx] = H2MatrixBlock(
            MatrixBlock(
                blk,
                test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
                trial_fars[far[2]].M.σ[trial_fars[far[2]].M.M.σ],
            ),
            test_fars[far[1]].M.τ,
            trial_fars[far[2]].M.σ,
            far[1],
            far[2],
        )

        verbose && next!(p)
    end

    return lowrankblocks
end

function assemble_couplingmatrices(
    matrixassembler, 
    ::Type{K},
    fars::Vector{Vector{Tuple{I, I}}}, 
    test_fars::Vector{PivotBlocks{I, K}}, 
    compressor::FastBEAST.ACAOptions{B, I, F}; 
    multithreading=true, 
    verbose=true
) where {B, I, F, K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{I, K}}(undef, length(fars))
    if verbose
        p = Progress(length(fars), desc="Assemble column bases: ")
    end
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        i = findfirst(x -> x == far[2], test_fars[far[1]].interactionmap[2])
        range = test_fars[far[1]].interactionmap[1][i]+1:test_fars[far[1]].interactionmap[1][i+1]
        
        blk = test_fars[far[1]].M.M.V[:, range][:, test_fars[far[2]].M.M.τ]

        lowrankblocks[idx] = H2MatrixBlock(
            MatrixBlock(
                blk,
                test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
                test_fars[far[2]].M.τ[test_fars[far[2]].M.M.τ],
            ),
            test_fars[far[1]].M.τ,
            test_fars[far[2]].M.τ,
            far[1],
            far[2],
        )

        verbose && next!(p)
    end

    return lowrankblocks
end


function assemble_couplingmatrices(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{I,I}}},
    test_fars::Vector{PivotBlocks{I, K}},
    trial_fars::Vector{PivotBlocks{I, K}},
    compressor::PCAOptions{B, I, F};
    multithreading=true, 
    verbose=true
) where {B, I, F, K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{I, K}}(undef, length(fars))
    
    if verbose
        p = Progress(length(fars), desc="Assemble column bases: ")
    end
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        blk = zeros(
            K,
            length(test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ]),
            length(trial_fars[far[2]].M.σ[trial_fars[far[2]].M.M.σ]),
        )
        matrixassembler(
            blk,
            test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
            trial_fars[far[2]].M.σ[trial_fars[far[2]].M.M.σ],
        )
        lowrankblocks[idx] = H2MatrixBlock(
            MatrixBlock(
                blk,
                test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
                trial_fars[far[2]].M.σ[trial_fars[far[2]].M.M.σ],
            ),
            test_fars[far[1]].M.τ,
            trial_fars[far[2]].M.σ,
            far[1],
            far[2],
        )

        verbose && next!(p)
    end

    return lowrankblocks
end


function assemble_couplingmatrices(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{I,I}}},
    test_fars::Vector{PivotBlocks{I, K}},
    compressor::PCAOptions{B, I, F};
    multithreading=true, 
    verbose=true
) where {B, I, F, K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{I, K}}(undef, length(fars))
    
    if verbose
        p = Progress(length(fars), desc="Assemble column bases: ")
    end
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        blk = zeros(
            K,
            length(test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ]),
            length(test_fars[far[2]].M.τ[test_fars[far[2]].M.M.τ]),
        )
        matrixassembler(
            blk,
            test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
            test_fars[far[2]].M.τ[test_fars[far[2]].M.M.τ],
        )
        lowrankblocks[idx] = H2MatrixBlock(
            MatrixBlock(
                blk,
                test_fars[far[1]].M.τ[test_fars[far[1]].M.M.τ],
                test_fars[far[2]].M.τ[test_fars[far[2]].M.M.τ],
            ),
            test_fars[far[1]].M.τ,
            test_fars[far[2]].M.τ,
            far[1],
            far[2],
        )

        verbose && next!(p)
    end

    return lowrankblocks
end