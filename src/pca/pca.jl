using Statistics

mutable struct PCAOptions{B, I, F}
    rowpivstrat::FastBEAST.PivStrat
    columnpivstrat::FastBEAST.PivStrat
    convergcrit::FastBEAST.ConvergenceCriterion
    maxrank::I
    tol::F
    svdrecompress::B
end

function PCAOptions(
    rowpivstrat,
    columnpivstrat;
    convergcrit=FastBEAST.Standard(),
    maxrank=50,
    tol=1e-4,
    svdrecompress=false
)

    return PCAOptions(rowpivstrat, columnpivstrat, convergcrit, maxrank, tol, svdrecompress)
end


function pivoting(
    pivstrat::FastBEAST.FD,
    usedidcs::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int64}}, true}
)

    nextpivot = FastBEAST.filldistance(pivstrat, usedidcs)
    while usedidcs[nextpivot][1]
        nextpivot -= 1
    end

    FastBEAST.update_filldistance!(pivstrat, nextpivot)
    
    return nextpivot
end

function pivoting(
    roworcolumn::Vector{K},
    acausedindices::SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int64}}, true}
) where K
    if maximum(roworcolumn) != 0 
        return argmax(roworcolumn .* (.!acausedindices))
    else 
        return argmin(acausedindices)
    end
end

function checklinearconvergence(oldnorms::Vector{F}, refnorm::F) where F
    meany = mean(log10.(oldnorms))
    x = Vector(1:length(oldnorms))
    meanx = mean(x)

    β = sum((x .- meanx).*(log10.(oldnorms) .- meany)) / sum((x.-meanx).^2)
    α = meany - β*meanx

    return (α + β*(length(oldnorms))) > log10(refnorm)
end

function pca_rm(
    M::LazyMatrix{I, K},
    am::PCAGlobalMemory{K},
    columnpivstrat::FastBEAST.FD;
    maxrank=Int(round(length(M.τ)*length(M.σ)/(length(M.τ)+length(M.σ)))),
    tol=1e-4
) where {I, K}
    #clear!(am)  
    #oldnorms = Float64[]

    (maxrows, maxcolumns) = size(M)
    columnpivstrat, nextcolumn = FastBEAST.firstpivot(columnpivstrat, M.σ)
    am.used_J[nextcolumn] = true
    am.J[am.npivots] = nextcolumn

    @views M.μ(
        am.U[1:maxrows, am.npivots:am.npivots], 
        M.τ[1:maxrows],
        M.σ[nextcolumn:nextcolumn]
    )

    am.V[1, 1] = 1.0
    @views nextrow = pivoting(
        abs.(am.U[1:maxrows, am.npivots]),
        am.used_I[1:maxrows],
    )
    am.used_I[nextrow] = true
    am.I[am.npivots] = nextrow
    
    norm(am.U[1:maxrows, am.npivots]) == 0.0 && return Matrix[], Int[], Int[]

    @views normU = norm(am.U[1:maxrows, 1])
    convergence = true
    while convergence && am.npivots < maxrank
        am.npivots += 1
        
        @views nextcolumn = pivoting(columnpivstrat, am.used_J[1:maxcolumns])
        am.used_J[nextcolumn] = true
        if length(am.J) < am.npivots
            println(size(M), am.npivots)
        end
        am.J[am.npivots] = nextcolumn

        @views M.μ(
            am.U[1:maxrows, am.npivots:am.npivots], 
            M.τ[1:maxrows],
            M.σ[nextcolumn:nextcolumn]
        )
        
        am.V[am.npivots, am.npivots] = 1.0
        for k = 1:am.npivots-1
           @views  am.V[k, am.npivots] = (1/am.U[am.I[k], k]) * am.U[am.I[k], am.npivots]
            for kk = 1:maxrows
                @views am.U[kk, am.npivots] -= am.U[kk, k] * am.V[k, am.npivots]
            end
        end
        
        @views nextrow = pivoting(
            abs.(am.U[1:maxrows, am.npivots]),
            am.used_I[1:maxrows],
        )
        am.used_I[nextrow] = true
        am.I[am.npivots] = nextrow
        normUV = norm(am.U[1:maxrows, am.npivots])
        
        #push!(oldnorms, normUV)
        #normU = norm(am.U[1:maxrows, 1])*norm(am.V[1, 1:am.npivots])
        @views convergence = normUV > tol * normU * norm(am.V[1, 1:am.npivots])
        #if !convergence
        #    convergence = convergence || checklinearconvergence(oldnorms, tol * normU)
        #end
    end

    retU = am.U[1:maxrows, 1:am.npivots]
    retV = am.V[1:am.npivots, 1:am.npivots]
    rpivots = am.I[1:am.npivots]
    cpivots = am.J[1:am.npivots]
    #am.I[1:am.npivots] .= 0
    #am.J[1:am.npivots] .= 0
    am.U[1:maxrows, 1:am.npivots] .= 0.0
    am.V[1:am.npivots, 1:am.npivots] .= 0.0
    am.used_I[rpivots] .= false
    am.used_J[cpivots] .= false
    am.npivots = 1 
    return retU, retV, rpivots, cpivots
    
end

function pca_cm(
    M::LazyMatrix{I, K},
    am::PCAGlobalMemory{K},
    rowpivstrat::FastBEAST.FD;
    maxrank=Int(round(length(M.τ)*length(M.σ)/(length(M.τ)+length(M.σ)))),
    tol=1e-14
) where {I, K}
    
    #clear!(am)  

    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = FastBEAST.firstpivot(rowpivstrat, M.τ)
    am.used_I[nextrow] = true
    am.I[am.npivots] = nextrow

    @views M.μ(
        am.V[am.npivots:am.npivots, 1:maxcolumns], 
        M.τ[nextrow:nextrow],
        M.σ[1:maxcolumns]
    )
    
    am.U[1, 1] = 1

    @views nextcolumn = pivoting(
        abs.(am.V[am.npivots, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
    )
    am.used_J[nextcolumn] = true
    am.J[am.npivots] = nextcolumn
    
    norm(am.V[am.npivots, 1:maxcolumns]) == 0.0 && return Matrix[], Int[], Int[]

   @views normV = norm(am.V[1, 1:maxcolumns])
    convergence = true

    while convergence && am.npivots < maxrank
        am.npivots += 1
        
        @views nextrow = pivoting(rowpivstrat, am.used_I[1:maxrows])
        am.used_I[nextrow] = true
        am.I[am.npivots] = nextrow

        @views M.μ(
            am.V[am.npivots:am.npivots, 1:maxcolumns], 
            M.τ[nextrow:nextrow],
            M.σ[1:maxcolumns]
        )
        
        am.U[am.npivots, am.npivots] = 1
        for k = 1:am.npivots-1
           @views  am.U[am.npivots, k] = (1/am.V[k, am.J[k]]) * am.V[am.npivots, am.J[k]]
            for kk = 1:maxcolumns
                @views am.V[am.npivots, kk] -= am.V[k, kk] * am.U[am.npivots, k]
            end
        end
        
        @views nextcolumn = pivoting(
            abs.(am.V[am.npivots, 1:maxcolumns]),
            am.used_J[1:maxcolumns],
        )
        am.used_J[nextcolumn] = true
        am.J[am.npivots] = nextcolumn
        normUV = norm(am.V[am.npivots, 1:maxcolumns])
        @views convergence = normUV > am.npivots/maxcolumns * tol * normV
    end
    
    if am.npivots == maxrank
        println(size(M))
        println("Aborted after maxrank.")
    end

    return am.U[1:am.npivots, 1:am.npivots], am.V[1:am.npivots, 1:maxcolumns], am.I[1:am.npivots], am.J[1:am.npivots]
    
end