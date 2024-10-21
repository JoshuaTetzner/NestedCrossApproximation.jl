struct PCAMemory{K}
    M::Matrix{K}
    N::Vector{K}
    #used_I::Vector{Bool}
    I::Vector{Int}
    #used_J::Vector{Bool}
    J::Vector{Int}
    #npivots::Int
end

function pivoting(
    pivstrat::FastBEAST.FD,
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
) where K

    return argmax(roworcolumn)
end


function pca_rm2(
    M::LazyMatrix{I, K},
    am::Matrix{K},
    pivstrat::FastBEAST.FD;
    maxrank=min(size(am, 2), length(M.σ)),#Int(round(length(M.τ)*length(M.σ)/(length(M.τ)+length(M.σ)))),
    tol=1e-4
) where {I, K}
    npivots = 1
    maxrows = size(M, 1)

    columnpivstrat, nextcolumn = FastBEAST.firstpivot(pivstrat, M.σ)
    am.J[npivots] = nextcolumn

    @views M.μ(
        am.M[1:maxrows, am.npivots:am.npivots], 
        M.τ[1:maxrows],
        M.σ[nextcolumn:nextcolumn]
    )
    
    #am.V[1, 1] = 1.0

    @views nextrow = pivoting(
        abs.(am.M[1:maxrows, am.npivots])
    )
    am.I[am.npivots] = nextrow
    
    norm(am.M[1:maxrows, am.npivots]) == 0.0 && return Matrix[], Int[], Int[]

    @views normU = norm(am.M[1:maxrows, 1])
    normV = 1.0
    convergence = true
    while convergence && am.npivots < maxrank
        am.npivots += 1
        
        @views nextcolumn = pivoting(columnpivstrat)
        am.J[am.npivots] = nextcolumn

        @views M.μ(
            am.M[1:maxrows, am.npivots:am.npivots], 
            M.τ[1:maxrows],
            M.σ[nextcolumn:nextcolumn]
        )
        
        normV += ((1/am.M[am.I[1], 1]) * am.M[am.I[1], am.npivots])^2
        for k = 1:am.npivots-1
           #@views  am.V[k, am.npivots] = (1/am.M[am.I[k], k]) * am.M[am.I[k], am.npivots]
            for kk = 1:maxrows
                @views am.M[kk, am.npivots] -= am.M[kk, k] * 
                    (1/am.M[am.I[k], k]) * am.M[am.I[k], am.npivots]#am.V[k, am.npivots]
            end
        end
        
        @views nextrow = pivoting(
            abs.(am.M[1:maxrows, am.npivots]),
        )
        am.I[am.npivots] = nextrow
        #normUV = norm(am.M[1:maxrows, am.npivots])
        
        #push!(oldnorms, normUV)
        #normU = norm(am.M[1:maxrows, 1])*norm(am.V[1, 1:am.npivots])
        @views convergence = norm(am.M[1:maxrows, am.npivots]) > tol * normU *sqrt(normV)
        #if !convergence
        #    convergence = convergence || checklinearconvergence(oldnorms, tol * normU)
        #end
    end

    retU = am.M[1:maxrows, 1:am.npivots]
    rpivots = am.I[1:am.npivots]
    cpivots = am.J[1:am.npivots]
    am.I[1:am.npivots] .= 0
    am.J[1:am.npivots] .= 0
    am.M[1:maxrows, 1:am.npivots] .= 0.0
    #am.V[1:am.npivots, 1:am.npivots] .= 0.0
    #am.Msed_I[rpivots] .= false
    #am.Msed_J[cpivots] .= false
    am.npivots = 1 

    return U, rpivots, cpivots
    
end