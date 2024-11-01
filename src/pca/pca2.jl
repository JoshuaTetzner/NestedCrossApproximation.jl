using Test

abstract type PivStrat end 

mutable struct PCA2Options{I, F}
    rpivstrat::NestedCrossApproximation.PivStrat
    cpivstrat::NestedCrossApproximation.PivStrat
    maxrank::I
    tol::F
end

struct MaximumValue <: PivStrat end

function (::MaximumValue)(row::Vector{K}) where K 
    return argmax(row)
end

struct MyPivoting{F, T} <: PivStrat 
    fct::Function
    w::Vector{F}
    h::Vector{F}
    pos::Vector{T}
end

function MyPivoting(fct::Function, pos::Vector{SVector{3, F}}; ref=SVector(0.0, 0.0, 0.0)) where F
    w = zeros(F, length(pos))
    h = zeros(F, length(pos))
    for (i, val) in enumerate(pos)
        w[i] = fct(norm(val - ref))
    end
    h .= 1/minimum(w)
    return MyPivoting(fct, w, h, pos)
end

function (strat::MyPivoting{F, T})(idcs::Vector{Int}, ref::SVector{3, F}) where {F, T}
    w = zeros(F, length(idcs))
    h = zeros(F, length(idcs))
    for (i, val) in enumerate(strat.pos[idcs])
        w[i] = 1.0/norm(val - ref)#strat.fct(norm(val - ref))::F
    end
    h .= 1/minimum(w)

    return MyPivoting(strat.fct, w, h, strat.pos[idcs])
end

function updatefd!(strat::MyPivoting{F, T}, pivot::Int) where {F, T}
    for k in eachindex(strat.h)
        if strat.h[k] > norm(strat.pos[k] - strat.pos[pivot])
            strat.h[k] = norm(strat.pos[k] - strat.pos[pivot])
        end
    end
end

function (pivstrat::MyPivoting)()
    pivot = argmax(pivstrat.h .* pivstrat.w)
    updatefd!(pivstrat, pivot)
    
    return pivot
end


function pca(
    M::LazyMatrix{I, K},
    rows::Union{SubArray, Array},
    #cols::SubArray,
    rowbuffer::Union{SubArray, Array},
    colbuffer::Union{SubArray, Array},
    #cpivots::Vector{Int},
    rpivstrat::PivStrat;
    tol=1e-4
) where {I, K}
    #@test norm(rowbuffer) == 0.0
    #@test norm(colbuffer) == 0.0
    colbuffer .= 0

    (maxrows, maxcolumns) = size(M)
    maxrank = size(colbuffer, 2)
    npivot=1
    usedrows = zeros(Bool, length(rows))
   
    @views M.μ(
        colbuffer[1:maxrows, npivot:npivot], 
        M.τ[1:maxrows],
        M.σ[npivot:npivot]
    )
    rowbuffer[1, 1] = 1.0

    rows[npivot] = rpivstrat(abs.(colbuffer[1:maxrows, npivot]))
    if usedrows[rows[npivot]]
        println("fail")
    end
    usedrows[rows[npivot]]=true



    norm(colbuffer[1:maxrows, 1]) == 0.0 && return Matrix[], Int[], Int[]

    @views normU = norm(colbuffer[1:maxrows, 1])
    #println("normU", normU)

    convergence = true
    while convergence && npivot < maxrank
        npivot += 1
        
        @views M.μ(
            colbuffer[1:maxrows, npivot:npivot], 
            M.τ[1:maxrows],
            M.σ[npivot:npivot]
        )
        
        rowbuffer[npivot, npivot] = 1.0
        for k = 1:npivot-1
            @views rowbuffer[k, npivot] = (1/colbuffer[rows[k], k]) * colbuffer[rows[k], npivot]
            for kk = 1:maxrows
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end
        
        rows[npivot] = rpivstrat(abs.(colbuffer[1:maxrows, npivot]))
        if usedrows[rows[npivot]]
            println("fail")
        end
        usedrows[rows[npivot]]=true
        normUV = norm(colbuffer[1:maxrows, npivot])
        convergence = normUV > tol * normU * norm(rowbuffer[1, 1:npivot])
        
     #   println(normUV," > ",tol * normU * norm(rowbuffer[1, 1:npivot]))
    end
    #println("notmV", rowbuffer[npivot, npivot])
    @views colbuffer[1:maxrows, 1:npivot]*rowbuffer[1:npivot, 1:npivot]
    rpivots = rows[1:npivot]
    rowbuffer[1:npivot, 1:npivot] .= 0.0

    return rpivots
end
