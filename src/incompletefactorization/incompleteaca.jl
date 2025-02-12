struct iACA{RowPivType,ColPivType,ConvCritType}
    rowpivoting::RowPivType
    columnpivoting::ColPivType
    convergence::ConvCritType

    function iACA(rowpivoting, columnpivoting, convergence)
        return new{typeof(rowpivoting),typeof(columnpivoting),typeof(convergence)}(
            rowpivoting, columnpivoting, convergence
        )
    end
end

function iACA(
    pos::Vector{SVector{D,F}};
    rowpivoting=LRF.MaximumValue(),
    columnpivoting=IACAPivoting(pos),
    convergence=IncompleteNormEstimator(F[], F(0.0)),
) where {D,F<:Real}
    return iACA(rowpivoting, columnpivoting, convergence)
end

function init(
    iaca::iACA{RPT,CPT,CCT},
    M::LRF.LazyMatrix{Int,K};
    ref=sum(iaca.rowpivoting.pos[M.σ]) / length(M.σ),
) where {K,RPT<:GeoPivStrat,CPT<:LRF.PivStrat,CCT<:LRF.ConvCrit}
    return iACA(
        iaca.rowpivoting(M.τ; ref=ref), iaca.columnpivoting(M.σ), iaca.convergence(M)
    )
end

function init(
    iaca::iACA{RPT,CPT,CCT},
    M::LRF.LazyMatrix{Int,K};
    ref=sum(iaca.columnpivoting.pos[M.τ]) / length(M.τ),
) where {K,RPT<:LRF.PivStrat,CPT<:GeoPivStrat,CCT<:LRF.ConvCrit}
    return iACA(
        iaca.rowpivoting(M.τ), iaca.columnpivoting(M.σ; ref=ref), iaca.convergence(M)
    )
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    M::LRF.LazyMatrix{Int,K},
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int,
    tol::F,
) where {
    F<:Real,K,RowPivType<:GeoPivStrat,ColPivType<:LRF.PivStrat,ConvCritType<:LRF.ConvCrit
}
    maxcolumn = size(M, 2)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1

    rows[npivot] = iaca.rowpivoting()
    @views M.μ(
        rowbuffer[npivot:npivot, 1:maxcolumn],
        M.τ[rows[npivot]:rows[npivot]],
        M.σ[1:maxcolumn],
    )
    iaca.convergence.normUV = norm(rowbuffer[npivot:npivot, 1:maxcolumn])
    colbuffer[1, 1] = K(1.0)
    cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])

    conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        rows[npivot] = iaca.rowpivoting(npivot)
        @views M.μ(
            rowbuffer[npivot:npivot, 1:maxcolumn],
            M.τ[rows[npivot]:rows[npivot]],
            M.σ[1:maxcolumn],
        )

        # Norm update
        updatenorm!(iaca.convergence, rowbuffer[npivot, 1:maxcolumn], npivot)

        colbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views colbuffer[npivot, k] =
                rowbuffer[k, cols[k]]^-1 * rowbuffer[npivot, cols[k]]
            for kk in 1:maxcolumn
                @views rowbuffer[npivot, kk] -= rowbuffer[k, kk] * colbuffer[npivot, k]
            end
        end

        cols[npivot] = iaca.columnpivoting(rowbuffer[npivot, 1:maxcolumn])
        conv = iaca.convergence(rowbuffer[npivot, 1:maxcolumn], npivot, tol)
    end

    return rows[1:npivot], cols[1:npivot]
end

function (iaca::iACA{RowPivType,ColPivType,ConvCritType})(
    M::LRF.LazyMatrix{Int,K},
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    maxrank::Int,
    tol::F,
) where {
    F<:Real,K,RowPivType<:LRF.PivStrat,ColPivType<:GeoPivStrat,ConvCritType<:LRF.ConvCrit
}
    maxrow = size(M, 1)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)
    npivot = 1

    cols[npivot] = iaca.columnpivoting()
    @views M.μ(
        colbuffer[1:maxrow, npivot:npivot], M.τ[1:maxrow], M.σ[cols[npivot]:cols[npivot]]
    )
    iaca.convergence.normUV = norm(colbuffer[1:maxrow, npivot:npivot])
    rowbuffer[1, 1] = K(1.0)
    rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])

    conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot, tol)

    while conv && npivot < maxrank
        npivot += 1

        cols[npivot] = iaca.columnpivoting(npivot)
        @views M.μ(
            colbuffer[1:maxrow, npivot:npivot],
            M.τ[1:maxrow],
            M.σ[cols[npivot]:cols[npivot]],
        )

        # Norm update
        updatenorm!(iaca.convergence, colbuffer[1:maxrow, npivot], npivot)

        rowbuffer[npivot, npivot] = K(1.0)
        for k in 1:(npivot - 1)
            @views rowbuffer[k, npivot] =
                colbuffer[rows[k], k] .^ -1 * colbuffer[rows[k], npivot]
            for kk in 1:maxrow
                @views colbuffer[kk, npivot] -= colbuffer[kk, k] * rowbuffer[k, npivot]
            end
        end

        rows[npivot] = iaca.rowpivoting(colbuffer[1:maxrow, npivot])
        conv = iaca.convergence(colbuffer[1:maxrow, npivot], npivot, tol)
    end

    return rows[1:npivot], cols[1:npivot]
end
