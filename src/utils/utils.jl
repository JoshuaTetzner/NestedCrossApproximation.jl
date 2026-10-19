# CSR offsets from per-entry counts: entry i owns ptr[i]:ptr[i+1]-1.
function ptrfromcounts(counts::AbstractVector{<:Integer})
    ptr = Vector{Int}(undef, length(counts) + 1)
    ptr[1] = 1
    @inbounds for i in eachindex(counts)
        ptr[i + 1] = ptr[i] + Int(counts[i])
    end
    return ptr
end

function collect_assigned(v::Vector{B}) where {B}
    nassigned = 0
    @inbounds for i in eachindex(v)
        nassigned += isassigned(v, i)
    end

    compact = Vector{B}(undef, nassigned)
    nodes = Vector{Int}(undef, nassigned)

    k = 0
    @inbounds for i in eachindex(v)
        if isassigned(v, i)
            k += 1
            compact[k] = v[i]
            nodes[k] = i
        end
    end

    return compact, nodes
end

"""
    SingularCrossMatrixError(kind, npivots, rank)

Raised when the cross matrix `A[pivots, cols]` selected for a cluster is singular, so the
nested basis or transfer matrix for that cluster cannot be formed.
"""
struct SingularCrossMatrixError <: Exception
    kind::Symbol
    npivots::Int
    rank::Int
end

function Base.showerror(io::IO, e::SingularCrossMatrixError)
    return print(
        io,
        "SingularCrossMatrixError: the cross matrix for a $(e.kind) is singular - ",
        "$(e.npivots) pivots were selected but only $(e.rank) of them are linearly ",
        "independent.\nThe pivot search never selects the same index twice, so a ",
        "dependent pivot set points at the factorized matrix rather than at the ",
        "pivoting: the usual cause is duplicate rows/columns, i.e. coincident points ",
        "(a degenerate or zero-radius cluster).",
    )
end

# LU of a cross matrix; throws SingularCrossMatrixError if the pivots are dependent.
function _crossfactorization(X::AbstractMatrix{T}, kind::Symbol) where {T}
    F = lu!(X; check=false)
    k = size(X, 1)
    k == 0 && return F

    U = F.factors
    R = real(T)
    d2max = zero(R)
    d2min = typemax(R)
    @inbounds for i in 1:k
        d2 = abs2(U[i, i])
        d2 > d2max && (d2max = d2)
        d2 < d2min && (d2min = d2)
    end

    tol2 = (10 * eps(R))^2 * d2max
    (issuccess(F) && d2min > tol2) && return F

    rank = 0
    @inbounds for i in 1:k
        abs2(U[i, i]) > tol2 && (rank += 1)
    end
    return throw(SingularCrossMatrixError(kind, k, rank))
end

"""
    _scheduler(ntasks)

Scheduler of the parallel loops on `ntasks` tasks: a `DynamicScheduler` with `ntasks`
chunks, or serial execution on the calling task for `ntasks = 1`. The public functions take
only `ntasks`.
"""
function _scheduler(ntasks::Integer)
    ntasks >= 1 || throw(ArgumentError("ntasks must be positive, got $ntasks"))
    return ntasks == 1 ? SerialScheduler() : DynamicScheduler(; ntasks)
end
