mutable struct PCAGlobalMemory{K}
    U::Matrix{K}
    V::Matrix{K}
    used_I::Vector{Bool}
    I::Vector{Int}
    used_J::Vector{Bool}
    J::Vector{Int}
    npivots::Int
end

function clear!(am::PCAGlobalMemory{K}) where {K}
    am.U .= K(0.0)
    am.V .= K(0.0)
    am.used_I .= false
    am.I .= 0
    am.used_J .= false
    am.J .= 0
    am.npivots = 1
end

maxrank(acamemory::PCAGlobalMemory{K}) where K = size(acamemory.M, 2)

function allocate_pca_memory_rm(
    ::Type{K}, maxrows::I, maxcolumns::I; maxrank=40
) where {I, K}
    U = zeros(K, maxrows, maxrank)
    V = zeros(K, maxrank, maxrank)
    Is = zeros(I, maxrank)
    Js = zeros(I, maxrank)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)

    return PCAGlobalMemory(U, V, used_I, Is, used_J, Js, 1)
end

function allocate_pca_memory_rm(
    ::Type{K}, maxrows::I, maxcolumns::I, multithreading::B; maxrank=40
) where {I, B, K}
    if multithreading
        return [allocate_pca_memory_rm(
            K, maxrows, maxcolumns, maxrank=maxrank
        ) for i = 1:Threads.nthreads()]   
    else
        return [allocate_pca_memory_rm(K, maxrows, maxcolumns, maxrank=maxrank)]
    end
end

function allocate_pca_memory_cm(
    ::Type{K}, maxrows::I, maxcolumns::I; maxrank=40
) where {I, K}
    U = zeros(K, maxrank, maxrank)
    V = zeros(K, maxrank, maxcolumns)
    Is = zeros(I, maxrank)
    Js = zeros(I, maxrank)
    used_I = zeros(Bool, maxrows)
    used_J = zeros(Bool, maxcolumns)

    return PCAGlobalMemory(U, V, used_I, Is, used_J, Js, 1)
end

function allocate_pca_memory_cm(
    ::Type{K}, maxrows::I, maxcolumns::I, multithreading::B; maxrank=40
) where {I, B, K}
    if multithreading
        return [allocate_pca_memory_cm(
            K, maxrows, maxcolumns, maxrank=maxrank
        ) for i = 1:Threads.nthreads()]   
    else
        return [allocate_pca_memory_cm(K, maxrows, maxcolumns, maxrank=maxrank)]
    end
end