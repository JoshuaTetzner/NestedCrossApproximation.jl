bufferidx(level::Int) = iseven(level) ? 1 : 2

function testbuffer(::TopDown, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return (zeros(T, size(farmatrix, 1), maxrank), zeros(T, size(farmatrix, 1), maxrank))
end

function testbuffer(::BottomUp, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, size(farmatrix, 1), maxrank)
end
function trialbuffer(::TopDown, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return (zeros(T, maxrank, size(farmatrix, 2)), zeros(T, maxrank, size(farmatrix, 2)))
end
function trialbuffer(::BottomUp, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, maxrank, size(farmatrix, 2))
end
function fartestbuffer(::ACA, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, maxrank, size(farmatrix, 2))
end
function fartestbuffer(::IACA, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, maxrank, maxrank)
end
function fartrialbuffer(::ACA, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, size(farmatrix, 1), maxrank)
end
function fartrialbuffer(::IACA, farmatrix::AbstractKernelMatrix{T}, maxrank::Int) where {T}
    return zeros(T, maxrank, maxrank)
end
