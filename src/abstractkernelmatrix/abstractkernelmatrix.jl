abstract type AbstractKernelMatrix{T} end

function AbstractKernelMatrix(operator, testspace, trialspace; args...) end

function (::AbstractKernelMatrix)(tdata, sdata, matrixblock) end

Base.eltype(::AbstractKernelMatrix{T}) where {T} = T
