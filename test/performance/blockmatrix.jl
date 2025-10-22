using BEAST
using BlockSparseMatrices
using NestedCrossApproximation
using CompScienceMeshes

##
λ = 1
k = 2pi / λ

Γ = meshsphere(1.0, 0.1)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)

fm = NestedCrossApproximation.BEASTKernelMatrix(op, space, space);
size(fm)
##
function buildsingle(fmatrix)
    lk = Threads.SpinLock()
    rows = Vector{Int}[]
    cols = Vector{Int}[]
    blks = Matrix{ComplexF64}[]
    for i in 1:100:4000
        for j in 0:3
            blk = zeros(ComplexF64, 100, 100)
            fmatrix(blk, i:(i + 100 - 1), (i + j * 100):(i + 100 - 1 + j * 100))

            lock(lk) do
                push!(blks, blk)
                push!(rows, Vector(i:(i + 100 - 1)))
                push!(cols, Vector((i + j * 100):(i + 100 - 1 + j * 100)))
            end
        end
    end
    return BlockSparseMatrix(blks, rows, cols, size(fmatrix); ntasks=16)
end

function buildsingle2(fmatrix)
    lk = Threads.SpinLock()
    rows = Vector{Int}[]
    cols = Vector{Int}[]
    blks = Matrix{ComplexF64}[]
    for i in 1:100:4000
        blk = zeros(ComplexF64, 100, 400)
        fmatrix(blk, i:(i + 100 - 1), (i):(i + 100 - 1 + 3 * 100))

        lock(lk) do
            push!(blks, blk)
            push!(rows, Vector(i:(i + 100 - 1)))
            push!(cols, Vector((i):(i + 100 - 1 + 3 * 100)))
        end
    end
    return BlockSparseMatrix(blks, rows, cols, size(fmatrix); ntasks=16)
end
##
@time bm1 = buildsingle(fm);
@time bm2 = buildsingle2(fm);
##
x = rand(ComplexF64, size(bm1, 2))
##
using Test
using LinearAlgebra
@time transpose(bm1 * x);
@time transpose(bm2 * x);
@test norm(transpose(bm1) * x - transpose(bm2) * x) == 0