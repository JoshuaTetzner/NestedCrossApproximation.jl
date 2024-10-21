using FastBEAST
using BEAST
using NestedCrossApproximation
using StaticArrays
using CompScienceMeshes
using LinearAlgebra

function oneoverR(x, y)
    if x == y
        return 0
    else
        return 1/norm(x-y)
    end
end

function fullACA(
    M::Matrix{K}; maxrank=50, tol=1e-4
) where {K}
    rows = Int[]
    cols = Int[]
    refnorm = norm(M)

    inds = argmax(abs.(M))
    push!(rows, inds[1])
    push!(cols, inds[2])

    R = M[:, cols]*M[rows, cols]^-1*M[rows, :]-M

    while norm(R) > tol * refnorm
        inds = argmax(abs.(R))
        push!(rows, inds[1])
        push!(cols, inds[2])

        R = M - M[:, cols]*M[rows, cols]^-1*M[rows, :]
    end

    return rows, cols
end

Γ1 = meshrectangle(1.0, 1.0, 0.1)
Γ2 = translate(meshrectangle(4.0, 1.0, 0.1), SVector(2.0, 0.0, 0.0))

#=op = KernelFunction{Float64}(oneoverR)
cxd01 = FastBEAST.PointSpace{Float64}(Γ1.vertices);
cxd02 = FastBEAST.PointSpace{Float64}(Γ2.vertices);=#

op = Helmholtz3D.singlelayer()
cxd01 = lagrangecxd0(Γ1)
cxd02 = lagrangecxd0(Γ2)

p1 = reshape([point[i] for point in cxd01.pos for i in 1:3], (3, length(cxd01.pos)))
p2 = reshape([point[i] for point in cxd02.pos for i in 1:3], (3, length(cxd02.pos)))

#=
A = zeros(Float64, length(cxd01.pos), length(cxd02.pos))
for i in eachindex(cxd01.pos)
    for j in eachindex(cxd02.pos)
        A[i, j] = oneoverR(Γ1.vertices[i], Γ2.vertices[j])
    end
end=#
A = assemble(op, cxd01, cxd02)
#B = assemble(op, cxd01, cxd02)
##
@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)
Uaca, Vaca, raca, caca = aca(lm, tol=10^-12, svdrecompress=false, maxrank=80)
println("Error ACA: ", norm(Uaca*Vaca - A)/norm(A))
println("n pivots:", length(raca))  


#rtrue, ctrue = fullACA(A, tol=1e-6)
#println("Error TrueACA: ", norm(A[:, ctrue]*A[rtrue, ctrue]^-1*A[rtrue, :] - A)/norm(A))
#println("n pivots:", length(raca)) 

#pivfct(x) = 1/(x)^2
#am = NestedCrossApproximation.allocate_pca_memory_rm(Float64, size(A, 1), size(A, 2), maxrank=100)
#piv = NestedCrossApproximation.PCAPivoting(SVector(0.5, 0.5, 0.0), cxd02.pos, pivfct)
#Upca, Vpca, rpca, cpca = NestedCrossApproximation.pca_rm(lm, am, piv, tol=1e-6)
#println("Error PCA: ", norm(A[:, cpca]*A[rpca, cpca]^-1*A[rpca, :] - A)/norm(A))
#println("n pivots:", length(rpca)) 

##
using Plots
plotlyjs()

scatter(p1[1, :], p1[2, :], color=RGBA(0.8, 0.8, 0.8, 1.0), ratio=1)
scatter!(p2[1, :], p2[2, :], color=RGBA(0.8, 0.8, 0.8, 1.0))
scatter!(p1[1, raca], p1[2, raca], color=:red, markersize=10)
scatter!(p2[1, caca], p2[2, caca], color=:red, markersize=10)
scatter!(p1[1, rtrue], p1[2, rtrue], color=:green, markersize=7)
scatter!(p2[1, ctrue], p2[2, ctrue], color=:green, markersize=7)
scatter!(p1[1, rpca], p1[2, rpca], color=:blue)
scatter!(p2[1, cpca], p2[2, cpca], color=:blue)


##
rs = norm.([cxd01.pos[1] .- i for i in cxd02.pos])
f(x) = 1/(x)
plot(rs, A[1, :]*1e-1)
plot!(rs, abs.(B[1, :]))
