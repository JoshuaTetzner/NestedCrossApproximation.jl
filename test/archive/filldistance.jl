using CompScienceMeshes
using BEAST
using LinearAlgebra
using StaticArrays

function fullACA(A)
    rows = Int[]
    cols = Int[]
    rc = argmax(abs.(A))
    push!(rows, rc[1])
    push!(cols, rc[2])
    R = A - A[:, rc[2]] .* 1/A[rc[1], rc[2]] * transpose(A[rc[1], :])
    refnorm = norm(A)
    iter = 1
    while norm(R) > 1e-4 * refnorm && iter < 50
        iter += 1 
        rc = argmax(abs.(R))
        push!(rows, rc[1])
        push!(cols, rc[2])
        R = R - R[:, rc[2]] .* 1/R[rc[1], rc[2]] * transpose(R[rc[1], :])
        println(norm(R))
    end
    return rows, cols
end

##

λ = 10
k = 2 * π / λ

Γ = meshsphere(0.5, 0.4)
rm = translate(meshrectangle(1.0, 2.5, 0.3), SVector(1.5, 0.0, 0.0))
rm2 = translate(meshrectangle(1.5, 1.0, 0.3), SVector(0.0, 1.5, 0.0))
Γ2 = weld(rm, rm2)
op = Maxwell3D.singlelayer(wavenumber=k)
X1 = raviartthomas(Γ)
X2 = raviartthomas(Γ2)

A = assemble(op, X1, X2)
##
r, c = fullACA(A)
##
points = X2.pos[r]
truep = reshape([point[i] for point in points for i in 1:3], (3, length(points)))

##
using Plots
plotlyjs()
#scatter(fp[1, :], fp[2, :], fp[3, :])
scatter(truep[1, :], truep[2, :], truep[3, :])