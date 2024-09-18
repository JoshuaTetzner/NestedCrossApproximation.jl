using BEAST
using Dates
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using BenchmarkTools

include("../hmat.jl")
include("../h2mat.jl")

##
hs = ["3216", "13134", "52530", "210114", "840450"]#, 0.0085]
for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh")
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/rafale/hmat.txt"

    @time comparehmat(op, space, file, tol=1e-4, η=1.0)
end

for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh")
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/rafale/h2mat.txt"

    @time compare(op, space, file, tol=1e-4, η=1.0)
end
##