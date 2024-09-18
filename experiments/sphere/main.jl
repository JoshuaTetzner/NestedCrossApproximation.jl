using BEAST
using Dates
using FastBEAST
using StaticArrays

using CompScienceMeshes
using NestedCrossApproximation
using BenchmarkTools

include("h2mat.jl")
include("h2matpca.jl")

##
hs = [0.06, 0.0415, 0.0295, 0.0206]#, 0.0145]

for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat.txt"

    @time compare(op, space, file, tol=1e-4, η=1.0)
end
##
for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat.txt"

    @time comparepcaerr(op, space, file, tol=1e-4, η=1.0)
end
##
hs = [0.0295]
tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
for h in hs
    for tol in tols
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat.txt"

    @time comparepcaerr(op, space, file, tol=tol, η=1.0)
    end
end

## Rafale
hs = ["3216", "s_5805", "13134", "s_23210", "52530"]#, 
hs = ["210114"]
for h in hs
    println(h)
    λ = 30
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh") 
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat_rafale.txt"

    @time compare(op, space, file, tol=1e-4, η=1.0, nmin=50)
end
 
## Mig
hs = ["1726", "6898", "27586", "110338"]

for h in hs
    println(h)
    λ = 20
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/mig/Mig17_"*h*".msh")
    Γ.vertices = Γ.vertices .* 200
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat_mig.txt"

    @time comparepcaerr(op, space, file, tol=1e-4, η=1.0, nmin=50)
end
## Spaceship
hs = ["1262", "5042", "20162", "80642"]
for h in hs
    println(h)
    λ = 100
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/spaceship/Shuttle"*h*".msh")
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/sphere/h2mat_spaceship.txt"

    @time compare(op, space, file, tol=1e-4, η=1.0, nmin=50)
end