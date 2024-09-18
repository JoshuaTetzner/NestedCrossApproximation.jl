using BEAST
using Dates
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using BenchmarkTools

include("aca.jl")
include("pca.jl")
include("nca.jl")

##

hs = ["1726", "6898", "27586", "110338"]
hs = ["441346"]
for h in hs
    println(h)
    λ = 20
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/mig/Mig17_"*h*".msh")
    Γ.vertices = Γ.vertices .* 200
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    file = "/experiments/spherebig/resultsmigaca.txt"


    compareaca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 20
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/mig/Mig17_"*h*".msh")
    Γ.vertices = Γ.vertices .* 200
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    file = "/experiments/spherebig/resultsmigpca.txt"


    comparepca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 20
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/mig/Mig17_"*h*".msh")
    Γ.vertices = Γ.vertices .* 200
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    file = "/experiments/spherebig/resultsmignca.txt"


    comparenca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end