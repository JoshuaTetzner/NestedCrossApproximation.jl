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

hs = ["3216", "13134", "52530", "210114"]
hs = ["210114"]
for h in hs
    println(h)
    λ = 30
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh") 
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    
    file = "/experiments/spherebig/resultsrafaleaca.txt"

    compareaca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 30
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh") 
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    
    file = "/experiments/spherebig/resultsrafalepca.txt"

    comparepca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 30
    k = 2 * π / λ
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd()*"/experiments/rafale/geo/rafale_"*h*".msh") 
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    
    file = "/experiments/spherebig/resultsrafalenca.txt"

    comparenca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end