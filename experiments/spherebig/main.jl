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
hs = [0.06, 0.0415, 0.0295, 0.0206, 0.0145, 0.010175, 0.0072]
hs = [0.0145, 0.010175, 0.0072]
#hs = [0.06, 0.04, 0.02, 0.0175, 0.015, 0.0125, 0.01]

for h in hs
    println(h)
    λ = 1
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    
    file = "/experiments/spherebig/resultsaca.txt"

    compareaca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 1
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    file = "/experiments/spherebig/resultspca.txt"

    comparepca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end
##
for h in hs
    println(h)
    λ = 1
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
    
    file = "/experiments/spherebig/resultsnca.txt"

    comparenca(op, space, tree, file, tol=1e-4, η=1.0, multithreading=false)
end

