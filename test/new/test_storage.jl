using BEAST
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes

##Bebendorf
h = 0.02
k = 0.15 / h
λ = 2 * pi / k
##
path = "/home/jt286/.julia/dev/NestedCrossApproximation/test/geo/ellipsoid.geo"
pathsave = "/home/jt286/.julia/dev/NestedCrossApproximation/test/geo/ellipsoid.msh"
run(`gmsh $path -2 -clmax $h -format msh2 -o $pathsave`)
##
Γ = CompScienceMeshes.read_gmsh_mesh(pathsave)
X = raviartthomas(Γ)
nmin = 100
tree = H2Trees.TwoNTree(X, X, 0.01; minvaluestest=nmin, minvaluestrial=nmin)
values, nearvalues = H2Trees.nearinteractions(
    tree; isnear=NestedCrossApproximation.isnear(k), extractselfvalues=false
)
nearstorage = 0
for i in eachindex(values)
    nearstorage += length(values[i]) * length(nearvalues[i])
end
println(nearstorage * 8 / 10^6, " MB")

length(X)
##

k = 12.0
h = 1.3 / k

Γ = meshsphere(1.0, h; generator=:gmsh)
X = raviartthomas(Γ)
nmin = 50
tree = H2Trees.TwoNTree(X, X, 0.01; minvaluestest=nmin, minvaluestrial=nmin)
values, nearvalues = H2Trees.nearinteractions(
    tree;
    isnear=NestedCrossApproximation.isnear(k; ηₗ=5.0, ηₕ=20.0),
    extractselfvalues=false,
)

##

nearstorage = 0
for i in eachindex(values)
    nearstorage += length(values[i]) * length(nearvalues[i])
end
println(nearstorage * 8 / 10^6, " MB")
println("Börm ref: ", 131000 * 23 / 10^3, " MB")

# börm ref
