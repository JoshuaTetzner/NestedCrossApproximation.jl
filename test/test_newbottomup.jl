using FastBEAST
using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using ClusterTrees
using StaticArrays
using Test


Γ=meshsphere(1.0, 0.08)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)
fct(x) = 1/x

piv = NestedCrossApproximation.MyPivoting(fct, space.pos)
##
compressor1 = NestedCrossApproximation.PCA2Options(
    NestedCrossApproximation.MaximumValue(),
    NestedCrossApproximation.MyPivoting(fct, space.pos),
    40,
    10^-4
);


##
tree = create_tree(space.pos, KMeansTreeOptions(nmin=50));
##

@time h2mat1 = NestedCrossApproximation.GalerkinNCA2(
    op, space, tree=tree, compressor=compressor1, multithreading=true
);
##
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    maxrank=40,
    tol=10^-4
);

@time h2mat = NestedCrossApproximation.GalerkinNCA(
    op, space, tree=tree, compressor=compressor, multithreading=true
);

##
x = rand(size(h2mat1, 2))
##
using LinearAlgebra
M = assemble(op, space, space)
norm(M*x-h2mat*x)/norm(M*x)
norm(M*x-h2mat1*x)/norm(M*x)
##
@time hmat = HM.assemble(op, space, tree=tree, multithreading=true);

##
A = zeros(10, 10)

norm(A) == 0.0

##
using LinearAlgebra

@time begin
    mydic = Dict{Int, Vector{Int}}()
    for i in 1:20000
        v = rand(Int, 100)
        push!(mydic, 1 => v)
    end

    for i in eachindex(mydic)
        mydic[i] .+= 2
    end 
end

@time begin
    mydic = Vector{Vector{Int}}(undef, 20000)
    for i in 1:20000
        v = rand(Int, 100)
        mydic[i]=v
    end

    for i in eachindex(mydic)
        if isassigned(mydic, i)
            mydic[i] .+= 2
        end
    end 
end

@time begin
    mydic = zeros(Int, 100, 20000)
    for i in 1:20000
        v = rand(Int, 100)
        mydic[1:100, i]=v
    end
    for i in 1:size(mydic, 2)
        if norm(mydic[:, i]) == 0
            mydic[:, i] .+= 2
        end
    end 
end