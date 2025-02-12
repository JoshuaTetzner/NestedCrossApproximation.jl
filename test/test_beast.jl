using CompScienceMeshes, BEAST
using LinearAlgebra
using FastBEAST
using NestedCrossApproximation
using IterativeSolvers

Γ = meshsphere(1.0, 0.06)
X = raviartthomas(Γ)
Y = BEAST.buffachristiansen(Γ)
f = 1e8

μ = 4π * 1e-7
ε = 8.854187812e-12
c = 1 / sqrt(ε * μ)

λ = c / f
k = 2π / λ
ω = 2π * f
##
𝓣 = Maxwell3D.singlelayer(; wavenumber=k, alpha=-im * ω * μ, beta=-1 / (im * ω * ε))
#TA = Maxwell3D.singlelayer(; wavenumber=k, alpha=-im * ω * μ, beta=0.0im)
#Tphi = Maxwell3D.singlelayer(; wavenumber=k, alpha=0.0im, beta=-1 / (im * ω * ε))

𝓝 = BEAST.NCross()

𝑬 = Maxwell3D.planewave(; direction=-x̂, polarization=ẑ, wavenumber=k)
𝒆 = (n × 𝑬) × n
##
function build(op, X, Y)
    testtree = create_tree(X.pos, KMeansTreeOptions(; nmin=50, maxlevel=50))
    trialtree = create_tree(Y.pos, KMeansTreeOptions(; nmin=50, maxlevel=50))
    testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
    trialcomp = NestedCrossApproximation.TopDownCompressor(
        NestedCrossApproximation.iACA(
            space.pos;
            rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
            columnpivoting=FastBEAST.LRF.MaximumValue(),
        ),
        nothing,
    )
    return PetrovGalerkinNCA(
        op,
        X,
        Y;
        testtree=testtree,
        trialtree=trialtree,
        testcompressor=testcomp,
        trialcompressor=trialcomp,
        multithreading=true,
        maxrank=100,
    )
end

function buildHM(op, X, Y)
    return HM.assemble(op, X, Y)
end

##
@time Txx = build(𝓣, X, X);
println("primal discretisation assembled.");
#@time assemble(𝓣, Y, Y);
@time Tyy = build(𝓣, Y, Y);
@time Tyy = HM.assemble(𝓣, Y, Y);

##
println("dual discretisation assembled.");
Nxy = assemble(𝓝, X, Y;);
println("duality form assembled.");
e = assemble(𝒆, X)
##
iNxy = BEAST.GMRESSolver(Nxy; restart=50, reltol=1e-12, maxiter=1000)
##
x, ch = IterativeSolvers.gmres(
    Tyy * iNxy * Txx,
    Tyy * (iNxy * e);
    maxiter=200,
    restart=50,
    reltol=1e-3,
    log=true,
    verbose=true,
)

xf, ch = IterativeSolvers.gmres(
    Tyy * iNxy * Txxf,
    Tyy * (iNxy * e);
    maxiter=100,
    restart=50,
    reltol=1e-3,
    log=true,
    verbose=true,
)
##
fcr, geo = facecurrents(x, X)

using WriteVTK
cellV = typeof(MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 4, 3]))[]
for (ind, face) in enumerate(Γ.faces)
    push!(cellV, MeshCell(VTKCellTypes.VTK_TRIANGLE, [face[1], face[2], face[3]]))
end

vtk_grid(pwd() * "/h2mat", vertexarray(Γ)', cellV) do vtk
    vtk["amplitude"] = norm.(fcr)
    vtk["real"] = norm.(real.(fcr))
    vtk["imag"] = norm.(imag.(fcr))
end

fcr, geo = facecurrents(xf, X)

using WriteVTK
cellV = typeof(MeshCell(VTKCellTypes.VTK_TRIANGLE, [2, 4, 3]))[]
for (ind, face) in enumerate(Γ.faces)
    push!(cellV, MeshCell(VTKCellTypes.VTK_TRIANGLE, [face[1], face[2], face[3]]))
end

vtk_grid(pwd() * "/fullassembled", vertexarray(Γ)', cellV) do vtk
    vtk["amplitude"] = norm.(fcr)
    vtk["real"] = norm.(real.(fcr))
    vtk["imag"] = norm.(imag.(fcr))
end
