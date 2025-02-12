using NestedCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using FastBEAST

Γt = meshrectangle(1.0, 1.0, 0.1)
Γs = CompScienceMeshes.translate(meshrectangle(4.0, 1.0, 0.1), SVector(2.0, 0.0, 0.0))

t = raviartthomas(Γt)
s = raviartthomas(Γs)

λ = 5
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; alpha=0.0 * im, wavenumber=k)

A = assemble(op, t, s)
@views function fct(B, x, y)
    return B[:, :] = A[x, y]
end
##
lm = LRF.LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), ComplexF64)
iaca = iACA(s.pos)
iaca = NestedCrossApproximation.init(iaca, lm; ref=SVector(0.5, 0.5, 0.0))

U = zeros(ComplexF64, length(t.pos), 100)
V = zeros(ComplexF64, 100, length(s.pos))
rpivots, cpivots = iaca(lm, V, U, 100, 1e-4)

norm(A - (A[:, cpivots] / A[rpivots, cpivots]) * A[rpivots, :]) / norm(A)
##
using Plots
norms =
    log.(
        10,
        [
            0.0038521684008272234,
            0.0005440364795558678,
            0.0006851426382786326,
            0.00123667226880349,
            0.0003486662090477208,
            0.00019116684134948644,
            4.665618000653638e-5,
            6.645098322943002e-5,
            0.00039791604128091716,
            8.187782221034195e-5,
            1.7234450362264797e-6,
            1.5781276422616232e-5,
            5.2006692247872834e-6,
            1.23356723702185e-5,
            2.268411206767885e-6,
            1.0248529289471572e-7,
            3.7388778405251596e-6,
            2.168209717056945e-6,
            2.182549445076961e-7,
            7.438123971014253e-7,
            1.522796299950526e-8,
        ],
    )
f(x) = -2.476582699991303 + -0.2031589111086878 * x
x = Vector(1:length(norms))
plot(Vector(1:length(norms)), norms)
plot!(x, f.(x))
rms = sqrt.(sum([(n - f(i))^2 for (i, n) in enumerate(norms)]) / length(norms))
plot!(x, [log(10, 2.0548232157018673e-7) for i in x])

norms[end]
f.(length(norms))
