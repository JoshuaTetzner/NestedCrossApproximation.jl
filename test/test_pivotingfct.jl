using BEAST
using CompScienceMeshes
using Plots
plotlyjs()
using StaticArrays
using LinearAlgebra
##
Γ1 = meshrectangle(1.0, 1.0, 0.3)
Γ2 = CompScienceMeshes.translate(meshrectangle(2.0, 1.0, 0.3), SVector(2.0, 0.0, 0.0))

##

λ = 10.0
k = 2 * π / λ

op = Helmholtz3D.singlelayer(wavenumber = k)
tspace = lagrangec0d1(Γ1)
sspace = lagrangec0d1(Γ2)
M = assemble(op, tspace, sspace)
r = [norm(tspace.pos[1]-s) for s in sspace.pos]
fct(r) = 1/r
plot(r, abs.(M[1, :]))
ys = fct.(r)
y = maximum(abs.(M[1, :]))/maximum(ys)*fct.(r)
plot!(r, y)

##
λ = 2.0
k = 2 * π / λ

op = Helmholtz3D.hypersingular(wavenumber = k)
tspace = lagrangec0d1(Γ1)
sspace = lagrangec0d1(Γ2)
M = assemble(op, tspace, sspace)
r = [norm(tspace.pos[1]-s) for s in sspace.pos]
fct(r) = 1/r^(2.2)
plot(r, abs.(M[1, :]))
ys = fct.(r)
y = maximum(abs.(M[1, :]))/maximum(ys)*fct.(r)
plot!(r, y)

##
λ = 10.0
k = 2 * π / λ

op = Maxwell3D.singlelayer(wavenumber=k, beta=0.0im)
tspace = raviartthomas(Γ1)
sspace = raviartthomas(Γ2)
M = assemble(op, tspace, sspace)
r = [norm(tspace.pos[1]-s) for s in sspace.pos]
fct(r) = 1/r^3 + 1/r
scatter(r, abs.(M[1, :]))
ys = fct.(r)
y = maximum(abs.(M[1, :]))/maximum(ys)*fct.(r)
scatter!(r, y)
inds = findall(x-> x < 0.00009, abs.(M[1, :]))
pos = zeros(Float64, length(inds), 3)
for i in eachindex(inds)
    pos[i, 1] = sspace.pos[i][1]
    pos[i, 2] = sspace.pos[i][2]
    pos[i, 3] = sspace.pos[i][3]
end
pos
##
using PlotlyJS

PlotlyJS.plot([
    CompScienceMeshes.wireframe(Γ2),
    PlotlyJS.scatter3d(x = pos[:, 1], y=pos[:, 2], z = pos[:,3], mode = "markers"),
    PlotlyJS.scatter3d(
        x=[tspace.pos[1][1]],
        y=[tspace.pos[1][2]],
        z=[tspace.pos[1][3]], mode = "markers")])
