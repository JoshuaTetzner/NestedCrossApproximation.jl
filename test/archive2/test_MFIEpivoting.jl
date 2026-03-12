using CompScienceMeshes
using NestedCrossApproximation
using StaticArrays

Γ1 = meshrectangle(1.0, 1.0, 0.1)
Γ2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(1.0, 0.0, 1.0))

##
x = NestedCrossApproximation.IACAPivotingMFIE(Γ1.vertices, Γ2.vertices)

tidcs = Vector(1:length(Γ2.vertices))
pidcs = Vector(1:length(Γ1.vertices))
y = x(tidcs, pidcs)

y.w
