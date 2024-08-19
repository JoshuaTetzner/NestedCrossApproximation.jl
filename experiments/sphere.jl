using BEAST

using CompScienceMeshes
using Dates

include("main.jl")
##
hs = [0.05, 0.04, 0.03, 0.02, 0.01]

for h in hs
    println(h)
    Γ = meshsphere(1.0, h)
    op = Helmholtz3D.singlelayer()
    space = lagrangec0d1(Γ)
    file = "/experiments/HH3D_c0d1_FD.txt"

    @time compare(op, space, file)
end

##

hs = [0.05, 0.04, 0.03, 0.02, 0.015]

for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(wavenumber=k)
    space = raviartthomas(Γ)
    file = "/experiments/MW3D_FD.txt"

    @time compare(op, space, file)
end

##
hs = [0.05, 0.04, 0.03, 0.02, 0.015, 0.014, 0.013]
etas = [0.8, 1.0, 1.5]
tols = [1e-3, 1e-4]
for η in etas
    for tol in tols
        for h in hs
            println(h)
            λ = 3
            k = 2 * π / λ
            Γ = meshsphere(1.0, h)
            op = Helmholtz3D.hypersingular()
            space = lagrangec0d1(Γ)
            file = "/experiments/HH3D_HS_r_c0d1.txt"
        
            @time compare(op, space, file, tol=tol, η=η)
        end
    end
end

##
η=1.0
hs = [0.012, 0.011]
tol=1e-4
for h in hs
    println(h)
    λ = 3
    k = 2 * π / λ
    Γ = meshsphere(1.0, h)
    op = Helmholtz3D.hypersingular()
    space = lagrangec0d1(Γ)
    file = "/experiments/HH3D_HS_r_c0d1.txt"

    @time compare(op, space, file, tol=tol, η=η)
end
##

ηs = [1.0]
hs = [0.02]#[0.05, 0.04, 0.03, 0.02, 0.015, 0.013]
tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
for η in ηs
    for h in hs
        for tol in tols
            println(h)
            λ = 3
            k = 2 * π / λ
            Γ = meshsphere(1.0, h)
            op = Helmholtz3D.hypersingular()
            space = lagrangec0d1(Γ)
            file = "/experiments/HH3D_HS_c0d1.txt"

            @time compare(op, space, file, tol=tol, η=η)
        end
    end
end