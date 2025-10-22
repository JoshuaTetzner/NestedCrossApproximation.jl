using BEAST
using CompScienceMeshes
using StaticArrays
using LinearAlgebra

function mimic(
    tidx::Int,
    tpos::Vector{SVector{3,Float64}},
    spos::Vector{SVector{3,Float64}},
    usedidcs::Vector{Bool},
)
    scenter = SVector(2.5, 0.5, 0.0)#sum(spos) / length(spos)
    sxmaxp = 0.5#maximum(norm.([i[1] for i in spos] .- scenter[1]))
    smaxn = 3.5
    symax = maximum(norm.([i[2] for i in spos] .- scenter[2]))
    tcenter = sum(tpos) / length(tpos)
    txmax = maximum(norm.([i[1] for i in tpos] .- tcenter[1]))
    tymax = maximum(norm.([i[2] for i in tpos] .- tcenter[2]))

    map = tcenter - tpos[tidx]
    sx = sxmaxp
    if map[1] > 0
        sx = smaxn
    end
    map = map .* SVector(sx, symax, 0.0) ./ SVector(txmax, tymax, 1.0)
    mini = norm(spos[1] - (scenter + map))
    nextidx = 1
    for idx in eachindex(spos)
        if !usedidcs[idx] && mini > norm(spos[idx] - (scenter + map))
            mini = norm(spos[idx] - (scenter + map))
            nextidx = idx
        end
    end
    usedidcs[nextidx] = true

    return nextidx
end

function mimicri(A::Matrix, t, s; maxiter=20)
    err = Float64[]

    R = copy(A)
    usedidcs = zeros(Bool, length(s.pos))
    tidcs = []
    sidcs = [1]
    sidx = 1
    tidx = argmax(abs.(R[:, sidx]))
    push!(tidcs, tidx)
    push!(err, norm(R))
    R = R - (R[:, sidx] .* 1 / R[tidx, sidx] * transpose(R[tidx, :]))
    push!(err, norm(R))
    for i in 1:maxiter
        sidx = mimic(tidx, t.pos, s.pos, usedidcs)
        tidx = argmax(abs.(R[:, sidx]))
        push!(sidcs, sidx)
        push!(tidcs, tidx)
        R = R - (R[:, sidx] .* 1 / R[tidx, sidx] * transpose(R[tidx, :]))
        push!(err, norm(R))
    end
    return tidcs, sidcs, err
end

function fullpivots(M::Matrix{F}; maxiter=100) where {F}
    R = copy(M)
    norms = Float64[]
    refnorm = norm(R)
    rows = Int[]
    cols = Int[]
    push!(norms, refnorm)

    for i in 1:maxiter
        ij = argmax(abs.(R))
        R -= R[:, ij[2]] * R[ij[1], ij[2]]^-1 * transpose(R[ij[1], :])
        push!(rows, ij[1])
        push!(cols, ij[2])
        push!(norms, norm(R))
    end
    return rows, cols, norms
end
##
λ = 0.8
k = 2 * pi / λ

Γt = meshrectangle(1.0, 1.0, 0.08)
Γs = CompScienceMeshes.translate(meshrectangle(4.0, 1.0, 0.08), SVector(2.0, 0.0, 0.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
t = raviartthomas(Γt)
s = raviartthomas(Γs)

A = assemble(op, t, s)

ftidcs, fsidcs, ferr = fullpivots(A; maxiter=rank(A) - 3)
tidcs, sidcs, err, = mimicri(A, t, s; maxiter=rank(A) - 3)

##
using Plots
plotlyjs()

tpos = reshape([point[i] for point in t.pos for i in 1:3], (3, length(t.pos)))
spos = reshape([point[i] for point in s.pos for i in 1:3], (3, length(s.pos)))

tt = reshape([point[i] for point in t.pos[ftidcs] for i in 1:3], (3, length(ftidcs)))
ss = reshape([point[i] for point in s.pos[fsidcs] for i in 1:3], (3, length(fsidcs)))
ftt = reshape([point[i] for point in t.pos[tidcs] for i in 1:3], (3, length(tidcs)))
fss = reshape([point[i] for point in s.pos[sidcs] for i in 1:3], (3, length(sidcs)))

scatter(tpos[1, :], tpos[2, :]; aspect_ratio=1.0, mc=:orange)
scatter!(spos[1, :], spos[2, :]; mc=:orange)
scatter!(tt[1, :], tt[2, :]; mc=:blue)
scatter!(ss[1, :], ss[2, :]; mc=:blue)
scatter!(ftt[1, :], ftt[2, :]; mc=:green)
scatter!(fss[1, :], fss[2, :]; mc=:green)

##
plot(ferr; yaxis=:log)
plot!(err)
