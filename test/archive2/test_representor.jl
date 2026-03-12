##
ref = [(@SVector rand(3)).*SVector(1.0, 4.0, 0.0) .+ SVector(1.0, 0.0, 4.0) for i = 1:100]
vp=zeros(length(ref), 3)
for r in eachindex(ref)
    vp[r, 1] = ref[r][1]
    vp[r, 2] = ref[r][2]
    vp[r, 3] = ref[r][3]
end
Plots.scatter(vp[:, 1], vp[:, 2], vp[:, 3])
rot, hs, ct = covmat(ref)
hs
rot
x = [(@SVector rand(3)).*SVector(1.0, 1.0, 1.0) - SVector(0.5, 0.5, 0.5)  for i = 1:100]
nodess = mapper(rot, 2hs, ct, x)

vp=zeros(length(nodess), 3)
for r in eachindex(nodess)
    vp[r, 1] = nodess[r][1]
    vp[r, 2] = nodess[r][2]
    vp[r, 3] = nodess[r][3]
end
Plots.scatter!(vp[:, 1], vp[:, 2], vp[:, 3])
##
x = [(@SVector rand(3)).*SVector(1.0, 1.0, .0) - SVector(0.5, 0.5, 0.0) for i = 1:100]
vp=zeros(length(x), 3)
for r in eachindex(x)
    vp[r, 1] = x[r][1]
    vp[r, 2] = x[r][2]
    vp[r, 3] = x[r][3]
end
Plots.scatter(vp[:, 1], vp[:, 2], vp[:, 3])


##
ct = sum(x)./length(x)
map = [-1/sqrt(2) 1/sqrt(2) 0; 1/sqrt(2) 1/sqrt(2) 0; 0 0 1]


for i in eachindex(x)
    x[i] = map*(x[i].*hs) + SVector(0, 0, 1.0)
end


vp=zeros(length(x), 3)
for r in eachindex(x)
    vp[r, 1] = x[r][1]
    vp[r, 2] = x[r][2]
    vp[r, 3] = x[r][3]
end
Plots.scatter!(vp[:, 1], vp[:, 2], vp[:, 3])
##