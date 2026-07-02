function collect_assigned(v::Vector{B}) where {B}
    nassigned = 0
    @inbounds for i in eachindex(v)
        nassigned += isassigned(v, i)
    end

    compact = Vector{B}(undef, nassigned)
    nodes = Vector{Int}(undef, nassigned)

    k = 0
    @inbounds for i in eachindex(v)
        if isassigned(v, i)
            k += 1
            compact[k] = v[i]
            nodes[k] = i
        end
    end

    return compact, nodes
end
