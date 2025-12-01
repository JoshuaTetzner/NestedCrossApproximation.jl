abstract type DirectionalData end

function angle(a::SVector{3,F}, b::SVector{3,F}) where {F}
    return acos(min(dot(a, b) / (norm(a) * norm(b)), 1.0))
end

function admissiblelevel(tree, data::DirectionalData)
    lflevel = 0
    hflevel = 0
    for level in levels(tree)
        lflevelinteractions = 0
        hflevelinteractions = 0
        for node in H2Trees.LevelIterator(tree, level)
            if data.F[node] != Int[] && data.𝓔[node] != Int[]
                hflevelinteractions += 1
            elseif data.F[node] != Int[]
                lflevelinteractions += 1
            end
            (hflevelinteractions != 0 && lflevelinteractions != 0) && break
        end
        hflevelinteractions != 0 && hflevel += 1
        lflevelinteractions != 0 && lflevel += 1
    end

    return max(lflevel, hflevel)
end
