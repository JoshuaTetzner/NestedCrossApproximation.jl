using BEAST
using FastBEAST
using NestedCrossApproximation
using BenchmarkTools

function storage(h2mat)
    ref = size(h2mat, 1)*size(h2mat, 2)
    h2stor = 0.0
    for frb in h2mat.nearinteractions.self
        h2stor += length(frb.τ)*length(frb.σ)
    end
    for frb in h2mat.nearinteractions.nears
        h2stor += length(frb.τ)*length(frb.σ)
    end
    for lrb in h2mat.i2otranslator
        h2stor += size(lrb.Z.M, 1)*size(lrb.Z.M, 2)
    end
    for ind in eachindex(h2mat.momentcollection)
        if isassigned(h2mat.momentcollection, ind)
            ntb = h2mat.momentcollection[ind]
            if ntb.T isa Vector
                for t in ntb.T
                    h2stor += size(t, 1)*size(t, 2)
                end
            else
                h2stor += size(ntb.T, 1)*size(ntb.T, 2)
            end
        end
    end

    for ind in eachindex(h2mat.translator)
        if isassigned(h2mat.translator, ind)
            ntb = h2mat.translator[ind]
            if ntb.T isa Vector
                for t in ntb.T
                    h2stor += size(t, 1)*size(t, 2)
                end
            else
                h2stor += size(ntb.T, 1)*size(ntb.T, 2)
            end
        end
    end

    return h2stor * 8 * 10^-9, h2stor/ref
end

function lowrankmatrix(h2mat::NestedCrossApproximation.GalerkinNCA, ::Type{K}) where K
    return NestedCrossApproximation.GalerkinNCA{K}(
        h2mat.tree,
        FastBEAST.GalerkinBlockMatrix{Int, K, FastBEAST.MatrixBlock{Int, K, Matrix{K}}}(
            MatrixBlock{Int, K, Matrix{K}}[], MatrixBlock{Int, K, Matrix{K}}[], size(h2mat)
        ),
        h2mat.momentcollection,
        h2mat.i2otranslator,
        h2mat.translator,
        h2mat.fars,
        h2mat.dim,
        h2mat.verbose,
        h2mat.ismultithreaded
    )    
end

function lowrankmatrix(h2mat::NestedCrossApproximation.PetrovGalerkinNCA, ::Type{K}) where K
    return NestedCrossApproximation.PetrovGalerkinNCA{K}(
        h2mat.tree,
        FastBEAST.BlockMatrix{Int, K, FastBEAST.MatrixBlock{Int, K, Matrix{K}}}(
            MatrixBlock{Int, K, Matrix{K}}[], size(h2mat)
        ),
        h2mat.testmomentcollection,
        h2mat.trialmomentcollection,
        h2mat.i2itranslator,
        h2mat.i2otranslator,
        h2mat.o2otranslator,
        h2mat.fars,
        h2mat.dim,
        h2mat.verbose,
        h2mat.ismultithreaded
    )    
end

function comparenca(op, space::BEAST.Space, tree, filename; η=1.0, tol=1e-4, multithreading=false)

    println("ACA")
    taca = @elapsed h2mataca = NestedCrossApproximation.GalerkinNCA(
        op, space, compressor=FastBEAST.ACAOptions(tol=tol, maxrank=50), tree=tree, 
        η=η, verbose=false, multithreading=multithreading
    );

    saca = storage(h2mataca)

    x = rand(ComplexF64, size(h2mataca, 2))
    mvaca = @elapsed y = h2mataca*x
    mvaca = @elapsed y = h2mataca*x
    mvaca += @elapsed y = h2mataca*x
    mvaca += @elapsed y = h2mataca*x
    mvaca += @elapsed y = h2mataca*x
    mvaca += @elapsed y = h2mataca*x
    mvaca = mvaca/5


    results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
    results = results * "NCA: \n level: " * string(length(tree.levels)) * ", tol: " * string(tol) * ", η: " * string(η)
    results = results * "\nN \t storcomp-aca \t fulltime-aca \t time-mv pca \t err_pca \t err_aca \t lrb err_pca \t lrb err_aca\n"
    #---------------------------------------
    # Write data
    #---------------------------------------
    file = open(pwd() * filename, "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * filename, "w")
    results = oldresults * results * string(length(space.pos)) * "\t" * 
        string(saca) * "\t" * 
        string(taca)* "\t" *
        string(mvaca) * "\n \n" 
    write(file, results)
    close(file)
    #--------------------------------------
    # Finished data
    #--------------------------------------
end
