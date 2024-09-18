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

function comparepcaerr(op, space::BEAST.Space, filename; η=1.0, tol=1e-4, maxrank=100, nmin=50)
   
    compressor = NestedCrossApproximation.PCAOptions(
        NestedCrossApproximation.PCAPivoting(space.pos),
        NestedCrossApproximation.PCAPivoting(space.pos),
        tol=tol,
        maxrank=maxrank
    );
    
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=nmin, nchildren=2))
    println("PCA")
    tpca = @elapsed h2matpca = NestedCrossApproximation.GalerkinNCA(
        op, space, compressor=compressor, tree=tree, η=η
    );
    #println("ACA")
    #taca = @elapsed h2mataca = NestedCrossApproximation.GalerkinNCA(
    #    op, space, compressor=FastBEAST.ACAOptions(tol=tol, maxrank=50), tree=tree, η=η
    #);
    println("ref")
    ref = NestedCrossApproximation.GalerkinNCA(
        op, space, compressor=FastBEAST.ACAOptions(tol=1e-2*tol, maxrank=150), tree=tree, η=η
    );

    spca = storage(h2matpca)
    #saca = storage(h2mataca)

    lrbpca = lowrankmatrix(h2matpca, scalartype(op))
    #lrbaca = lowrankmatrix(h2mataca, scalartype(op))
    lrbref = lowrankmatrix(ref, scalartype(op))

    println("RelDif")
    #relaca = estimate_reldifference(h2mataca, ref, tol=1e-3)
    relpca = estimate_reldifference(h2matpca, ref, tol=1e-3)
    #rellrbaca = estimate_reldifference(lrbaca, lrbref, tol=1e-3)
    rellrbpca = estimate_reldifference(lrbpca, lrbref, tol=1e-3)

   
    results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
    results = results * "level: " * string(length(tree.levels)) * ", tol: " * string(tol) * ", η: " * string(η)
    results = results * "\nN \t storcomp-pca \t fulltime-pca \t time-mv pca \t time-mv pca \t err_pca \t lrb err_pca \n"
    #---------------------------------------
    # Write data
    #---------------------------------------
    file = open(pwd() * filename, "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * filename, "w")
    results = oldresults * results * string(length(space.pos)) * "\t" * 
        string(spca)* "\t" *
        string(tpca) * "\t" *
        string(relpca) * "\t" *
        string(rellrbpca) * "\n \n"  
    write(file, results)
    close(file)
    #--------------------------------------
    # Finished data
    #--------------------------------------
end
