

function compareacanca(op, space::BEAST.Space, tree, filename; η=1.0, tol=1e-4, maxrank=50)

    println("ACA")
    tnca = @elapsed h2mataca = NestedCrossApproximation.PetrovGalerkinNCA(
        op, space, space, compressor=FastBEAST.ACAOptions(tol=tol, maxrank=maxrank), testtree=tree, trialtree=tree, η=η, verbose=false
    );

    compressor = FastBEAST.ACAOptions(
        tol=tol, maxrank=maxrank
    )
    
    println("ACA")
    taca = @elapsed hmataca = FastBEAST.HM.assemble(
        op, space, space, compressor=compressor, testtree=tree, trialtree=tree, η=η
    );

    results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
    results = results * "level: " * string(length(tree.levels)) * ", tol: " * string(tol) * ", η: " * string(η)
    results = results * "\nN \t time nca \t time aca \n"
    #---------------------------------------
    # Write data
    #---------------------------------------
    file = open(pwd() * filename, "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * filename, "w")
    results = oldresults * results * string(length(space.pos)) * "\t" * 
        string(tnca)* "\t" *
        string(taca) * "\n \n" 
    write(file, results)
    close(file)
    #--------------------------------------
    # Finished data
    #--------------------------------------
end