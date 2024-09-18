using BEAST
using FastBEAST
using NestedCrossApproximation
using BenchmarkTools


function compareaca(op, space::BEAST.Space, tree, filename; η=1.0, tol=1e-4, maxrank=50, multithreading=true)
   
    compressor = FastBEAST.ACAOptions(
        tol=tol, maxrank=maxrank
    )
    

    println("ACA")
    taca = @elapsed hmataca = FastBEAST.HM.assemble(
        op, space, compressor=compressor, 
        tree=tree, η=η, multithreading=multithreading
    );

    saca = FastBEAST.HM.storage(hmataca)
    hmat = FastBEAST.HM.GalerkinHMatrix{Int, ComplexF64}(
        hmataca.nearinteractions,
        hmataca.farinteractions,
        hmataca.dim,
        false
    )
    x = rand(ComplexF64, size(hmataca, 2))
    #println(hmat)
    mvt = Float64[]

    @elapsed y = hmat * x 
    @time push!(mvt, @elapsed y = hmat * x)
    @time push!(mvt, @elapsed y = hmat * x) 
    @time push!(mvt, @elapsed y = hmat * x) 
    @time push!(mvt, @elapsed y = hmat * x)    
    @time push!(mvt, @elapsed y = hmat * x) 
    mvt = minimum(mvt)

    results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
    results = results * "level: " * string(length(tree.levels)) * ", tol: " * string(tol) * ", η: " * string(η)
    results = results * "\nN \t storcomp \t fulltime\t time-mv \n"
    #---------------------------------------
    # Write data
    #---------------------------------------
    file = open(pwd() * filename, "r")
    oldresults = read(file, String)
    close(file)
    file = open(pwd() * filename, "w")
    results = oldresults * results * string(length(space.pos)) * "\t" * 
        string(saca)* "\t" *
        string(taca) * "\t" *
        string(mvt) * "\n \n" 
    write(file, results)
    close(file)
    #--------------------------------------
    # Finished data
    #--------------------------------------
end
