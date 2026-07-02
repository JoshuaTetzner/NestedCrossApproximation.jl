using Test
using TestItems
using TestItemRunner
using NestedCrossApproximation
using Random

Random.seed!(1234)

@testset "NestedCrossApproximation" begin
    include("test_nearinteractions.jl")
    include("test_farinteractions.jl")
    include("directionalsubdivision/test_directionaldata.jl")
    include("directionalsubdivision/test_boundingballdirections.jl")
    include("directionalsubdivision/test_twondirections.jl")

    include("testutils.jl")
    include("test_buffer.jl")
    include("test_factorization.jl")
    include("test_nestedbasis_transfermatrices.jl")
    include("test_couplingmatrices.jl")
    include("test_topdown.jl")
    include("test_bottomup.jl")
    include("test_representor.jl")
    include("test_petrovgalerkinnca.jl")
    include("test_mvp_threading.jl")
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    using NestedCrossApproximation
    # persistent_tasks resolves a fresh environment from Project.toml alone
    # (no dev/path entries), which can't work while AdaptiveCrossApproximation
    # and H2Trees are unregistered dev dependencies. Re-enable once those are
    # registered with matching features.
    Aqua.test_all(NestedCrossApproximation; persistent_tasks=false)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    using NestedCrossApproximation
    @test JuliaFormatter.format(pkgdir(NestedCrossApproximation), overwrite=false)
end

@testitem "Static analysis (JET.jl)" begin
    using JET
    using NestedCrossApproximation
    JET.test_package(NestedCrossApproximation; target_modules=(NestedCrossApproximation,))
end

@testitem "Explicit imports (ExplicitImports.jl)" begin
    using ExplicitImports
    using NestedCrossApproximation
    @test ExplicitImports.check_no_stale_explicit_imports(NestedCrossApproximation) ===
        nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(
        NestedCrossApproximation
    ) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(NestedCrossApproximation) ===
        nothing
end

@run_package_tests verbose = true
