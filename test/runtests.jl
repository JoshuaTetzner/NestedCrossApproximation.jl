using Test
using TestItems
using TestItemRunner
using NestedCrossApproximation
using Random

Random.seed!(1234)

@testset "NestedCrossApproximation" begin
    # Shared helpers first.
    include("testutils.jl")

    # Same order as the includes in src/NestedCrossApproximation.jl.
    include("admissibility/test_nearinteractions.jl")
    include("admissibility/test_farinteractions.jl")
    include("admissibility/directionalsubdivision/test_abstractdirections.jl")
    include("admissibility/directionalsubdivision/test_twondirections.jl")
    include("admissibility/directionalsubdivision/test_boundingballdirections.jl")

    include("utils/test_utils.jl")
    # utils/workload.jl (no test file)
    include("test_nestedbasis.jl")
    include("test_transfermatrices.jl")

    include("compressors/test_abstractcompressor.jl")
    include("compressors/test_topdown.jl")
    include("compressors/test_bottomup.jl")
    include("compressors/test_batchedbottomup.jl")
    include("compressors/test_buffer.jl")

    include("pivoting/test_representor.jl")
    include("pivoting/test_pivots.jl")

    include("test_couplingmatrices.jl")

    include("nca/test_abstractnca.jl")
    # nca/{aggregate,couple,disaggregate}.jl covered by test_petrovgalerkinnca.jl
    include("nca/test_petrovgalerkinnca.jl")
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
