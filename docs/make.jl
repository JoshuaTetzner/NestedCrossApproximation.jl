using Documenter
using NestedCrossApproximation

DocMeta.setdocmeta!(
    NestedCrossApproximation,
    :DocTestSetup,
    :(using NestedCrossApproximation);
    recursive=true,
)

makedocs(;
    sitename="NestedCrossApproximation.jl",
    authors="Joshua M. Tetzner <joshua.tetzner@uni-rostock.de> and contributors",
    modules=[NestedCrossApproximation],
    pages=[
        "Introduction" => "index.md",
        "Manual" => Any[
            "General Usage" => "./manual/manual.md",
            "Application Examples" => "./manual/examples.md",
        ],
        "Further Details" => Any[
            "H2-Matrices" => "./details/h2matrices.md",
            "TopDown Compression" => "./details/topdown.md",
            "BottomUp Compression" => "./details/bottomup.md",
            "Wideband Admissibility" => "./details/wideband.md",
        ],
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaTetzner/NestedCrossApproximation.jl.git",
    target="build",
    devbranch="dev",
    versions=["stable" => "v^", "dev" => "dev"],
)
