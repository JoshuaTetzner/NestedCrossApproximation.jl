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
            "H²-Matrices" => "./manual/h2matrix.md",
            "Directional H²-Matrices" => "./manual/dh2matrix.md",
            "Application Examples" => "./manual/examples.md",
        ],
        "Theory" => Any[
            "H²-Matrices" => "./details/h2matrix.md",
            "Directional H²-Matrices" => "./details/dh2matrix.md",
        ],
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaTetzner/NestedCrossApproximation.jl.git",
    target="build",
    devbranch="main",
    versions=["stable" => "v^", "dev" => "dev"],
)
