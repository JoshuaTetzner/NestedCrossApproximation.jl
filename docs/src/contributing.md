# Contributing

Contributions are welcome. Please open a pull request against the `dev` branch. Before doing so:

- Follow the style of the surrounding code.
- Supplement the documentation.
- Write tests and verify that the test suite passes.

---

## Style

A consistent style is enforced by [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) using the rules in `.JuliaFormatter.toml`. Before committing, run:

```julia
using JuliaFormatter
format(pkgdir(NestedCrossApproximation))
```

!!! note
    Correct formatting is verified as part of the test suite. The CI build will fail if any file is not formatted.

---

## Documentation

Document any new public function, type, or keyword with a docstring. Follow the style of existing docstrings and add entries to the relevant pages under `docs/src/`. Build the docs locally to check for broken links or failed `@example` blocks:

```
julia --project=docs/ docs/make.jl
```

---

## Tests

Add tests for any new functionality. Run the full suite with:

```julia
using Pkg
Pkg.test("NestedCrossApproximation")
```

Coverage can be measured locally:

```julia
using Pkg
Pkg.test("NestedCrossApproximation"; coverage=true)

using Coverage
coverage = process_folder(pkgdir(NestedCrossApproximation) * "/src")
covered, total = get_summary(coverage)
@info "Coverage: $covered / $total lines ($(round(Int, 100covered/total)) %)"
```
