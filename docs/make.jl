#!/usr/bin/env julia

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using RODEConvergence

if "liveserver" in ARGS
    using Revise
    Revise.revise()
end

ENV["GKSwstype"] = "100"

# Generate markdown pages from Literate scripts and get the list of generated pages as `generate_examples`:
include(joinpath(@__DIR__(), "literate.jl"))

@time makedocs(
    sitename = "Euler method for RODEs",
    repo = "https://github.com/rmsrosa/RODEConvergence.jl",
    pages = [
        "Overview" => "index.md",
        "Examples" => [
            "Basic Linear RODEs" => [
                "examples/01-wiener_linearhomogeneous.md",
                "examples/02-wiener_linearnonhomogeneous.md",
                "examples/03-sin_gBm_linearhomogeneous.md",
            ],
            "examples/04-allnoises.md",
            "examples/05-fBm_linear.md",
            "examples/06-popdyn.md",
            "examples/07-toggle_switch.md",
            "examples/08-earthquake.md",
            "examples/09-risk.md",
            "examples/10-fisherkpp.md",
        ],
        "Noises" => [
            "noises/noiseintro.md",
            "noises/homlin.md",
            "noises/fBm.md"
        ],
        "api.md",
    ],
    authors = "Ricardo Rosa",
    draft = "draft" in ARGS,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/RODEConvergence.jl",
        edit_link = "main",
        repolink = "https://github.com/rmsrosa/RODEConvergence.jl",
    ),
    modules = [RODEConvergence],
)

if get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
    deploydocs(
        repo = "github.com/rmsrosa/RODEConvergence.jl.git",
        devbranch = "main",
        forcepush = true,
    )
end
