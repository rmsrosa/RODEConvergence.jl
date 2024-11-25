## Overview

This is the core package used for the numerical examples appearing in the paper "Strong order-one convergence of the Euler method for random ordinary differential equations driven by semi-martingale noises", by Peter E. Kloeden and Ricardo M. S. Rosa.

The package contains the implementation of the Euler and Heun methods for scalar equations and systems of equations and all the helper functions needed to define the noises, setup the problem, check the convergence of the methods, and plot the desired figures. The methods defined in this local package can be seen in the section [API](api.md).

The codes are written fully in the [Julia programming language](https://julialang.org).

The local package `RODEConvergence.jl` is *not* a registered package in Julia, as it is only used here for research purposes on the order of convergence of numerical approximation of Random ODEs, with the bare minimum needed for it. For a much more complete package for solving Random ODEs and other types of differential equations, check the [SciML: Open Source Software for Scientific Machine Learning](https://sciml.ai) ecosystem.

For the code used here, it is illustrative to see the first example [Homogenous linear RODE with a Wiener process noise coefficient](examples/01-wiener_linearhomogeneous.md), in which all the steps are explained in more details.

We use a few standard libraries ([Random](https://docs.julialang.org/en/v1/stdlib/Random/), [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/), [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/), [Test](https://docs.julialang.org/en/v1/stdlib/Test/)) and a few packages ([JuliaStats/Distributions.jl](https://juliastats.github.io/Distributions.jl/stable/), [JuliaMath/FFTW.jl](https://juliamath.github.io/FFTW.jl/stable/), [JuliaPlots/Plots.jl](https://docs.juliaplots.org/stable)).

This documentation makes use of [Documenter.jl](https://documenter.juliadocs.org/stable/) and [Literate.jl](https://fredrikekre.github.io/Literate.jl/stable/), with the help of [LiveServer.jl](https://tlienart.github.io/LiveServer.jl/stable/) and [Revise.jl](https://timholy.github.io/Revise.jl/stable/), during development.

Some extra material uses [JuliaCI/BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable).
