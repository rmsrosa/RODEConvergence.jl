```@meta
EditURL = "../../literate/examples/11-eachnoise.jl"
```

# Scalar equation with different noises

This time we consider a family of scalar equations with different noises, similarly to what we have done earlier, but now with varying parameters, to see how that affects the convergence of the Heun method.

## The equation

More precisely, we consider the nonlinear RODE
```math
  \begin{cases}
    \displaystyle \frac{\mathrm{d}X_t}{\mathrm{d} t} = - \|Y_t\|^2 (X_t - X_t^3) + sin(Y_t), \qquad 0 \leq t \leq T, \\
  \left. X_t \right|_{t = 0} = X_0,
  \end{cases}
```
where $\{Y_t\}_{t\geq 0}$ is a scalar noise.

The *target* solution is construced by solving the system via Heun method at a much higher resolution.

## Numerical approximation

### Setting up the problem

First we load the necessary packages

````@example 11-eachnoise
using Plots
using Measures
using LinearAlgebra
using Random
using Distributions
using RODEConvergence
````

Then we set up some variables. First, the RNG:

````@example 11-eachnoise
rng = Xoshiro(123)
nothing # hide
````

The time interval is given by the end points

````@example 11-eachnoise
t0, tf = 0.0, 1.0
nothing # hide
````

and the mesh parameters are set to

````@example 11-eachnoise
ntgt = 2^18
ns = 2 .^ (6:9)
````

and

````@example 11-eachnoise
nsample = ns[[1, 2, 3]]
````

Now we define all the noise parameters, along with the number of Monte-Carlo simulations used for each noise

````@example 11-eachnoise
y0 = 0.2

ν = 0.3
σ = 0.5

μ = 0.3

μ1 = 0.5
μ2 = 0.3
ω = 3π
primitive_of_a(t; μ1 = 0.5, μ2 = 0.3, ω = 3π) = μ1 * t - μ2 * cos(ω * t) / ω
primitive_of_b2(t; σ = 0.5, ω = 3π) = σ^2 * ( t/2 - sin(ω * t) * cos(ω * t) / 2ω )

λ = 2.0
dylaw = Exponential(0.5)

steplaw = Uniform(0.0, 1.0)

λ₀ = 1.0
a = 0.5
δ = 2.0

nr = 6
transportcbrt(t, r) = mapreduce(ri -> cbrt(sin(ri * t)), +, r) / length(r)
transportabs(t, r) = mapreduce(ri -> abs(sin(ri * t)), +, r) / length(r)
transport23(t, r) = mapreduce(ri -> sign(sin(ri * t)) * cbrt(sin(ri * t)) ^ 2, +, r) / length(r)
transportsqrt(t, r) = mapreduce(ri -> sign(sin(ri * t)) * sqrt(abs(sin(ri * t))), +, r) / length(r)
transport(t, r) = mapreduce(ri -> sin(ri * t), +, r) / length(r)
ylaw = Gamma(7.5, 2.0)

nothing # hide
````

The list of noises, with the abbreviation, qualification, the noise itself, and the number of Monte-Carlo simulations

````@example 11-eachnoise
noises = [
    ("W", "Wiener process", WienerProcess(t0, tf, 0.0), 80)
    ("cP", "Compound Poisson λ = $λ, dylaw = $dylaw", CompoundPoissonProcess(t0, tf, λ, dylaw), 200)
    ("sP", "Poisson Step", PoissonStepProcess(t0, tf, λ, steplaw), 120)
    ("H-Exp", "Hawkes Exponential", ExponentialHawkesProcess(t0, tf, λ₀, a, δ, Exponential(0.5)), 400)
    ("H-Beta", "Hawkes Beta", ExponentialHawkesProcess(t0, tf, 2λ₀, a/2, 3δ, Beta(2, 3)), 400)
    ("H-Gamma", "Hawkes Gamma", ExponentialHawkesProcess(t0, tf, 2λ₀, a/2, δ/3, Gamma(1, 0.5)), 400)
    ("T", "Transport", TransportProcess(t0, tf, ylaw, transport, nr), 60)
    ("T-abs", "Transport abs", TransportProcess(t0, tf, ylaw, transportabs, nr), 60)
    ("T-2/3", "Transport two-thirds", TransportProcess(t0, tf, ylaw, transport23, nr), 40)
    ("T-sqrt", "Transport sqrt", TransportProcess(t0, tf, ylaw, transportsqrt, nr), 40)
    ("T-cbrt", "Transport cbrt", TransportProcess(t0, tf, ylaw, transportcbrt, nr), 40)
    ("fBm 0.6", "Fractional Brownian motion H = 0.6", FractionalBrownianMotionProcess(t0, tf, y0, 0.6, ntgt), 80)
    ("fBm 0.75", "Fractional Brownian motion H = 0.75", FractionalBrownianMotionProcess(t0, tf, y0, 0.75, ntgt), 100)
    ("fBm 0.9", "Fractional Brownian motion H=0.9", FractionalBrownianMotionProcess(t0, tf, y0, 0.9, ntgt), 100)
]
nothing # hide
````

### Scalar equations with the individual noises

Now we simulate a series of Random ODEs, each with one of the noises above, instead of the system with all combined noises.

In the univariate case, the right hand side of the equation becomes

````@example 11-eachnoise
f(t, x, y, p) = y^2 * (x - x^3) + sin(y)

params = nothing
````

The initial condition is also univariate

````@example 11-eachnoise
eachx0law = Normal()
````

and so is the Euler method

````@example 11-eachnoise
targetHeun = RandomHeun()
methodEuler = RandomEuler()
methodHeun = RandomHeun()
````

Now we compute the error for each noise and gather the order of convergence in a vector.

````@example 11-eachnoise
psHeun = Float64[]
pminsHeun = Float64[]
pmaxsHeun = Float64[]

for noisej in noises
    suiteHeun = ConvergenceSuite(t0, tf, eachx0law, f, noisej[3], params, targetHeun, methodHeun, ntgt, ns, noisej[4])

    @time resultHeun = solve(rng, suiteHeun)

    @info "noise = $(noisej[2]) => p = $(resultHeun.p) ($(resultHeun.pmin), $(resultHeun.pmax))"

    push!(psHeun, resultHeun.p)
    push!(pminsHeun, resultHeun.pmin)
    push!(pmaxsHeun, resultHeun.pmax)
end
````

We print them out for clarity:

````@example 11-eachnoise
for (noisej, pj, pminj, pmaxj) in zip(noises, psHeun, pminsHeun, pmaxsHeun)
    println("$(noisej[1]) ($(noisej[2]) & $(round(pj, sigdigits=6)) & $(round(pminj, sigdigits=6)) & $(round(pmaxj, sigdigits=6)) \\\\")
end
````

The following plot helps in visualizing the result.

````@example 11-eachnoise
plt_Heun = plot(title="Order of convergence for the Heun method with various", titlefont=10, ylims=(-0.1, 2.2), ylabel="\$p\$", guidefont=10, legend=:bottomright)
scatter!(plt_Heun, getindex.(noises, 1), psHeun, yerror=(psHeun .- pminsHeun, pmaxsHeun .- psHeun), xrotation = 30, label="Heun")
hline!(plt_Heun, [1.0], linestyle=:dash, label="p=1",bottom_margin=5mm, left_margin=5mm)

display(plt_Heun)
````

````@example 11-eachnoise
nothing
````

Now we do the Euler

````@example 11-eachnoise
psEuler = Float64[]
pminsEuler = Float64[]
pmaxsEuler = Float64[]

for noisej in noises
    suiteEuler = ConvergenceSuite(t0, tf, eachx0law, f, noisej[3], params, targetHeun, methodEuler, ntgt, ns, noisej[4])

    @time resultEuler = solve(rng, suiteEuler)

    @info "noise = $(noisej[2]) => p = $(resultEuler.p) ($(resultEuler.pmin), $(resultEuler.pmax))"

    push!(psEuler, resultEuler.p)
    push!(pminsEuler, resultEuler.pmin)
    push!(pmaxsEuler, resultEuler.pmax)
end
````

We print them out for clarity:

````@example 11-eachnoise
for (noisej, pj, pminj, pmaxj) in zip(noises, psEuler, pminsEuler, pmaxsEuler)
    println("$(noisej[1]) ($(noisej[2]) & $(round(pj, sigdigits=6)) & $(round(pminj, sigdigits=6)) & $(round(pmaxj, sigdigits=6)) \\\\")
end
````

and add to the plot

````@example 11-eachnoise
plt_Euler = plot(title="Order of convergence for the Euler method with various", titlefont=10, ylims=(-0.1, 2.2), ylabel="\$p\$", guidefont=10, legend=:bottomright)
scatter!(plt_Euler, getindex.(noises, 1), psEuler, yerror=(psEuler .- pminsEuler, pmaxsEuler .- psEuler), xrotation = 30, label="Euler")
hline!(plt_Euler, [1.0], linestyle=:dash, label="p=1",bottom_margin=5mm, left_margin=5mm)

display(plt_Euler)
````

Strong order $p$ of convergence of the Euler method for $\mathrm{d}X_t/\mathrm{d}t = - Y_t^2 X_t + Y_t$ for a series of different noise $\{Y_t\}_t$ (scattered dots: computed values; dashed line: expected $p = 1;$ with 95% confidence interval).

````@example 11-eachnoise
nothing
````

# IVP
Playing with some noises

````@example 11-eachnoise
x0 = 0.5

tt = range(t0, tf, length=last(ns)+1)
xt = zeros(last(ns) + 1)
yt = zeros(last(ns) + 1)

noises_for_samples = [
    noises[begin:end-3]
    ("fBm 0.6", "Fractional Brownian motion H = 0.6", FractionalBrownianMotionProcess(t0, tf, y0, 0.6, last(ns)))
    ("fBm 0.8", "Fractional Brownian motion H = 0.8", FractionalBrownianMotionProcess(t0, tf, y0, 0.75, last(ns)))
    ]

for noisej in noises_for_samples
    plt_noise = plot(title="Sample noises of $(noisej[2])", titlefont=10, legend=false)
    plt_heun = plot(title="Sample solutions with $(noisej[2])", titlefont=10, legend=false)
    println("noisej = $(noisej[1])")
    for i in 1:10
        rand!(rng, noisej[3], yt)
        plot!(plt_noise, tt, yt)
        solve!(xt, t0, tf, x0, f, yt, params, RandomHeun())
        plot!(plt_heun, tt, xt)
    end
    display(plt_noise)
    display(plt_heun)
end
````

````@example 11-eachnoise
nothing
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

