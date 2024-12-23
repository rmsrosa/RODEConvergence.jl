```@meta
EditURL = "../../literate/examples/08-earthquake.jl"
```

# Mechanical structural under random Earthquake-like seismic disturbances

Now we consider a mechanical structure problem under ground-shaking Earthquake-like excitations. The problem is modeled by a second-order Random ODE driven by a random disturbance in the form of a transport process. The equation is inspired by the model in [Bogdanoff, Goldberg & Bernard (1961)](https://doi.org/10.1785/BSSA0510020293) (see also [Chapter 18]{NeckelRupp2013} and {HousnerJenning1964} with this and other models).

There are a number of models for earthquake-type forcing, such as the ubiquitous Kanai-Tajimi and Clough-Penzien models, where the noise has a characteristic spectral density, determined by the mechanical properties of the ground layer. The ideia, from {Kanai1957}, is that the spectrum of the noise at bedrock is characterized by a constant pattern, while at the ground surface it is modified by the vibration property of the ground layer. This interaction between the bedrock and the ground layer is modeled as a stochastic oscillator driven by a zero-mean Gaussian white noise, and whose solution leads to a noise with a characteristic power spectrum.

Another important aspect concerns the fact that the aftershocks tend to come in clusters, with the ocurrence of an event increasing the chances for subsequent events. As such, self-exciting intensity processes have been successful in modeling the arrival times of the aftershocks (see e.g. [Pratiwi, Slamet, Saputro & Respatiwulan (2017)](https://doi.org/10.1088/1742-6596/855/1/012033)). The decaying kernel is usually an inverse power law, starting with the celebrated Omori formula [T. Utsu, Y. Ogata & R. S. Matsu'ura, The centenary of the Omori formula for a decay law of aftershock activity, Journal of Physics of the Earth, Volume 43 (1995), no. 1, 1-33](https://doi.org/10.4294/jpe1952.43.1)). Exponentially decaying kernels are also used and, in this case, leads to a noise in the form of an exponentially decaying self-excited Hawkes process. The intensity, or rate, of this inhomogenous Poisson point process, for the interarrival times, is not directly related to the magnitude of the aftershocks, so this process should be coupled with another process for the magnitude of each shock.

We follow, however, the Bogdanoff-Goldberg-Bernard model, which takes the form of a transport process noise. We chose the later so we can illustrate the improved convergence for such type of noise, complementing the other examples. This model is described in more details shortly. Let us introduce first the model for the vibrations of the mechanical structure.

A single-storey building is considered, with its ground floor centered at position $M_t$ and its ceiling at position $M_t + X_t$. The random process $X_t$ refers to the motion relative to the ground. The ground motion $M_t$ affects the motion of the relative displacement $X_t$ as an excitation force proportional to the ground acceleration $\ddot M_t$. The damping and elastic forces are in effect within the structure. In this framework, the equation of motion for the relative displacement $X_t$ of the ceiling of the single storey building takes the following form.

```math
  \ddot X_t + 2\zeta_0\omega_0\dot X_t + \omega_0^2 X_t = - \ddot M_t.
```
where $\zeta_0$ and $\omega_0$ are damping and elastic model parameters depending on the structure.

For the numerical simulations, the second-order equation is written as a system of first-order equations:

```math
  \begin{cases}
      \dot X_t = V_t, \\
      \dot V_t = - \omega_0^2 X_t - 2\zeta_0\omega_0 X_t - Y_t,
  \end{cases}
```
where $\{V_t\}_t$ is the random velocity of the celing relative to the ground and where $\{Y_t\}_t$ is the stochastic noise excitation term given as the ground acceleration, $Y_t = \ddot M_t$, generated by an Earthquake and its aftershocks, or any other type of ground motion.

The structure is originally at rest, so we have the initial conditions

```math
X_0 = 0, \quad V_0 = \dot X_0 = 0.
```

In the Bogdanoff-Goldberg-Bernard model \cite{BogdanoffGoldbergBernard1961}, the excitation $\ddot M_t$ is made of a composition of oscillating signals $a_j t e^{-\delta_j t}\cos(\omega_j t + \theta_j)$ with random frequencies $\omega_j$, modulated by a linear attack rate $a_j t$ followed by an exponential decay $e^{-\delta_j t}$.

In order to simulate the start of the first shock-wave and the subsequent aftershocks, we modify this model sligthly to be a combination of such terms but at different incidence times. We also remove the attack rate from the excitation to obtain a rougher instantaneous, discontinuous excitation, which is connected with a square power attact rate for the displacement itself. Finally, for simulation purposes, we model directly the displacement $M_t$ and compute the associated excitation $\ddot M_t$, but in such a way that the ground-motion excitation follows essentially the proposed type of signal.

Thus, with this framework in mind, we model the ground displacement as a transport process composed of a series of time-translations of a square-power ``attack" front, with an exponentially decaying tail and an oscillating background wave:
```math
   M_t = \sum_{i=1}^k \gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)),
```
where $k\in \mathbb{N}$ is given, $(t-\tau_i)_+ = \max\{0, t - \tau_i\}$ is the positive part of the function, and the parameters $\gamma_i,$ $\tau_i,$ $\delta_i,$ and $\omega_i$ are all random variables, with $\tau_i$ being exponentially distributed, and $\gamma_i$, $\delta_i$, and $\omega_i$ being uniformly distributed, each with different support values, and all of them independent of each other. Each front (disregarding the oscillatory part) peaks at $t = \tau_i + 2/\delta_i,$ with peak value approximately $\gamma / 2\delta^2.$

The excitation itself becomes

```math
\begin{align*}
   \ddot M(t) = \; & \sum_{i=1}^k \gamma_i e^{-\delta_i (t - \tau_i)} \bigg\{ \left(2 H(t - \tau_i) - 4\delta_i(t - \tau_i)_+ + (\delta_i^2 - \omega_i^2) (t - \tau_i)_+^2 \right) \cos(\omega_i (t - \tau_i)) \\
      &  \hspace{2in} + \left( -4\omega_i (t - \tau_i)_+ + 2\delta_i\omega_i (t - \tau_i)_+^2 \right) \sin(\omega_i (t - \tau_i)) \bigg\}
\end{align*}
```
where $H = H(s)$ is the Heaviside function, where, for definiteness, we set $H(s) = 1,$ for $s \geq 1,$ and $H(s) = 0$, for $s < 0$.

More specifically, for the numerical simulations, we use $\zeta_0 = 0.6$ and $\omega_0 = 15\,\texttt{rad}/\texttt{s}$ as the structural parameters. We set $T = 2.0,$ as the final time. For the transport process, we set $k=8$ and define the random parameters as
```math
   \begin{align*}
       \tau_i & \sim \textrm{Exponential}(1/2) \\
       \gamma_i & \sim \textrm{Unif}(16, 32), \\
       \delta_i & \sim \textrm{Unif}(12, 16), \\
       \omega_i & \sim \textrm{Unif}(16\pi, 32\pi).
    \end{align*}
```

## Numerical approximation

### Setting up the problem

First we load the necessary packages:

````@example 08-earthquake
using Plots
using Measures
using Random
using LinearAlgebra
using Distributions
using RODEConvergence
````

Then we set up some variables, starting with the random seed, for reproducibility of the pseudo-random number sequence generator:

````@example 08-earthquake
rng = Xoshiro(123)
nothing # hide
````

We define the evolution law for the displacement $X_t$ driven by a noise $Y_t$. Since it is a system of equations, we use the in-place formulation. Notice the noise is a product of the background colored noise `y[1]` and the envelope noise `y[2]`. The parameters are hard-coded for simplicity.

````@example 08-earthquake
ζ₀ = 0.6
ω₀ = 15.0

params = (ζ₀, ω₀)

function f!(dx, t, x, y, p)
    ζ₀ = p[1]
    ω₀ = p[2]
    dx[1] = x[2]
    dx[2] = - 2 * ζ₀ * ω₀ * x[2] - ω₀ ^ 2 * x[1] - y
    return dx
end
nothing # hide
````

The time interval is defined by the following end points:

````@example 08-earthquake
t0, tf = 0.0, 2.0
nothing # hide
````

The structure is initially at rest, so the probability law is a vectorial product of two Dirac delta distributions, in $\mathbb{R}^2$:

````@example 08-earthquake
x0law = product_distribution(Dirac(0.0), Dirac(0.0))
````

As described above, we assume the ground motion is an additive combination of translated exponentially decaying wavefronts of the form
```math
  m_i(t) = \gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)),
```
where $(t - \tau_i)_+ = \max\{0, t - \tau_i\}$, i.e. it vanishes for $t \leq \tau_i$ and is simply $(t - \tau_i)$ for $t\geq \tau_i$. The associated noise is a combination of the second derivatives $\ddot m_i(t)$, which has jump discontinuities. Indeed, we have the ground velocities

```math
  \begin{align*}
  \dot m_i(t) = \; & 2\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & -\delta_i\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & -\omega_i\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
    = \; & \gamma_i e^{-\delta_i (t - \tau_i)} \left\{ \left(2(t - \tau_i)_+ - \delta_i (t - \tau_i)_+^2 \right) \cos(\omega_i (t - \tau_i)) - \omega_i (t - \tau_i)_+^2 \sin(\omega_i (t - \tau_i)) \right\}
  \end{align*}
```
and the ground accelerations

```math
  \begin{align*}
  \ddot m_i(t) = \; & 2\gamma_i H(t - \tau_i) e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & - 2\gamma_i \delta_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & - 2\gamma_i \omega_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
      & - 2\delta_i\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & +\delta_i^2\gamma_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
      & +\delta_i\gamma_i\omega_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
      & -2\omega_i\gamma_i (t - \tau_i)_+ e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
      & +\omega_i\gamma_i\delta_i (t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\sin(\omega_i (t - \tau_i)) \\
      & -\omega_i^2\gamma_i(t - \tau_i)_+^2 e^{-\delta_i (t - \tau_i)}\cos(\omega_i (t - \tau_i)) \\
    = \; & \gamma_i e^{-\delta_i (t - \tau_i)} \bigg\{ \left(2 H(t - \tau_i) - 4\delta_i(t - \tau_i)_+ + (\delta_i^2 - \omega_i^2) (t - \tau_i)_+^2 \right) \cos(\omega_i (t - \tau_i)) \\
      &  \hspace{2in} + \left( -4\omega_i (t - \tau_i)_+ + 2\delta_i\omega_i (t - \tau_i)_+^2 \right) \sin(\omega_i (t - \tau_i)) \bigg\}
  \end{align*}
```

where $H=H(s)$ is the Heaviside function, where, for definiteness, we set $H(s) = 1,$ for $s \geq 1,$ and $H(s) = 0$, for $s < 0$.

We implement these functions as

````@example 08-earthquake
function gm(t::T, τ::T, γ::T, α::T, ω::T) where {T}
    tshift = max(zero(T), t - τ)
    m = γ * tshift ^2 * exp( -α * tshift ) * cos( ω * tshift )
    return m
end

function dgm(t::T, τ::T, γ::T, δ::T, ω::T) where {T}
    tshift₊ = max(zero(T), t - τ)
    tshift₊² = tshift₊ ^ 2
    expδt₊ = exp( -δ * tshift₊ )
    sinωt₊, cosωt₊ = sincos(ω * tshift₊)
    ṁ = γ * expδt₊ * ( ( 2tshift₊ - δ * tshift₊² ) * cosωt₊ - ω * tshift₊²  * sinωt₊ )
    return ṁ
end

function ddgm(t::T, τ::T, γ::T, δ::T, ω::T) where {T}
    h = convert(T, t ≥ τ)
    tshift₊ = ( t - τ ) * h
    tshift₊² = tshift₊ ^ 2
    expδt₊ = exp( -δ * tshift₊ )
    sinωt₊, cosωt₊ = sincos(ω * tshift₊)
    m̈ = γ * expδt₊ * ( ( 2h + ( δ^2 - ω^2 ) * tshift₊² - 4δ * tshift₊) * cosωt₊ + ( -4ω * tshift₊ + 2δ * ω * tshift₊² ) * sinωt₊ )
    return m̈
end
nothing # hide
````

Let us visualize a cooked up example of ground motion with two components

````@example 08-earthquake
tt = range(0.0, 2.0, length=2^9)
dt = Float64(tt.step)

τ₁ = 0.2
γ₁ = 4.0
δ₁ = 12.0
ω₁ = 24π
τ₂ = 0.8
γ₂ = 2.0
δ₂ = 16.0
ω₂ = 32π

mt = gm.(tt, τ₁, γ₁, δ₁, ω₁) .+ gm.(tt, τ₂, γ₂, δ₂, ω₂)

dmt = dgm.(tt, τ₁, γ₁, δ₁, ω₁) .+ dgm.(tt, τ₂, γ₂, δ₂, ω₂)

ddmt = ddgm.(tt, τ₁, γ₁, δ₁, ω₁) .+ ddgm.(tt, τ₂, γ₂, δ₂, ω₂)

plt1 = plot(tt, mt, xlabel="\$t\$", ylabel="\$M_t\$")
plt2 = plot(tt, dmt, xlabel="\$t\$", ylabel="\$\\dot{M}_t\$")
plt3 = plot(tt, ddmt, xlabel="\$t\$", ylabel="\$\\ddot{M}_t\$")
plot(plt1, plt2, plt3, layout=(3, 1), legend=false)
````

We also numerically integrate the acceleration and the velocity and compare them against the velocity and the position, to make sure we implemented the derivatives correctly.

````@example 08-earthquake
mt2 = accumulate(+, dmt) * dt
dmt2 = accumulate(+, ddmt) * dt

maximum(abs, mt2 - mt)
````

Notice this is of the order of the time step

````@example 08-earthquake
dt
````

The absolute error for the derivative is high, though, because the frequency is relatively high

````@example 08-earthquake
maximum(abs, dmt2 - dmt)
````

The relative errors, though, are both modest, considering the low order integration method used here.

````@example 08-earthquake
maximum(abs, (mt2 - mt)) / maximum(abs, mt)
````

````@example 08-earthquake
maximum(abs, (dmt2 - dmt)) / maximum(abs, dmt)
````

Let us now illustrate a sample path with the laws used for the random variables and the associated transport process.

````@example 08-earthquake
τlaw = Exponential(tf/4)
γlaw = Uniform(16.0, 32.0)
δlaw = Uniform(12.0, 16.0)
ωlaw = Uniform(16π, 32π)

ylaw = product_distribution(τlaw, γlaw, δlaw, ωlaw)

nr = 8
g(t, r) = mapreduce(ri -> ddgm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(r))
noise = TransportProcess(t0, tf, ylaw, g, nr)
nothing # hide
````

````@example 08-earthquake
n = 2^12
tt = range(t0, tf, length=n)
yt = Vector{Float64}(undef, n)
nothing # hide
````

Sample ground acceleration

````@example 08-earthquake
rand!(rng, noise, yt)
nothing # hide
````

Associated ground motion $m_t$:

````@example 08-earthquake
mt = [mapreduce(ri -> gm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(noise.rv)) for t in range(t0, tf, length=length(yt))]
nothing # hide
````

Associated ground velocity

````@example 08-earthquake
dmt = [mapreduce(ri -> dgm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(noise.rv)) for t in range(t0, tf, length=length(yt))]
nothing # hide
````

Visualization

````@example 08-earthquake
plt1 = plot(tt, mt, xlabel="\$t\$", ylabel="\$M_t\$", label=nothing)
plt2 = plot(tt, dmt, xlabel="\$t\$", ylabel="\$\\dot{M}_t\$", label=nothing)
plt3 = plot(tt, yt, xlabel="\$t\$", ylabel="\$\\ddot{M}_t\$", label=nothing)
plt_ground = plot(plt1, plt2, plt3, layout = (3, 1))
````

Now we are ready to check the order of convergence. We set the target resolution, the convergence test resolutions, the sample convergence resolutions, and the number of sample trajectories for the Monte-Carlo approximation of the strong error.

````@example 08-earthquake
ntgt = 2^18
ns = 2 .^ (6:9)
````

The number of simulations for the Monte Carlo estimate is set to

````@example 08-earthquake
m = 100
nothing # hide
````

We add some information about the simulation, for the caption of the convergence figure.

````@example 08-earthquake
info = (
    equation = "mechanical structure model under ground-shaking random excitations",
    noise = "transport process noise",
    ic = "\$X_0 = \\mathbf{0}\$"
)
nothing # hide
````

We define the *target* solution as the Euler approximation, which is to be computed with the target number `ntgt` of mesh points, and which is also the one we want to estimate the rate of convergence, in the coarser meshes defined by `ns`.

````@example 08-earthquake
target = RandomEuler(length(x0law))
method = RandomEuler(length(x0law))
````

### Order of convergence

With all the parameters set up, we build the convergence suites for each noise:

````@example 08-earthquake
suite = ConvergenceSuite(t0, tf, x0law, f!, noise, params, target, method, ntgt, ns, m)
````

Then we are ready to compute the errors:

````@example 08-earthquake
@time result = solve(rng, suite)
nothing # hide
````

The computed strong error for each resolution in `ns` is stored in field `errors`, and raw LaTeX tables can be displayed for inclusion in the article:

````@example 08-earthquake
println(generate_error_table(result, suite, info)) # hide
nothing # hide
````

The calculated order of convergence is given by field `p`:

````@example 08-earthquake
println("Order of convergence `C Δtᵖ` with p = $(round(result.p, sigdigits=2)) and 95% confidence interval ($(round(result.pmin, sigdigits=3)), $(round(result.pmax, sigdigits=3)))")
nothing # hide
````

### Plots

We create plots with the rate of convergence with the help of a plot recipe for `ConvergenceResult`:

````@example 08-earthquake
plt_result = plot(result)
````

For the sake of illustration, we plot a sample of an approximation of a target solution, in each case:

````@example 08-earthquake
nsample = ns[[1, 2, 3]]
plt_sample = plot(suite, ns=nsample)
````

We also combine some plots into a single figure, to summarize the results.

````@example 08-earthquake
plt_combined = plot(plt_result, plt_sample, plt1, plt2, plt3, layout=@layout([[a; b] [c; d; e]]), size=(800, 480), title=["(a) seismic model" "(b) sample path" "(c) sample ground motion" "(d) sample ground velocity" "(e) sample ground acceleration"], legendfont=7, titlefont=10, bottom_margin=5mm, left_margin=5mm)
````

We finally draw an animation of the motion of the single-storey building driven by the transport-modulated noise.

````@example 08-earthquake
dt = ( tf - t0 ) / ( ntgt - 1 )
mt = [mapreduce(ri -> gm(t, ri[1], ri[2], ri[3], ri[4]), +,  eachcol(noise.rv)) for t in range(t0, tf, length=ntgt)]

@time anim = @animate for i in 1:div(ntgt, 2^9):div(ntgt, 1)
    ceiling = mt[i] + suite.xt[i, 1]
    height = 3.0
    halfwidth = 2.0
    aspectfactor = (4/6) * 4halfwidth / height
    plot([mt[i] - halfwidth; ceiling - halfwidth; ceiling + halfwidth; mt[i] + halfwidth], [0.0; height; height; 0.0], xlim = (-2halfwidth, 2halfwidth), ylim=(0.0, aspectfactor * height), xlabel="\$\\mathrm{displacement}\$", ylabel="\$\\textrm{height}\$", fill=true, title="Building at time t = $(round((i * dt), digits=3))", titlefont=10, legend=false)
end
nothing # hide
````

````@example 08-earthquake
gif(anim, fps = 30) # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

