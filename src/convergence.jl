using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/zsteve/OptimalTransport.jl", rev="symmetric_quad")

using OptimalTransport
using Distances
using Distances: pairwise
using Plots
using StatsBase
using GLM
using DataFrames
using LinearAlgebra

d = 4 # 2, 3, 4

# d-sphere
function get_plot(N)
    # θs = range(-π, π; length = N)
    # X = mapreduce(x -> [cos(x), sin(x)], hcat, θs)
    X = randn(d, N)
    X ./= reshape(map(norm, eachcol(X)), 1, :)
    @info size(X)
    C = pairwise(SqEuclidean(), X)
    # C /= mean(C)
    εvals = 10 .^range(5, -3; length = 100)
    μ = fill(1/size(X, 2), size(X, 2))
    function dualpotentials(ε)
        # @info ε
        solver = OptimalTransport.build_solver(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(δ = 1e-5); maxiter = 100)
        OptimalTransport.solve!(solver)
        solver.cache.u
    end
    uvals = map(x -> dualpotentials(x), εvals)
    umeans = map(mean, uvals)
    # df1 = DataFrame(logu = log10.(umeans[εvals .< 0.05]), logeps = log10.(εvals[εvals .< 0.05]))
    # fit1 = lm(@formula(logu ~ logeps), df1)
    df2 = DataFrame(logu = log10.(umeans[εvals .> 25]), logeps = log10.(εvals[εvals .> 25]))
    fit2 = lm(@formula(logu ~ logeps), df2)
    plt=scatter(log10.(εvals), log10.(umeans); markersize = 1, alpha = 0.25, label = "data", legend = :topleft, title = "N = $N")
    # plot!(plt, df1.logeps, predict(fit1); label = "α = $(coef(fit1)[2])")
    plot!(plt, df2.logeps, predict(fit2); label = "α = $(coef(fit2)[2])")
    # hline!(plt, [log10.(Cthresh), ]; label = "u_thresh")
    plt
end

Ns = [1_000, ]

plots_all = map(x -> get_plot(x), Ns)

plot(plots_all[1]; size = (500, 500), title = "d = $(d-1), 2/(2+d) = $(round(2/(2+d-1); digits = 3))", xlabel = "ε", ylabel = "log(u)")
savefig("isotropic_d_$d.pdf")

ENV["R_HOME"] = "/usr/lib64/R"
using RCall
@rlibrary TDA
using LaTeXStrings

x0 = [0, 0.5, 0]
d = 2
f1(x) = dot([3, 0, 7], x.^2 /2)
r, R = 0.5, 1

function sample(f; N = 250, C0 = 10.0, α = 1.5)
    X = vcat(x0', convert(Array, torusUnif(N, r, R)))'
    y = map(f, eachcol(X))
    C = pairwise(SqEuclidean(), X)
    μ = fill(1/size(X, 2), size(X, 2));
    ε = C0*N^α # pick exponent 3/2
    K_eps_N = ε^(2/(d+2)) * N^(-4/(d+2))
    # solve QOT
    solver = OptimalTransport.build_solver(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(δ = 1e-5); maxiter = 25)
    OptimalTransport.solve!(solver)
    W = OptimalTransport.plan(solver)
    W .*= size(X, 2) 
    (W * y)[1] / K_eps_N
end

using ProgressMeter
M = 10
# α = 2.0 # corresponds to a constant value of \eps in the continuous case
α = 1.05
Ns = [100, 250, 500, 1000, 2500, 5000]
L_samples = @showprogress [[sample(f1; N = k, α = α, C0 = 1) for _ in 1:M] for k in Ns]
# what is the effective epsilon?
eps_eff = [C0*k^α / k^2 for k in Ns]

using StatsPlots
using LaTeXStrings
plt=boxplot(hcat(L_samples...), xticks = (1:length(Ns), Ns), color = :lightgrey, xlabel = "N", ylabel = L"K_{ε, N}^{-1} (W y)(x_0)", legend = nothing, title = "α = $(α)", linecolor = :black, ylim = (1, 3.5))
savefig("qot_torus_alpha_$(α).pdf")
plt
