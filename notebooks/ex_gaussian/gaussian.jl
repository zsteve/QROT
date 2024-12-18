using Pkg
Pkg.activate(".")

using OptimalTransport
using Plots
using LinearAlgebra
using Distributions
using Random
using NearestNeighbors
using SparseArrays
using MultivariateStats
using SimpleWeightedGraphs
using GraphPlot
using PyCall
using Clustering
using Glob
using Serialization
using StatsPlots
using LaTeXStrings
using ProgressMeter
using ArgParse

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 100
    "--d" 
    arg_type = Int 
    default = 250
    "--pca"
    arg_type = Int
    default = -1
	"--seed"
	arg_type = Int
	default = 1
    "--threads" 
    arg_type = Int
    default = 4
    "--outdir"
    arg_type = String
    default = "output"
    "--save_mats"
    arg_type = Bool
    default = false
    "--eps_quad_idx"
    arg_type = Int 
    default = -1
    "--eps_ent_idx"
    arg_type = Int 
    default = -1
    "--k_idx"
    arg_type = Int 
    default = -1
end
args = parse_args(s)
Random.seed!(args["seed"])

n_clust = 3
σs = [0.3, 0.6, 1.0]

# generate data
y = vcat([fill(i, args["N"]) for i = 1:3]...)
μs = hcat([vcat([sin(t), cos(t)], zeros(args["d"]-2)) for t in [0, 2π/3, -2π/3]]...)*1.5
X = hcat([Diagonal(σ*ones(args["d"]))*randn(args["d"], args["N"]) .+ μ for (σ, μ) in zip(σs, eachcol(μs))]...);

if args["pca"] > 0
    pca_op = fit(PCA, X; maxoutdim = args["pca"])
    X = predict(pca_op, X)
end

# also get subspace angles
A = hcat([y .== i for i in unique(y)]...) * 1.0
sp_la = pyimport_conda("scipy.linalg", "scipy")

eps_quad = 10 .^range(-2, 2, 25)
if args["eps_quad_idx"] > 0
    eps_quad = [eps_quad[args["eps_quad_idx"]], ]
end
eps_ent = 10 .^range(-2, 2, 25)
if args["eps_ent_idx"] > 0
    eps_ent = [eps_ent[args["eps_ent_idx"]], ]
end
ks = collect(5:5:125)
if args["k_idx"] > 0
    ks = [ks[args["k_idx"]], ]
end

function get_score(K)
    get_eigvecs(K, k) = eigen(Hermitian(I-K)).vectors[:, 1:k]
    nmi = mutualinfo(clust(K, n_clust), y)
    theta = mean(sp_la.subspace_angles(get_eigvecs(K, 2*n_clust), A))
    nmi, theta
end

skcluster = pyimport_conda("sklearn.cluster", "scikit-learn")
function clust(W, k)
    clust_op = skcluster.SpectralClustering(n_clusters = k, affinity = "precomputed", n_jobs = args["threads"]);
    clust_op.fit_predict(W); 
end

unzip = x -> zip(x...)
@info "method: quad"
nmi_quad, theta_quad = map(x -> get_score(kernel_ot_quad(X, x)), eps_quad) |> unzip
@info "method: quad_fat"
nmi_quad_fat, theta_quad_fat = map(x -> get_score(kernel_ot_quad(X, x, diag_inf = false)), eps_quad) |> unzip
@info "method: ent"
nmi_ent, theta_ent = map(x -> get_score(kernel_ot_ent(X, x)), eps_ent) |> unzip
@info "method: ent_fat"
nmi_ent_fat, theta_ent_fat = map(x -> get_score(kernel_ot_ent(X, x, diag_inf = false)), eps_ent) |> unzip
@info "method: knn"
nmi_knn, theta_knn = map(x -> get_score(Array(norm_kernel(symm(knn_adj(X, x)), :sym))), ks) |> unzip

using DataStructures
using DataFrames
using CSV
df = DataFrame(OrderedDict("eps_quad" => eps_quad,
                           "nmi_quad" => collect(nmi_quad),
                           "theta_quad" => collect(theta_quad),
                           "nmi_quad_fat" => collect(nmi_quad_fat),
                           "theta_quad_fat" => collect(theta_quad_fat),
                           "eps_ent" => eps_ent,
                           "nmi_ent" => collect(nmi_ent),
                           "theta_ent" => collect(theta_ent),
                           "nmi_ent_fat" => collect(nmi_ent_fat),
                           "theta_ent_fat" => collect(theta_ent_fat),
                           "k" => ks,
                           "nmi_knn" => collect(nmi_knn),
                           "theta_knn" => collect(theta_knn)))
CSV.write(joinpath(args["outdir"], "gaussian_results_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).csv"), df)

# write data
using NPZ
npzwrite(joinpath(args["outdir"], "X_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), X)
npzwrite(joinpath(args["outdir"], "y_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), y)

