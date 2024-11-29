using Pkg
Pkg.activate("MNIST")

using Plots
using OptimalTransport
using MLDatasets
using LinearAlgebra
using Distances
using StatsBase
using PyCall
using Clustering
using Random
using ProgressMeter
using DataFrames
using CSV
using ArgParse
using MultivariateStats

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 100
	"--seed"
	arg_type = Int
	default = 42
    "--threads" 
    arg_type = Int
    default = 4
    "--outdir"
    arg_type = String
    default = "output"
end
args = parse_args(s)
Random.seed!(args["seed"])

skcluster = pyimport_conda("sklearn.cluster", "scikit-learn")
function clust(W, k)
    clust_op = skcluster.SpectralClustering(n_clusters = k, affinity = "precomputed", n_jobs = args["threads"]);
    clust_op.fit_predict(W); 
end

mnist = MNIST()
mnist = mnist[randperm(length(mnist))]
digit_inds = [findall(mnist.targets .== x)[1:args["N"]] for x in [1, 2, 7, 9]]
n_clust = 4
X = [reshape(mnist.features[:, :, x], 28*28, :) for x in digit_inds]
y = [mnist.targets[x] for x in digit_inds]

X = hcat(X...)
y = vcat(y...)

pca_op = fit(PCA, X; maxoutdim = 50);
X_pca = predict(pca_op, X)

eps_quad = 10 .^range(-2, 2, 25)
eps_ent = 10 .^range(-2, 2, 25)
ks = 5:5:125

@info "method: quad"
nmi_quad = @showprogress map(x -> mutualinfo(clust(kernel_ot_quad(X, x), n_clust), y), eps_quad)
nmi_quad_pca = @showprogress map(x -> mutualinfo(clust(kernel_ot_quad(X_pca, x), n_clust), y), eps_quad)
@info "method: quad_fat"
nmi_quad_fat = @showprogress map(x -> mutualinfo(clust(kernel_ot_quad(X, x, diag_inf = false), n_clust), y), eps_quad)
nmi_quad_fat_pca = @showprogress map(x -> mutualinfo(clust(kernel_ot_quad(X_pca, x, diag_inf = false), n_clust), y), eps_quad)
@info "method: ent"
nmi_ent = @showprogress map(x -> mutualinfo(clust(kernel_ot_ent(X, x), n_clust), y), eps_ent)
nmi_ent_pca = @showprogress map(x -> mutualinfo(clust(kernel_ot_ent(X_pca, x), n_clust), y), eps_ent)
@info "method: ent_fat"
nmi_ent_fat = @showprogress map(x -> mutualinfo(clust(kernel_ot_ent(X, x, diag_inf = false), n_clust), y), eps_ent)
nmi_ent_fat_pca = @showprogress map(x -> mutualinfo(clust(kernel_ot_ent(X_pca, x, diag_inf = false), n_clust), y), eps_ent)
@info "method: knn"
nmi_knn = @showprogress map(x -> mutualinfo(clust(knn_adj(X, x), n_clust), y), ks)
nmi_knn_pca = @showprogress map(x -> mutualinfo(clust(knn_adj(X_pca, x), n_clust), y), ks)

using DataStructures
df = DataFrame(OrderedDict("eps_quad" => eps_quad, "quad" => nmi_quad, "quad_pca" => nmi_quad_pca, "quad_fat" => nmi_quad_fat, "quad_fat_pca" => nmi_quad_fat_pca, 
                           "eps_ent" => eps_ent, "ent" => nmi_ent, "ent_pca" => nmi_ent_pca, "ent_fat" => nmi_ent_fat, "ent_fat_pca" => nmi_ent_fat_pca, 
                           "k" => ks, "knn" => nmi_knn, "knn_pca" => nmi_knn_pca))
CSV.write(joinpath(args["outdir"], "mnist_results_N_$(args["N"])_seed_$(args["seed"]).csv"), df)
