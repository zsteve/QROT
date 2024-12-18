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
	default = 500
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
# digit_inds = [findall(mnist.targets .== x)[1:args["N"]] for x in [2, 3, 7, 9]]
n_clust = 4
X = [reshape(mnist.features[:, :, x], 28*28, :) for x in digit_inds]
y = [mnist.targets[x] for x in digit_inds]

X = hcat(X...)
y = vcat(y...)

ds_pca = Int.(round.(collect(range(5, size(X, 1), 25))))
pca_ops = [fit(PCA, X; maxoutdim = d) for d in ds_pca];
X_pcas = [predict(pca_op, X) for pca_op in pca_ops]

# also get subspace angles
A = hcat([y .== i for i in unique(y)]...) * 1.0
sp_la = pyimport_conda("scipy.linalg", "scipy")

eps_quad = 10 .^range(-2, 2, 25)
eps_ent = 10 .^range(-2, 2, 25)
ks = 5:5:125

eps_quad_ = eps_quad[13]
eps_ent_ = eps_ent[5]
k_ = ks[2]

function get_score(K)
    get_eigvecs(K, k) = eigen(Hermitian(I-K)).vectors[:, 1:k]
    nmi = mutualinfo(clust(K, n_clust), y)
    theta = mean(sp_la.subspace_angles(get_eigvecs(K, 2*n_clust), A))
    nmi, theta
end


unzip = x -> zip(x...)
@info "method: quad"
nmi_quad, theta_quad = map(x -> get_score(kernel_ot_quad(X, x)), eps_quad) |> unzip
nmi_quad_pca, theta_quad_pca = map(x -> get_score(kernel_ot_quad(x, eps_quad_)), X_pcas) |> unzip
@info "method: quad_fat"
nmi_quad_fat, theta_quad_fat = map(x -> get_score(kernel_ot_quad(X, x, diag_inf = false)), eps_quad) |> unzip
nmi_quad_fat_pca, theta_quad_fat_pca = map(x -> get_score(kernel_ot_quad(x, eps_quad_, diag_inf = false)), X_pcas) |> unzip
@info "method: ent"
nmi_ent, theta_ent = map(x -> get_score(kernel_ot_ent(X, x)), eps_ent) |> unzip
nmi_ent_pca, theta_ent_pca = map(x -> get_score(kernel_ot_ent(x, eps_ent_)), X_pcas) |> unzip
@info "method: ent_fat"
nmi_ent_fat, theta_ent_fat = map(x -> get_score(kernel_ot_ent(X, x, diag_inf = false)), eps_ent) |> unzip
nmi_ent_fat_pca, theta_ent_fat_pca = map(x -> get_score(kernel_ot_ent(x, eps_ent_, diag_inf = false)), X_pcas) |> unzip
@info "method: knn"
nmi_knn, theta_knn = map(x -> get_score(Array(norm_kernel(symm(knn_adj(X, x)), :sym))), ks) |> unzip
nmi_knn_pca, theta_knn_pca = map(x -> get_score(Array(norm_kernel(symm(knn_adj(x, k_)), :sym))), X_pcas) |> unzip

using DataStructures
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
CSV.write(joinpath(args["outdir"], "mnist_results_N_$(args["N"])_seed_$(args["seed"]).csv"), df)

df_pca = DataFrame(OrderedDict("d" => ds_pca,
                               "nmi_quad" => collect(nmi_quad_pca),
                               "theta_quad" => collect(theta_quad_pca),
                               "nmi_ent" => collect(nmi_ent_pca),
                               "theta_ent" => collect(theta_ent_pca),
                               "nmi_knn" => collect(nmi_knn_pca),
                               "theta_knn" => collect(theta_knn_pca)))
CSV.write(joinpath(args["outdir"], "mnist_results_N_$(args["N"])_seed_$(args["seed"])_pca.csv"), df_pca)

# write some examples
using NPZ
npzwrite(joinpath(args["outdir"], "X_N_$(args["N"])_seed_$(args["seed"]).npy"), X)
npzwrite(joinpath(args["outdir"], "y_N_$(args["N"])_seed_$(args["seed"]).npy"), y)
npzwrite(joinpath(args["outdir"], "X_pca_N_$(args["N"])_seed_$(args["seed"]).npy"), X_pcas[1])

K_quad_ = kernel_ot_quad(X, eps_quad_)
K_ent_ = kernel_ot_ent(X, eps_ent_)

clust_quad = clust(K_quad_, n_clust)
clust_ent = clust(K_ent_, n_clust)

npzwrite(joinpath(args["outdir"], "K_quad_N_$(args["N"])_seed_$(args["seed"]).npy"), K_quad_)
npzwrite(joinpath(args["outdir"], "clust_quad_N_$(args["N"])_seed_$(args["seed"]).npy"), clust_quad)
npzwrite(joinpath(args["outdir"], "K_ent_N_$(args["N"])_seed_$(args["seed"]).npy"), K_ent_)
npzwrite(joinpath(args["outdir"], "clust_ent_N_$(args["N"])_seed_$(args["seed"]).npy"), clust_ent)



