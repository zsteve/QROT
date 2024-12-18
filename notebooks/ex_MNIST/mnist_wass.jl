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
using CUDA
# _cu = x -> x 
_cu = cu

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 100
	"--seed"
	arg_type = Int
	default = 1
    "--threads" 
    arg_type = Int
    default = 4
    "--outdir"
    arg_type = String
    default = "output_wass"
    "--save_mats"
    arg_type = Bool
    default = false
    "--w" 
    arg_type = Int
    default = 2
end
args = parse_args(s)
Random.seed!(args["seed"])

skcluster = pyimport_conda("sklearn.cluster", "scikit-learn")
function clust(W, k)
    clust_op = skcluster.SpectralClustering(n_clusters = k, affinity = "precomputed", n_jobs = args["threads"]);
    clust_op.fit_predict(W); 
end

using Images
using ImageTransformations
using CoordinateTransformations

l = 28
w = args["w"]

function enlarge(image::Matrix{<:AbstractFloat})
    padarray(image, Fill(0, (w, w)))
end

function shift(image::Matrix{<:AbstractFloat})
    max_shift = w
    shift_x = rand(-max_shift:max_shift)
    shift_y = rand(-max_shift:max_shift)
	t = Translation(shift_x, shift_y)
	warp(image, t, indices_spatial(image), fillvalue = 0);
end

mnist = MNIST()
mnist = mnist[randperm(length(mnist))]
digit_inds = [findall(mnist.targets .== x)[1:args["N"]] for x in [1, 2, 7, 9]]
n_clust = 4
X_orig = [mnist.features[:, :, x] for x in digit_inds]
y = [mnist.targets[x] for x in digit_inds]
X = [hcat([vec(shift(collect(enlarge(Array(y))))) for y in eachslice(x; dims = 3)]...) for x in X_orig]

X = hcat(X...)
y = vcat(y...)
X_orig = hcat([hcat([vec(collect(enlarge(Array(y)))) for y in eachslice(x; dims = 3)]...) for x in X_orig]...)

C = pairwise(SqEuclidean(), hcat(vec(([[x, y]*1.0 for (x, y) in Iterators.product(1:(l+2w), 1:(l + 2w))]))...))
C /= mean(C)

# Calculate true distances (using GPU)
X_norm = X ./ sum(X; dims = 1)
ε = 0.05
Sxx = [sinkhorn2(x, x, C, ε) for x in eachcol(X_norm)]
C_cu = _cu(C)
X_norm_cu = _cu(X_norm)
Sxy = @showprogress [Array(sinkhorn2(x, X_norm_cu, C_cu, ε; maxiter = 1_000)) for x in eachcol(X_norm_cu)]
S = hcat(Sxy...) .- (Sxx .+ Sxx')/2
S = (S + S')/2

h = median(S)
K = exp.(-S/h)

# Use Nystrom approximation
ns = 5:5:100
Ks_nys = []
lst = randperm(size(X, 2))

using NNlib
for n_landmark in ns
    landmark_idx = sort(lst[1:n_landmark])
    Sxy_nys = S[landmark_idx, :]
    Sxx_nys = Sxy_nys[:, landmark_idx]
    Kxy_nys = exp.(-Sxy_nys / h)
    Kxx_nys = exp.(-Sxx_nys / h)
    K_nys = Kxy_nys' * pinv(Kxx_nys, rtol = 1e-6) * Kxy_nys
    push!(Ks_nys, K_nys)
end 

if args["save_mats"]
    for (i, n) in enumerate(ns)
        CSV.write(joinpath(args["outdir"], "K_nys_$(n)_w_$(args["w"]).csv"), DataFrame(Ks_nys[i], :auto))
    end
end

eps_quad = exp.(range(log(0.1), log(100.0), 25))
nmi_quad = []
# nmi_nys_quad = []
@showprogress for ε_l2 in eps_quad
    K_proj_l2 = kernel_l2_proj(K, ε_l2, norm_cost = false)
    push!(nmi_quad, mutualinfo(clust(K_proj_l2, n_clust), y))
    Ks_nys_proj_l2 = [kernel_l2_proj(x, ε_l2, norm_cost = false) for x in Ks_nys]
    # push!(nmi_nys_quad, [mutualinfo(clust(x, n_clust), y) for x in Ks_nys_proj_l2])
    if args["save_mats"]
        CSV.write(joinpath(args["outdir"], "K_proj_l2_eps_$(ε_l2)_w_$(args["w"]).csv"), DataFrame(K_proj_l2, :auto))
        for (i, n) in enumerate(ns)
            CSV.write(joinpath(args["outdir"], "K_nys_$(n)_proj_l2_eps_$(ε_l2)_w_$(args["w"]).csv"), DataFrame(Ks_nys_proj_l2[i], :auto))
        end
    end
end

eps_ent = exp.(range(log(0.01), log(5.0), 25))
nmi_ent = []
# nmi_nys_ent = []
@showprogress for ε_ent in eps_ent
    K_proj_ent = kernel_ent_proj(K, ε_ent, norm_cost = false)
    push!(nmi_ent, mutualinfo(clust(K_proj_ent, n_clust), y))
    Ks_nys_proj_ent = [kernel_ent_proj(x, ε_ent, norm_cost = false) for x in Ks_nys]
    # push!(nmi_nys_ent, [mutualinfo(clust(x, n_clust), y) for x in Ks_nys_proj_ent])
    if args["save_mats"]
        CSV.write(joinpath(args["outdir"], "K_proj_ent_eps_$(ε_ent)_w_$(args["w"]).csv"), DataFrame(K_proj_ent, :auto))
        for (i, n) in enumerate(ns)
            CSV.write(joinpath(args["outdir"], "K_nys_$(n)_proj_ent_eps_$(ε_ent)_w_$(args["w"]).csv"), DataFrame(Ks_nys_proj_ent[i], :auto))
        end
    end
end

if args["save_mats"] CSV.write(joinpath(args["outdir"], "K_w_$(args["w"]).csv"), DataFrame(K, :auto)) end

ks = 5:5:125
pca_op = fit(PCA, X; maxoutdim = 50);
X_pca = predict(pca_op, X);
K_knn = [norm_kernel(symm(knn_adj(X, x)), :sym) for x in ks]
if args["save_mats"]
    for (k, K) in zip(ks, K_knn)
        CSV.write(joinpath(args["outdir"], "K_knn_$(k)_w_$(args["w"]).csv"), DataFrame(K, :auto))
    end
end
nmi_knn = @showprogress map(x -> mutualinfo(clust(x, n_clust), y), K_knn)
# nmi_knn_pca = @showprogress map(x -> mutualinfo(clust(knn_adj(X_pca, x), n_clust), y), ks)

df = DataFrame(:eps_quad => eps_quad, :quad => nmi_quad, :eps_ent => eps_ent, :ent => nmi_ent)
df[!, :knn] = nmi_knn
# df[!, :knn_pca] = nmi_knn_pca
CSV.write(joinpath(args["outdir"], "mnist_wass_results_N_$(args["N"])_w_$(args["w"])_seed_$(args["seed"]).csv"), df)

if args["save_mats"]
    CSV.write(joinpath(args["outdir"], "labels_seed_$(args["seed"])_w_$(args["w"]).csv"), DataFrame(:label => y))
end
