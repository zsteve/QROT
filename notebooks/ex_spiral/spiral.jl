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
using QuadGK
using ForwardDiff
using Roots
using NNlib
using NPZ

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 1000
    "--d" 
    arg_type = Int 
    default = 100
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
    default = true
    "--eps_quad_idx"
    arg_type = Int 
    default = 1
    "--eps_ent_idx"
    arg_type = Int 
    default = 1
    "--k_idx"
    arg_type = Int 
    default = 1
    "--eps_epanech_idx"
    arg_type = Int 
    default = 1
    "--eps_gauss_idx"
    arg_type = Int 
    default = 1
    "--eps_gauss_l2_idx"
    arg_type = Int 
    default = 1
    "--k_magic_idx"
    arg_type = Int 
    default = 1
end
args = parse_args(s)
Random.seed!(args["seed"])

### Generate spiral data
d = args["d"]
N = args["N"]

f(t) = [cos(t)*(0.5cos(6t)+1) sin(t)*(0.4cos(6t)+1) 0.4sin(6t)]
arclength(t::Real) = quadgk(t -> norm(ForwardDiff.derivative(f, t)), 0, t, rtol = 1e-8)[1]
L_tot = arclength(2π)
t_range_scaled = range(0, 1; length = N+1)[1:end-1]
θ_range = map(x -> find_zero(t -> arclength(t)/L_tot - x, (0., 2π)), t_range_scaled)
R = qr(randn(d, d)).Q[:, 1:3]
X = vcat(f.(θ_range)...)' 
ρ(θ) = 0.05 + 1.25*(1 + cos(6θ))/2
η = hcat([x/norm(x)*ρ(θ) for (θ, x) in zip(θ_range, eachcol(randn(d, size(X, 2))))]...);
X_orig_lowdim = copy(X)
X_orig = R * X
X = X_orig + η;

arclength(X) = cumsum([norm(x-y) for (x, y) in zip(eachcol(X[:, 1:end-1]), eachcol(X[:, 2:end]))]);
l = [quadgk(t -> norm(ForwardDiff.derivative(f, t)), 0, tfinal, rtol = 1e-8)[1] for tfinal in θ_range]
extrema(l[2:end].-l[1:end-1]) # check we are sampling w.r.t. arclength

if args["pca"] > 0
    pca_op = fit(PCA, X; maxoutdim = args["pca"])
    X = predict(pca_op, X)
end

sp_la = pyimport_conda("scipy.linalg", "scipy")

# Compute reference eigenvector embedding
# function rw_norm_eigvecs(W)
#     # Compute row-normalized eigenvectors for symmetric W
#     D = Diagonal(vec(sum(W; dims = 2)))
#     D^(-1/2) * eigen(Hermitian(W)).vectors
# end
K_ref = norm_kernel(form_kernel(X_orig_lowdim, 0.05; k = 3), :sym)
U_ref = eigen(K_ref).vectors[:, end:-1:1];

eps_quad = 10 .^range(-2, 2, 25)
# if args["eps_quad_idx"] > 0
#     eps_quad = [eps_quad[args["eps_quad_idx"]], ]
# end
eps_ent = 10 .^range(-2, 2, 25)
# if args["eps_ent_idx"] > 0
#     eps_ent = [eps_ent[args["eps_ent_idx"]], ]
# end
ks = collect(5:5:125)
# if args["k_idx"] > 0
#     ks = [ks[args["k_idx"]], ]
# end
eps_epanech = (eps_quad) .^ (2 / (2 + 1)) 

function get_theta(K; ndim = 10)
    get_eigvecs(K, k) = eigen(I-K).vectors[:, 1:k]
    theta = mean(sp_la.subspace_angles(get_eigvecs(K, ndim), U_ref[:, 1:ndim]))
    theta
end

function save(K, fname; cond = false)
	if cond 
		npzwrite(joinpath(args["outdir"], fname), K)
	end
	K
end

suffix = "pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"])"

unzip = x -> zip(x...)
@info "method: quad"
theta_quad = @showprogress map(x -> get_theta(save(Hermitian(kernel_ot_quad(X, x)), "K_quad_eps_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == eps_quad[args["eps_quad_idx"]]))), eps_quad)
@info "method: quad_fat"
theta_quad_fat = @showprogress map(x -> get_theta(Hermitian(kernel_ot_quad(X, x, diag_inf = false))), eps_quad)
@info "method: ent"
theta_ent = @showprogress map(x -> get_theta(save(Hermitian(kernel_ot_ent(X, x)), "K_ent_eps_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == eps_ent[args["eps_ent_idx"]]))), eps_ent)
@info "method: ent_fat"
theta_ent_fat = @showprogress map(x -> get_theta(Hermitian(kernel_ot_ent(X, x, diag_inf = false))), eps_ent)
@info "method: knn"
theta_knn = @showprogress map(x -> get_theta(save(Hermitian(Array(norm_kernel(symm(knn_adj(X, x)), :sym))), "K_knn_k_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == ks[args["k_idx"]]))), ks)
@info "method: epanechnikov"
theta_epanech = @showprogress map(x -> get_theta(save(Hermitian(norm_kernel(kernel_epanech(X, x), :sym)), "K_epanech_eps_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == eps_epanech[args["eps_epanech_idx"]]))), eps_epanech)
@info "method: gaussian"
theta_gauss = @showprogress map(x -> get_theta(save(Hermitian(norm_kernel(form_kernel(X, x; k = Inf), :sym)), "K_gauss_eps_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == eps_ent[args["eps_gauss_idx"]]))), eps_ent)
@info "method: gaussian_l2"
theta_gauss_l2 = @showprogress map(x -> get_theta(save(Hermitian(kernel_gaussian_l2_proj(X, x)), "K_gauss_l2_eps_$(x)_$(suffix).npy"; cond = args["save_mats"] && (x == eps_ent[args["eps_gauss_l2_idx"]]))), eps_ent)

using PyCall
magic = pyimport("magic")
magic_ops = [magic.MAGIC() for _ in ks]
[op.set_params(t = 5, knn = k) for (k, op) in zip(ks, magic_ops)]
[op.fit_transform(X', genes = "all_genes")' for op in magic_ops]
theta_magic = @showprogress map(((k, op), ) -> get_theta(save(op.diff_op.todense(), "K_magic_k_$(k)_$(suffix).npy"; cond = args["save_mats"] && (k == ks[args["k_magic_idx"]]))), zip(ks, magic_ops))

using DataStructures
using DataFrames
using CSV
df = DataFrame(OrderedDict("eps_quad" => eps_quad,
                           "theta_quad" => collect(theta_quad),
                           "theta_quad_fat" => collect(theta_quad_fat),
                           "eps_ent" => eps_ent,
                           "theta_ent" => collect(theta_ent),
                           "theta_ent_fat" => collect(theta_ent_fat),
						   "theta_gauss" => collect(theta_gauss), 
						   "theta_gauss_l2" => collect(theta_gauss_l2), 
                           "k" => ks,
                           "theta_knn" => collect(theta_knn), 
                           "theta_magic" => collect(theta_magic), 
						   "eps_epanech" => collect(eps_epanech), 
						   "theta_epanech" => collect(theta_epanech)))
CSV.write(joinpath(args["outdir"], "spiral_results_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).csv"), df)

# write data
npzwrite(joinpath(args["outdir"], "X_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), X)
npzwrite(joinpath(args["outdir"], "X_orig_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), X_orig)
npzwrite(joinpath(args["outdir"], "X_orig_lowdim_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), X_orig_lowdim)
npzwrite(joinpath(args["outdir"], "K_ref_pca_$(args["pca"])_N_$(args["N"])_d_$(args["d"])_seed_$(args["seed"]).npy"), K_ref)

