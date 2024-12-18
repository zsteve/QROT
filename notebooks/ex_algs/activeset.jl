using Pkg
Pkg.activate(".")
using OptimalTransport
using LinearAlgebra
using SparseArrays
using Distances
using StatsBase
using NNlib
using BenchmarkTools
using NearestNeighbors
using RandomMatrix
using ParallelNeighbors
using ProgressMeter
using ArgParse
using Random
using DataFrames
using CSV
using LinearAlgebra

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 5000
    "--d" 
    arg_type = Int 
    default = 100
	"--eps"
	arg_type = Float64
	default = 1.0
	"--k" 
	arg_type = Int 
	default = 50
	"--threads" 
	arg_type = Int 
	default = 4
	"--seed"
	arg_type = Int
	default = 1
    "--outdir"
    arg_type = String
    default = "output"
    "--save_mats"
    arg_type = Bool
    default = false
    "--skipdense"
    arg_type = Bool
    default = false
end
args = parse_args(s)
Random.seed!(args["seed"])

BLAS.set_num_threads(args["threads"])

function gen_data(N, d)
    μ_spt = randn(N, d)
    X = μ_spt
    μ = ones(N)
    C = sum(X.^2 ; dims = 2)/2 .+ sum(X.^2 ; dims = 2)'/2 - X * X'
    Cmean = mean(C)
    C[diagind(C)] .= Inf
    return X, μ, C / Cmean
end

function knn_adj_parallel(X, k)
    indices, _ = ParallelNeighbors.knn(X, X, k);
    A = zeros(size(X, 2), size(X, 2));
    @inbounds for i = 1:size(A, 1)
        A[i, i] = 1; @inbounds for j in indices[i] A[i, j] = 1 end
    end
    A = sparse(A)
end

function construct_support(X, k)
    # S = knn_adj(X', k);
    S = knn_adj_parallel(X', k);
    S += sign.(mean([randPermutation(size(X, 1)) for _ in 1:k]));
    S += S';
    S = sign.(S); 
    S[diagind(S)] .= 0;
    dropzeros!(S);
    return S
end

ε = args["eps"]
k = args["k"]
N = args["N"]
d = args["d"]

X, μ, C = gen_data(N, d);
ENV["JULIA_DEBUG"] = ""; 
if args["skipdense"]
    elapsed_dense = [missing for _ in 1:11]
else
    elapsed_dense = @showprogress [(@timed sparse(quadreg(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100)))[[:time, :bytes]] for _ in 1:11]
end
elapsed_sparse = @showprogress [(@timed quadreg(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewtonAS(construct_support(X, k)); maxiter = 100))[[:time, :bytes]] for _ in 1:11]

df = DataFrame(:dense_time => [x.time for x in elapsed_dense], :sparse_time => [x.time for x in elapsed_sparse], 
               :dense_bytes => [x.bytes for x in elapsed_dense], :sparse_bytes => [x.bytes for x in elapsed_sparse])

CSV.write(joinpath(args["outdir"], "runtimes_eps_$(eps)_k_$(k)_N_$(N)_d_$(d)_seed_$(args["seed"]).csv"), df)

