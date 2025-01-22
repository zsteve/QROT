using Pkg
Pkg.activate(".")

using OptimalTransport
using Distances
using Distances: pairwise
using StatsBase
using DataFrames
using LinearAlgebra
using ArgParse
using ProgressMeter
using Random

include("../../src/util.jl")

s = ArgParseSettings()
@add_arg_table s begin
	"--N"
	arg_type = Int
	default = 5_000
    "--i"
    arg_type = Int
    default = 2
end

args = parse_args(s)
Random.seed!(0)

using NPZ
x0 = [0, 1, 0.5]
X_ = npzread("X_N_$(args["N"])_$(args["i"]).npy")
X = vcat(x0', X_)'

f(x) = dot([3, 5, 7], x.^2 /2)

y = map(f, eachcol(X))

C = pairwise(SqEuclidean(), X); C[diagind(C)] .= Inf
μ = fill(1/size(X, 2), size(X, 2));

C0 = 2.5; d = 2; N = args["N"]

alphas = [2, 1.75, 1.5, 1.25]
res = []
eps = []
K_eps = []
for alpha in alphas
	@info "alpha=$alpha"
	ε = C0*N^alpha 
	K_eps_N = ε^(2/(d+2)) * N^(-4/(d+2))
	# solve QOT
    solver = OptimalTransport.build_solver(μ, C, ε, OptimalTransport.SymmetricQuadraticOTNewton(δ = 1e-5); maxiter = 25)
	OptimalTransport.solve!(solver)
	W = OptimalTransport.plan(solver) * size(X, 2)
	L = I - W
	push!(res, (-2*(L * y)[1] / K_eps_N))
	push!(eps, ε)
	push!(K_eps, K_eps_N)
end 

df = DataFrame(:alpha => alphas, :res => res, :eps => eps, :K_eps => K_eps)
using CSV
CSV.write("output/res_N_$(args["N"])_$(args["i"]).csv", df)
