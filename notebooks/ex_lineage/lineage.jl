using Pkg
Pkg.activate(".")

using OptimalTransport
using CSV
using DataFrames
using LinearAlgebra
using StatsBase
using ProgressMeter
using Random
using NNlib

Random.seed!(0)

X_df = CSV.read("clone_hist_coarse_norm.csv", DataFrame)[!, 2:end] |> Array
X_df .+= 1e-6
C = CSV.read("C.csv", DataFrame)[!, 2:end] |> Array
C /= mean(C)

# Calculate true distances 
ε = 0.05
Sxx = [sinkhorn2(x, x, C, ε) for x in eachcol(X_df)]
Sxy = @showprogress [sinkhorn2(x, X_df, C, ε) for x in eachcol(X_df)]
S = hcat(Sxy...) .- (Sxx .+ Sxx')/2
CSV.write("S.csv", DataFrame(S, :auto))

h = median(S)

# Use Nystrom approximation
ns = 5:5:100
Ks_nys = []
lst = randperm(size(X_df, 2))

for n_landmark in ns
    landmark_idx = sort(lst[1:n_landmark])
    Sxy_nys = S[landmark_idx, :] # hcat([sinkhorn2(X_df[:, i], X_df, C, ε) for i in landmark_idx]...)' .- (Sxx[landmark_idx] .+ Sxx')/2
    Sxx_nys = Sxy_nys[:, landmark_idx]
    Kxy_nys = exp.(-Sxy_nys / h)
    Kxx_nys = exp.(-Sxx_nys / h)
    K_nys = Kxy_nys' * pinv(Kxx_nys; rtol = 1e-6) * Kxy_nys
    push!(Ks_nys, K_nys)
    CSV.write("K_nys_$(n_landmark).csv", DataFrame(K_nys, :auto))
end 

K = exp.(-S/h)
CSV.write("K.csv", DataFrame(K, :auto))

include("../../src/util.jl")
for ε_l2 in exp.(range(log(0.5), log(50.0), 25)) # [0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
    K_proj_l2 = kernel_l2_proj(K, ε_l2, norm_cost = false)
    Ks_nys_proj_l2 = [kernel_l2_proj(x, ε_l2, norm_cost = false) for x in Ks_nys]
    CSV.write("K_proj_l2_eps_$(ε_l2).csv", DataFrame(K_proj_l2, :auto))
    for (i, n) in enumerate(ns)
        CSV.write("K_nys_$(n)_proj_l2_eps_$(ε_l2).csv", DataFrame(Ks_nys_proj_l2[i], :auto))
    end
end

for ε_ent in exp.(range(log(0.01), log(0.5), 25)) # [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    K_proj_ent = kernel_ent_proj(K, ε_ent, norm_cost = false)
    Ks_nys_proj_ent = [kernel_ent_proj(x, ε_ent, norm_cost = false) for x in Ks_nys]
    CSV.write("K_proj_ent_eps_$(ε_ent).csv", DataFrame(K_proj_ent, :auto))
    for (i, n) in enumerate(ns)
        CSV.write("K_nys_$(n)_proj_ent_eps_$(ε_ent).csv", DataFrame(Ks_nys_proj_ent[i], :auto))
    end
end
