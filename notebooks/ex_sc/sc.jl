using CSV
using CodecZlib
using DataFrames
using OptimalTransport
using StatsBase
using LinearAlgebra
using PyCall
using Clustering

# fname = "leukemia_preprocessed.csv.gz"
fname = "liu_scrna_preprocessed.csv.gz"
df = CodecZlib.open("../../tools/OT-scOmics/data/$fname", "r") do io
        CSV.read(io, DataFrame)
end

y = map(x -> x[end], split.(names(df), "_")[2:end])
y = [findfirst(x .== unique(y)) for x in y]
X = Array(df[!, 2:end])

include("../../src/util.jl")

A = (y .== reshape(unique(y), 1, :))*1.0
n_clust = size(A, 2)
sp_la = pyimport_conda("scipy.linalg", "scipy")

skcluster = pyimport_conda("sklearn.cluster", "scikit-learn")
function clust(W, k)
    clust_op = skcluster.SpectralClustering(n_clusters = k, affinity = "precomputed", n_jobs = 4);
    clust_op.fit_predict(W); 
end

function get_score(K)
    get_eigvecs(K, k) = eigen(Hermitian(I-K)).vectors[:, 1:k]
    nmi = mutualinfo(clust(K, n_clust), y)
    theta = mean(sp_la.subspace_angles(get_eigvecs(K, 10), A))
    nmi, theta
end

eps_quad = 10 .^range(-2, 2, 25)
eps_ent = 10 .^range(-2, 2, 25)
ks = 5:5:125

unzip = x -> zip(x...)
@info "method: quad"
nmi_quad, theta_quad = map(x -> get_score(kernel_ot_quad(X, x)), eps_quad) |> unzip
@info "method: ent"
nmi_ent, theta_ent = map(x -> get_score(kernel_ot_ent(X, x)), eps_ent) |> unzip
@info "method: knn"
nmi_knn, theta_knn = map(x -> get_score(Array(norm_kernel(knn_adj(X, x), :sym))), ks) |> unzip
