using Pkg
Pkg.activate("gen")

using RCall
using Plots
@rlibrary TDA

r, R = 0.5, 1

Ns = [1_000, 2_500, 5_000, 10_000]
Xs = Dict(N => [convert(Array, torusUnif(N, r, R)) for _ = 1:10] for N in Ns)

using Serialization
serialize("Xs.dat", Xs)
