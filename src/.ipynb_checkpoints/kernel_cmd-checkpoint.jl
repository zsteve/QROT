# Command line script for QROT kernel computation
#
using OptimalTransport
using LinearAlgebra
using Random
using Distributions
using Distances
using NearestNeighbors
using SparseArrays
using NNlib
using MultivariateStats
using NPZ
using ArgParse

include("util.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "X"
        arg_type = String
        required = true
        help = "input matrix (.npy)"
    "eps"
        arg_type = Float64
        required = true
        help = "regularisation parameter (ε) for QROT" 
    "--diag_inf"
        arg_type = Bool
        default = true
        help = "if `true`, outputs a hollow projection"
    "--out_W"
        arg_type = String
        default = "W.npy"
        help = "output affinity matrix filename"
    "--out_u"
        arg_type = String
        default = "u.npy"
        help = "output dual potential filename"
    "--mode"
        arg_type = String
        default = "quad"
        help = "choice of regularisation, either quad or ent"
end

args = parse_args(ARGS, s)

@info "Loading input file $(args["X"])"
X = npzread(args["X"])

@info "Calculating kernel"
@time W, u = Symbol(args["mode"]) == :quad ? kernel_ot_quad(X', args["eps"], diag_inf = args["diag_inf"], potentials = true) : 
                                        (Symbol(args["mode"]) == :ent ? kernel_ot_ent(X', args["eps"], diag_inf = args["diag_inf"], potentials = true) : throw(ArgumentError("Invalid mode")))

@info "Writing output to $(args["out_W"]), $(args["out_u"])"
npzwrite(args["out_W"], W)
npzwrite(args["out_u"], u)

