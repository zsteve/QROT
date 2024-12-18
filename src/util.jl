# Utilities for numerical experiments with QROT and other kernel constructions

using Distances 
using NearestNeighbors
using SparseArrays

function get_cost(X; diag_inf = false)
    C = pairwise(SqEuclidean(), X, X)
    C = mean_norm(C)
    if diag_inf
        C[diagind(C)] .= Inf
    end
    C
end

mean_norm(x) = x ./ mean(x)

symm(A) = 0.5*(A.+A')

function kernel_ot_ent(X, ε; diag_inf = true, rtol = 1e-6, atol = 1e-6, potentials = false, kwargs...)
    solver=OptimalTransport.build_solver(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricSinkhornGibbs(); maxiter = 5_000, atol = atol, rtol = rtol, kwargs...)
    OptimalTransport.solve!(solver)
    K = norm_kernel(symm(OptimalTransport.sinkhorn_plan(solver)), :row)
    return potentials ? (K, solver.cache.u) : K
end

function kernel_ot_quad(X, ε; diag_inf = true, rtol = 1e-6, atol = 1e-6, potentials = false, kwargs...)
    solver=OptimalTransport.build_solver(ones(size(X, 2)), get_cost(X; diag_inf = diag_inf), ε, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100, atol = atol, rtol = rtol, kwargs...)
    OptimalTransport.solve!(solver)
    K = norm_kernel(symm(OptimalTransport.plan(solver)), :row)
    return potentials ? (K, solver.cache.u) : K
end

kernel_epanech = (X, ε; rtol = 1e-6, atol = 1e-9) -> norm_kernel(symm(relu.(1 .- get_cost(X; diag_inf = false)/ε)), :sym)

function kernel_gaussian_l2_proj(X, ε; diag_inf = true, rtol = 1e-6, atol = 1e-6, potentials = false, kwargs...)
    K = form_kernel(X, ε)
    C = (ones(size(K, 1)) * ones(size(K, 2))') - K
    if diag_inf
        C[diagind(C)] .= Inf
    end
    solver = OptimalTransport.build_solver(ones(size(X, 2)), C, 1, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100, atol = atol, rtol = rtol, kwargs...)
    OptimalTransport.solve!(solver)
    K = norm_kernel(symm(OptimalTransport.plan(solver)), :row)
    return potentials ? (K, solver.cache.u) : K
end

function kernel_l2_proj(K, ε; diag_inf = true, rtol = 1e-6, atol = 1e-6, potentials = false, norm_cost = false, kwargs...)
    _norm = norm_cost ? mean_norm : x -> x
    C = _norm((ones(size(K, 1)) * ones(size(K, 2))') - K) / ε
    if diag_inf
        C[diagind(C)] .= Inf
    end
    solver = OptimalTransport.build_solver(ones(size(K, 1)), C, 1, OptimalTransport.SymmetricQuadraticOTNewton(); maxiter = 100, atol = atol, rtol = rtol, kwargs...)
    OptimalTransport.solve!(solver)
    K = norm_kernel(symm(OptimalTransport.plan(solver)), :row)
    return potentials ? (K, solver.cache.u) : K
end

function kernel_ent_proj(K, ε; diag_inf = true, rtol = 1e-6, atol = 1e-6, potentials = false, norm_cost = false, kwargs...)
    _norm = norm_cost ? mean_norm : x -> x
    # C = _norm(-log.(K) / ε)
    C = _norm((ones(size(K, 1)) * ones(size(K, 2))') - K) / ε
    if diag_inf
        C[diagind(C)] .= Inf
    end
    solver=OptimalTransport.build_solver(ones(size(C, 1)), C, 1, OptimalTransport.SymmetricSinkhornGibbs(); maxiter = 5_000, atol = atol, rtol = rtol, kwargs...)
    OptimalTransport.solve!(solver)
    K = norm_kernel(symm(OptimalTransport.sinkhorn_plan(solver)), :row)
    return potentials ? (K, solver.cache.u) : K
end

function knn_adj(X, k)
    # indices, _ = knn_matrices(nndescent(X, k, Euclidean())); 
    indices, _ = knn(KDTree(X), X, k);
    A = spzeros(size(X, 2), size(X, 2));
    @inbounds for i = 1:size(A, 1)
        A[i, i] = 1
        @inbounds for j in indices[i]
            A[i, j] = 1
        end
    end
    return A
end

function form_kernel(X, ε; k = Inf)
    C = get_cost(X)
    K = exp.(-C/ε)
    if k < Inf
        K .= K .* knn_adj(X, k)
    end
    # K[diagind(K)] .= 0
    symm(K)
end

function norm_kernel(K, type)
    W = K
    if type == :unnorm
        # do nothing
    elseif type == :row
        W .= K ./ reshape(sum(K; dims = 2), :, 1)
    elseif type == :sym
        r = sum(K; dims = 2)
        W .= K .* sqrt.(1f0 ./reshape(r, :, 1)) .* sqrt.(1f0 ./reshape(r, 1, :))
    else
        throw(ArgumentError)
    end
    W
end
