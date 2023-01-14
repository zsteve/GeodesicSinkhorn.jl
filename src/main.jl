using OptimalTransport
using NPZ
using Plots

X = npzread("X.npy")
dpt = npzread("dpt.npy")
using MultivariateStats
pca_op = fit(PCA, X')
X_pca = predict(pca_op, X')'

scatter(X_pca[:, 1], X_pca[:, 2]; markerstrokewidth = 0, alpha = 0.25)

using NearestNeighbors
using SparseArrays
using Graphs
using GraphSignals
using Laplacians
using LinearAlgebra
using Expokit

mu = normalize(dpt .< 0.1, 1)
nu = normalize(dpt .> 0.9, 1)

# kNN and Laplacian construction
k(r; h = 2.5) = exp(-(r/h)^2)
kdtree = KDTree(X_pca')
idxs, dists = knn(kdtree, X_pca', 5);
A = spzeros(size(X_pca, 1), size(X_pca, 1));
for (i, (j, d), ) in enumerate(zip(idxs, dists))
    A[i, j] .= k.(d)
end
L = sparse(laplacian_matrix(max.(A, A'), Float64));

p = normalize(dpt .> 0.95, 1)
t = 10.0
p_t = chbv(-t*L, p)
scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = p_t)

# Sinkhorn
using ProgressMeter
function geodesic_sinkhorn(t, L, mu, nu; iter = 25)
    u = similar(mu)
    v = similar(nu); fill!(v, 1)
    for i = 1:iter
        u = mu ./ chbv(-t*L, v)
        v = nu ./ chbv(-t*L, u)
    end
    return u, v
end

u, v = geodesic_sinkhorn(t, L, mu, nu)
pi = (u .* v') .* exp(-Array(t*L))

using LogExpFunctions
dot(xlogx.(u), chbv(-t*L, v)) + dot(xlogx.(v), chbv(-t*L', u))

# check marginal constraints
norm(vec(sum(pi; dims = 1)) .- nu, 1)
norm(vec(sum(pi; dims = 2)) .- mu, 1)

# Barycenters
import Expokit.chbv
chbv(A, v::Matrix{T}) where T = hcat([chbv(A, Vector{T}(x)) for x in eachcol(v)]...)

t = 10.0

function geodesic_sinkhorn_barycenter(t, L, mu, w; iter = 25)
    a = similar(mu)
    u = similar(mu)
    Kv = similar(u);
    v = similar(mu); fill!(v, 1)
    for i = 1:iter
        Kv .= chbv(-t*L, v)
        a .= prod(Kv' .^ w; dims=1)'
        u .= a ./ Kv
        v .= mu ./ chbv(-t*L', u)
    end
    Kv .= chbv(-t*L, v)
    p = u[:, 1] .* Kv[:, 1]
    return p
end

mu_all = hcat(mu, nu)
w = [0.5, 0.5]

p = geodesic_sinkhorn_barycenter(t, L, mu_all, w)

using OptimalTransport
using Distances
using StatsBase
C = convert.(Float64, pairwise(SqEuclidean(), X_pca'))
C /= mean(C)
q = sinkhorn_barycenter(mu_all, C, 0.1, w, OptimalTransport.SinkhornGibbs())

plot(scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = mu, title = "μ"),
    scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = nu, title = "ν"),
    scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = p, title = "geodesic barycenter", clim = (0, quantile(p, 0.99))),
    scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = q, title = "barycenter", clim = (0, quantile(q, 0.99))))
