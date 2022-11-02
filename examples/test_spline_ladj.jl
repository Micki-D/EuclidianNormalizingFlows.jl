using FiniteDifferences 
using Flux
using Revise
using EuclidianNormalizingFlows

algo = central_fdm(5, 1)

trafo = trained_trafo
x = smpls_flat

x₁ = reshape(x[trafo.mask1,:], length(trafo.mask1), size(x,2))
x₂ = reshape(x[trafo.mask2,:], length(trafo.mask2), size(x,2))

θ₁ = trafo.nn1(x₂)
w₁, h₁, d₁ = get_params(θ₁, size(x₁,1))
spline₁ = RQSpline(w₁, h₁, d₁)
y₁, LogJac₁ = EuclidianNormalizingFlows.with_logabsdet_jacobian(spline₁, x₁)

θ₂ = trafo.nn2(y₁)
w₂, h₂, d₂ = get_params(θ₂, size(x₂,1))
spline₂ = RQSpline(w₂, h₂, d₂)
y₂, LogJac₂ = EuclidianNormalizingFlows.with_logabsdet_jacobian(spline₂, x₂)

y = EuclidianNormalizingFlows._sort_dimensions(y₁, y₂, trafo.mask1)
lj = LogJac₁ + LogJac₂;

############################
# FiniteDifferences Logjac # 
############################
fd_lj₁ = Float64[]
fd_lj₂ = Float64[]

xs = [x₁, x₂]
fdljs = [fd_lj₁, fd_lj₂]

ws = [w₁, w₂]
hs = [h₁, h₂]
ds = [d₁,d₂]

for i in 1:2
    x = xs[i]
    fd_lj = fdljs[i]
    w = ws[i]
    h = hs[i]
    d = ds[i]
    for j in axes(x, 2)
        xrun = x[:,j]
        w_tmp = reshape(w[:,:,j], size(w,1), size(w,2), 1)
        h_tmp = reshape(h[:,:,j], size(w,1), size(w,2), 1)
        d_tmp = reshape(d[:,:,j], size(w,1), size(w,2), 1)

        autodiff_jac = FiniteDifferences.jacobian(algo, xtmp -> RQSpline(w_tmp, h_tmp, d_tmp)(reshape(xtmp, size(x,1), 1)), xrun)[1]

        append!(fd_lj, log(abs(det(autodiff_jac))))
    end
end

fd_lj = fd_lj₁ + fd_lj₂;