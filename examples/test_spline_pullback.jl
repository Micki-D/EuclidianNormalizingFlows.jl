using Pkg
Pkg.activate("/home/micki/.julia/environments/splines") # old

using FiniteDifferences
using Revise 
using EuclidianNormalizingFlows
using LinearAlgebra

function sc_cost(trafo::Function, x::AbstractMatrix)
    nsamples = size(x, 2) 
    Y, ladj = EuclidianNormalizingFlows.with_logabsdet_jacobian(trafo, x)
    
    ll = (sum(EuclidianNormalizingFlows.std_normal_logpdf.(Y)) + sum(ladj)) / nsamples
    return -ll
end

export sc_cost

function skeleton_cost(trafo::RQSpline, x::AbstractMatrix)
    y = EuclidianNormalizingFlows.with_logabsdet_jacobian(trafo,x)[1]

    return sum(sum(y,dims=1))
end


function drop_low_vals(x::Real)
    return abs(x)>10^(-10) ? x : zero(typeof(x))
end 

export drop_low_vals



algo = central_fdm(5, 1)
rtol = 0.001


n_dims = 2
N = 100
B = 5.

x = randn(n_dims,N)

w_r = randn(10, n_dims, N)
h_r = randn(10, n_dims, N)
d_r = randn(9, n_dims, N)

w = cat(repeat([-B], 1, n_dims, N), _cumsum_tri(_softmax_tri(w_r)), dims = 1)
h = cat(repeat([-B], 1, n_dims, N), _cumsum_tri(_softmax_tri(h_r)), dims = 1)
d = cat(repeat([1], 1, n_dims, N), _softplus_tri(d_r), repeat([1], 1, n_dims, N), dims = 1)



trafo = RQSpline(w,h,d)

negll, back = Zygote.pullback(sc_cost, trafo, x)
Zgrad = back(one(eltype(x)))[1]


w_grad = drop_low_vals.(FiniteDifferences.grad(algo, par -> sc_cost(RQSpline(par,h,d), x), w)[1])
h_grad = drop_low_vals.(FiniteDifferences.grad(algo, par -> sc_cost(RQSpline(w,par,d), x), h)[1])
d_grad = drop_low_vals.(FiniteDifferences.grad(algo, par -> sc_cost(RQSpline(w,h,par), x), d)[1])


@test isapprox(w_grad, Zgrad.widths, rtol = rtol)
@test isapprox(h_grad, Zgrad.heights, rtol = rtol)
@test isapprox(d_grad, Zgrad.derivatives, rtol = rtol)

y, lj = EuclidianNormalizingFlows.with_logabsdet_jacobian(trafo, x)


for j in axes(x, 2)
    xrun = x[:,j]
    w_tmp = reshape(w[:,:,j], size(w,1), size(w,2), 1)
    h_tmp = reshape(h[:,:,j], size(w,1), size(w,2), 1)
    d_tmp = reshape(d[:,:,j], size(w,1), size(w,2), 1)

    autodiff_jac = FiniteDifferences.jacobian(algo, xtmp -> RQSpline(w_tmp, h_tmp, d_tmp)(reshape(xtmp, n_dims, 1)), xrun)[1]
    @test isapprox.(log(abs(det(autodiff_jac))), lj[1,j], rtol = rtol)
end

fd_lj = Float64[]

for j in axes(x, 2)
    xrun = x[:,j]
    w_tmp = reshape(w[:,:,j], size(w,1), size(w,2), 1)
    h_tmp = reshape(h[:,:,j], size(w,1), size(w,2), 1)
    d_tmp = reshape(d[:,:,j], size(w,1), size(w,2), 1)

    autodiff_jac = FiniteDifferences.jacobian(algo, xtmp -> RQSpline(w_tmp, h_tmp, d_tmp)(reshape(xtmp, n_dims, 1)), xrun)[1]

    append!(fd_lj, log(abs(det(autodiff_jac))))
end


####################################################


nn1, nn2 = EuclidianNormalizingFlows._get_nns(n_dims, 10, 20, device)

nn1 = Chain(
            Dense(Float64.(nn1.layers[1].weight),Float64.(nn1.layers[1].bias), relu),
            Dense(Float64.(nn1.layers[2].weight),Float64.(nn1.layers[2].bias), relu),
            Dense(Float64.(nn1.layers[3].weight),Float64.(nn1.layers[3].bias))
)

nn2 = Chain(
            Dense(Float64.(nn2.layers[1].weight),Float64.(nn2.layers[1].bias), relu),
            Dense(Float64.(nn2.layers[2].weight),Float64.(nn2.layers[2].bias), relu),
            Dense(Float64.(nn2.layers[3].weight),Float64.(nn2.layers[3].bias))
)

mask1 = [1]
mask2 = [2]

trafo = CouplingRQS(nn1, nn2, mask1, mask2)

negll, back = Zygote.pullback(sc_cost, trafo, x)
Zgrad = back(one(eltype(x)))[1]

nn1_grad = FiniteDifferences.grad(algo, par -> sc_cost(CouplingRQS(par,nn2, mask1, mask2), x), nn1)[1]


for i in 1:3
    try 
        @test isapprox(Zgrad.nn1.layers[i].weight, nn1_grad.layers[i].weight, rtol = rtol)
        @test isapprox(Zgrad.nn1.layers[i].bias, nn1_grad.layers[i].bias, rtol = rtol)
    catch
        @show(i) 
    end
end
