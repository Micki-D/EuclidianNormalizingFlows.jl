# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

function hard_flow(n_dims::Integer, device, S::Integer=2)

    blocks = Function[]

    for i in 1:S

        push!(blocks, HardFlow(Conv1x1(n_dims, device), CouplingRQS(get_nn(n_dims, 10, 20, device)),CouplingRQS(get_nn(n_dims, 10, 20, device)), false))
    end

    return fchain(blocks)
end

export hard_flow

function get_flow(n_dims::Integer, device, K::Integer=10, hidden::Integer=20, S::Integer=4, L::Integer=1)
    
    trafos = Function[]
    
    for i in 1:S
        push!(trafos, Conv1x1(n_dims), CouplingRQS(get_nn(n_dims, K, hidden, device)))
    end

    return fchain(trafos)
end 

export get_flow


struct DB <: Function
end
@functor DB

(f::DB)(x::AbstractMatrix) = dumb_batchnorm(x)

function dumb_batchnorm(x::AbstractMatrix)

    device = KernelAbstractions.get_device(X)

    μ = vec(mean(x; dims=2))
    σ_sqr = vec(var(x; dims=2))
    σ_inv = 1 ./ sqrt.(σ_sqr)

    y = (x .- μ) .* σ_inv
    logdet = device isa GPU ? gpu(fill(sum(log.(abs.(σ_inv))), 1, size(X,2))) : fill(sum(log.(abs.(σ_inv))), 1, size(X,2))

    return y, logdet
end

export DB
export dumb_batchnorm


function get_indie_flow(n_dims::Integer, N::Integer, device, K::Integer=10, hidden::Integer=20)
    d = floor(Int, n_dims/2) 
    i = 1
    all_dims = Integer[1:n_dims...]
    trafos = Function[]
    
    while d <= n_dims
        mask1 = [i:d...]
        # mask2 = all_dims[.![el in mask1 for el in all_dims]]
        mask2 = vcat(1:1:(i-1), (d+1):1:n_dims)
        nn1, nn2 = _get_nns(n_dims, K, hidden, device)
        
        params = fill(1, (3K-1)*(n_dims-d), N) 

        d+=1
        i+=1

        push!(trafos, PRQS(nn2, mask1, mask2, params))
    end

    return fchain(trafos)
end 

export get_indie_flow

function get_nn(n_dims::Integer, K::Integer, hidden::Integer, device)
    d = ceil(Int, n_dims/2)

    nn = Chain(
        Dense(n_dims-d => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => d*(3K-1))
    )

    if device isa GPU
        nn = fmap(cu, nn)
    end   

    return nn
end

export get_nn

function get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)

    N = size(θ_raw, 2)
    K = Int((size(θ_raw,1)/n_dims_trafo+1)/3)
    θ = reshape(θ_raw, :, n_dims_trafo, N)

    device = KernelAbstractions.get_device(θ_raw)

    w = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[1:K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[1:K,:,:])), dims = 1)
    h = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])), dims = 1)
    d = device isa GPU ? cat(cu(repeat([1], 1, n_dims_trafo, N)), _softplus_tri(θ[2K+1:end,:,:]), cu(repeat([1], 1, n_dims_trafo, N)); dims = 1) : cat(repeat([1], 1, n_dims_trafo, N), _softplus_tri(θ[2K+1:end,:,:]), repeat([1], 1, n_dims_trafo, N), dims = 1)

    return w, h, d
end

export get_params


function _softmax(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)

    val = cat([_softmax(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function _softmax_tri(x::AbstractArray)
    exp_x = exp.(x)
    inv_sum_exp_x = inv.(sum(exp_x, dims = 1))

    return inv_sum_exp_x .* exp_x
end

export _softmax_tri

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return cat([_cumsum(i) for i in eachrow(x)]..., dims=2)'
end

function _cumsum_tri(x::AbstractArray, B::Real = 5.)
    
    return 2 .* B .* cumsum(x, dims = 1) .- B 
end

export _cumsum_tri

function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    val = cat([_softplus(i) for i in eachrow(x)]..., dims=2)'

    return val
end

function _softplus_tri(x::AbstractArray)
    return log.(exp.(x) .+ 1) 
end

export _softplus_tri

midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
binary_log(x::T) where {T<:Integer} = 8 * sizeof(T) - leading_zeros(x - 1)

function searchsortedfirst_impl(
        v::AbstractVector, 
        x::Real
    )
    
    u = one(Integer)
    lo = one(Integer) - u
    hi = length(v) + u
    
    n = binary_log(length(v))+1
    m = one(Integer)
    
    @inbounds for i in 1:n
        m_1 = midpoint(lo, hi)
        m = Base.ifelse(lo < hi - u, m_1, m)
        lo = Base.ifelse(v[m] < x, m, lo)
        hi = Base.ifelse(v[m] < x, hi, m)
    end
    return hi
end

function linear_scan()

    return w1, w2, h1, h2, d1, d2
end


# function get_params(nns::AbstractArray, x::AbstractMatrix)
    
#     res = format_params(nns[1](x[:,1]))

#     for i in 2:length(nns)
#         res = hcat(res,format_params(nns[i](x[:,1])))
#     end

#     for j in 2:size(x,2)
#         res_tmp = format_params(nns[1](x[:,j]))
#         for k in 2:length(nns)
#             res_tmp = hcat(res_tmp,format_params(nns[k](x[:,j])))
#         end
#         res = cat(res,res_tmp,dims=3)
#     end

#     res = permutedims(res,(2,3,1))

#     K = Int((size(res,3)-3)/3)
    
#     w = res[:,:,1:K+1]
#     h = res[:,:,K+2:2(K+1)]
#     d = res[:,:,2(K+1)+1:end]

#     return w, h, d 
# end