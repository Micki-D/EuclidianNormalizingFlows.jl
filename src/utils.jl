# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

function get_flow(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)
    d = floor(Int, n_dims/2) 
    i = 1
    all_dims = Integer[1:n_dims...]
    trafos = Function[]
    
    while d <= n_dims
        mask1 = [i:d...]
        # mask2 = all_dims[.![el in mask1 for el in all_dims]]
        mask2 = vcat(1:1:(i-1), (d+1):1:n_dims)
        nn1, nn2 = _get_nns(n_dims, K, hidden, device)
        
        d+=1
        i+=1

        push!(trafos, CouplingRQS(nn1, nn2, mask1, mask2))
    end

    return fchain(trafos)
end 

export get_flow


function get_flow_musketeer(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)

    trafos = Function[]
    
    for i in 1:n_dims 
        mask1 = [i]
        # mask2 = all_dims[.![el in mask1 for el in all_dims]]
        mask2 = [1:(i-1)...,(i+1):n_dims...]
        nn1, nn2 = _get_nns_musketeer(n_dims, K, hidden, device)


        push!(trafos, CouplingRQS(nn1, nn2, mask1, mask2))
    end

    return fchain(trafos)
end 

export get_flow_musketeer



function get_flow_fd(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)

    trafos = Function[]

    mask1 = [1]
    mask2 = [2:n_dims...]
    nn1, nn2 = _get_nns(n_dims, K, hidden, device, 1)
    push!(trafos, CouplingRQS(nn1, nn2, mask1, mask2))

    mask3 = [2:n_dims...]
    mask4 = [1]
    nn3, nn4 = _get_nns(n_dims, K, hidden, device, n_dims-1)
    push!(trafos, CouplingRQS(nn3, nn4, mask3, mask4))

    return fchain(trafos)
end 

export get_flow_fd

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


struct BB <: Function
end
export BB

@functor BB

(f::BB)(x::AbstractMatrix) = blind_batchnorm(x)[1]


function ChangesOfVariables.with_logabsdet_jacobian(f::BB, x::AbstractMatrix)
    return blind_batchnorm(x)
end

function blind_batchnorm(x::AbstractMatrix)

    device = KernelAbstractions.get_device(x)

    μ = vec(mean(x; dims=2))
    σ_sqr = vec(var(x; dims=2))
    σ_inv = 1 ./ sqrt.(σ_sqr)

    y = (x .- μ) .* σ_inv
    logdet = device isa GPU ? gpu(fill(sum(log.(abs.(σ_inv))), 1, size(x,2))) : fill(sum(log.(abs.(σ_inv))), 1, size(x,2))

    return y, logdet
end

export blind_batchnorm



function _get_nns_musketeer(n_dims::Integer, K::Integer, hidden::Integer, device, d::Integer = floor(Int, n_dims/2),)

    nn1 = Chain(
        Dense((n_dims-1) => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => (3K-1))
        )


  

    # nn1 = Chain(
    #     #BatchNorm(n_dims-1),
    #     SkipConnection(
    #         Chain(Dense(n_dims-1 => hidden, relu),
    #         Dense(hidden => hidden, relu),

    #         #BatchNorm(hidden),
    #         Dense(hidden => n_dims-1, relu)),
    #         +),
    #     relu,
    #     #BatchNorm(n_dims-1),
    #     SkipConnection(
    #         Chain(Dense(n_dims-1 => hidden, relu),
    #         Dense(hidden => hidden, relu),

    #         #BatchNorm(hidden),
    #         Dense(hidden => n_dims-1, relu)),
    #         +),
    #     relu,
    #     #BatchNorm(n_dims-1),
    #     SkipConnection(
    #         Chain(Dense(n_dims-1 => hidden, relu),
    #         Dense(hidden => hidden, relu),
            
    #         #BatchNorm(hidden),
    #         Dense(hidden => n_dims-1, relu)),
    #         +),
    #     relu,
    #     #BatchNorm(n_dims-1),
    #     SkipConnection(
    #         Chain(Dense(n_dims-1 => hidden, relu),
    #         Dense(hidden => hidden, relu),

    #         #BatchNorm(hidden),
    #         Dense(hidden => n_dims-1, relu)),
    #         +),
    #     relu,
    #     #BatchNorm(n_dims-1),
    #     Dense(n_dims-1 => hidden, relu),
    #     Dense(hidden => hidden, relu),

    #     #BatchNorm(hidden),
    #     Dense(hidden => (3K-1))
    # )

    nn2 = Chain(
        Dense(1 => 1, relu)

    )

    if device isa GPU
        nn1 = fmap(cu, nn1)
        nn2 = fmap(cu, nn2)
    end   

    return nn1,nn2
end


function _get_nns(n_dims::Integer, K::Integer, hidden::Integer, device, d::Integer = floor(Int, n_dims/2),)

    # nn1 = Chain(
    #     # BatchNorm(n_dims-d),
    #     # SkipConnection(
    #     #     Chain(Dense(n_dims-d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => n_dims-d, relu)),
    #     #     +),
    #     # relu,
    #     # BatchNorm(n_dims-d),
    #     # SkipConnection(
    #     #     Chain(Dense(n_dims-d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => n_dims-d, relu)),
    #     #     +),
    #     # relu,
    #     # BatchNorm(n_dims-d),
    #     # SkipConnection(
    #     #     Chain(Dense(n_dims-d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => n_dims-d, relu)),
    #     #     +),
    #     # relu,
    #     BatchNorm(n_dims-d),
    #     SkipConnection(
    #         Chain(Dense(n_dims-d => hidden, relu),
    #         BatchNorm(hidden),
    #         Dense(hidden => n_dims-d, relu)),
    #         +),
    #     relu,
    #     BatchNorm(n_dims-d),
    #     Dense(n_dims-d => hidden, relu),
    #     BatchNorm(hidden),
    #     Dense(hidden => d*(3K-1))
    # )

    # nn2 = Chain(
    #     # BatchNorm(d),
    #     # SkipConnection(
    #     #     Chain(Dense(d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => d, relu)),
    #     #     +),
    #     # relu,
    #     # BatchNorm(d),
    #     # SkipConnection(
    #     #     Chain(Dense(d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => d, relu)),
    #     #     +),
    #     # relu,
    #     # BatchNorm(d),
    #     # SkipConnection(
    #     #     Chain(Dense(d => hidden, relu),
    #     #     BatchNorm(hidden),
    #     #     Dense(hidden => d, relu)),
    #     #     +),
    #     # relu,
    #     BatchNorm(d),
    #     SkipConnection(
    #         Chain(Dense(d => hidden, relu),
    #         BatchNorm(hidden),
    #         Dense(hidden => d, relu)),
    #         +),
    #     relu,
    #     BatchNorm(d),
    #     Dense(d => hidden, relu),
    #     BatchNorm(hidden),
    #     Dense(hidden => (n_dims-d)*(3K-1))
    # )

    nn1 = Chain(
        Dense(n_dims-d => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => d*(3K-1))
    )

    nn2 = Chain(
        Dense(d => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => (n_dims-d)*(3K-1))
    )

    # nn1 = Chain(
    #     Dense(n_dims-d => d*(3K-1))
    # )

    # nn2 = Chain(
    #     Dense(d => (n_dims-d)*(3K-1))
    # )

    if device isa GPU
        nn1 = fmap(cu, nn1)
        nn2 = fmap(cu, nn2)
    end   

    return nn1,nn2
end

function get_params(θ_raw::AbstractArray, n_dims_trafo::Integer, B::Real = 5.)

    N = size(θ_raw, 2)
    K = Int((size(θ_raw,1)/n_dims_trafo+1)/3)
    θ = reshape(θ_raw, :, n_dims_trafo, N)

    device = KernelAbstractions.get_device(θ_raw)

    w = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[1:K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[1:K,:,:])), dims = 1)
    h = device isa GPU ? cat(cu(repeat([-B], 1, n_dims_trafo, N)), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])); dims = 1) : cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])), dims = 1)
    d = device isa GPU ? cat(cu(repeat([1], 1, n_dims_trafo, N)), _softplus_tri(θ[2K+1:end,:,:]), cu(repeat([1], 1, n_dims_trafo, N)); dims = 1) : cat(repeat([1], 1, n_dims_trafo, N), _softplus_tri(θ[2K+1:end,:,:]), repeat([1], 1, n_dims_trafo, N), dims = 1)

    # w = cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[1:K,:,:])), dims = 1)
    # h = cat(repeat([-B], 1, n_dims_trafo, N), _cumsum_tri(_softmax_tri(θ[K+1:2K,:,:])), dims = 1)
    # d = cat(repeat([1], 1, n_dims_trafo, N), _softplus_tri(θ[2K+1:end,:,:]), repeat([1], 1, n_dims_trafo, N), dims = 1)

    return w, h, d
end

export get_params


# function _get_nn(n_dims::Integer, K::Integer, hidden::Integer)
#     d = floor(Int, n_dims/2)

#     nn = Chain(Dense(d => hidden, relu),
#                Dense(hidden => hidden, relu),
#                Dense(hidden => (n_dims-d)*(3K-1))
#                )

#     return nn
# end

# function format_params(raw_w::AbstractArray, raw_h::AbstractArray, raw_d::AbstractArray)

#     one_pad = repeat([1], size(raw_w, 1), size(raw_w, 2))
#     five_pad = repeat([-5], size(raw_w, 1), size(raw_w, 2))

#     w = cat(ignore_derivatives(five_pad),_cumsum_tri(_softmax_tri(raw_w)),dims=3)
#     h = cat(ignore_derivatives(five_pad),_cumsum_tri(_softmax_tri(raw_h)),dims=3)
#     d = cat(ignore_derivatives(one_pad),_softmax_tri(raw_d),ignore_derivatives(one_pad),dims=3)

#     return w,h,d
# end

# export format_params


function _sort_dimensions(y₁::AbstractMatrix, y₂::AbstractMatrix, mask1::AbstractVector)
    
    if 1 in mask1
        res = reshape(y₁[1,:],1,size(y₁,2))
        c1 = 2
        c2 = 1
    else
        res = reshape(y₂[1,:],1,size(y₁,2))
        c1 = 1
        c2 = 2
    end

    for i in 2:(size(y₁,1)+size(y₂,1))
        if i in mask1
            res = vcat(res, reshape(y₁[c1,:],1,size(y₁,2)))
            c1+=1
        else
            res = vcat(res, reshape(y₂[c2,:],1,size(y₂,2)))
            c2+=1
        end
    end
    
    return res
end


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