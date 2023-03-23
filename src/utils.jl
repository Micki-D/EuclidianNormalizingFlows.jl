# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT)

function get_flow(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)
    d = floor(Int, n_dims/2) 
    i = 1
    all_dims = Integer[1:n_dims...]
    trafos = Function[]


    while d <= n_dims

        mask = broadcast(x -> x in i:d, all_dims)
        i += 1
        d+=1
        
        nn1, nn2 = _get_nns(n_dims, K, hidden, device)

        push!(trafos, CouplingRQSBlock(nn1, mask))
        push!(trafos, CouplingRQSBlock(nn2, .~mask))

    end

    return fchain(trafos)
end 

export get_flow


function get_flow_musketeer(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)

    trafos = Function[ScaleShiftNorm(device)]
    for i in 1:n_dims 
        mask = fill(false, n_dims)
        mask[i] = true
        nn = _get_nn_musketeer(n_dims, K, hidden, device)
        push!(trafos, CouplingRQSBlock(nn, mask))
    end
    return fchain(trafos)
end 

export get_flow_musketeer

function get_flow_musketeer_efficient(n_dims::Integer, device, K::Integer=10, hidden::Integer=20)

    trafos = Function[ScaleShiftNorm(device)]
    for i in 1:n_dims 
        mask = fill(false, n_dims)
        mask[i] = true
        nn = _get_nn_musketeer(n_dims, K, hidden, device)
        push!(trafos, CouplingRQSBlock(nn, mask))
    end
    return fchain(trafos)
end 

export get_flow_musketeer





function _get_nn_musketeer(n_dims::Integer, K::Integer, hidden::Integer, device)

    nn = Chain(
        Dense((n_dims-1) => hidden, relu),
        Dense(hidden => hidden, relu),
        Dense(hidden => (3K-1))
        )
    if device isa GPU
        nn = fmap(cu, nn)
    end  
    return nn
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


function ghm_integration(smpls::AbstractArray, logd_orig::AbstractVector, ladj::AbstractVector, id)
    
    smpls= cpu(flatview(smpls))
    ladj = cpu(ladj)
    n_smpls = size(smpls, 2)

    logd_posterior = logd_orig - ladj

    frac = [logpdf(id, smpls[:,i]) / logd_posterior[i] for i in 1:n_smpls]
    integral = n_smpls * inv(sum(frac)) 
    variance = sqrt(integral^2/n_smpls * var(frac))
    
    return integral, variance 
end

export ghm_integration




function get_scale_shifted_samples(d, nsamples::Integer, device)
    
    samples = bat_sample(d, 
                    BAT.IIDSampling(nsamples=nsamples)
                    ).result;

    smpls_flat, ladj1 = scale_shift_norm(ValueShapes.flatview(unshaped.(samples.v)))

    if device isa GPU
        smpls_flat = gpu(smpls_flat)
        ladj1 = gpu(ladj1)
    end

    return samples, smpls_flat, ladj1
end

export get_scale_shifted_samples

# just a hack, pls dont judge :(
function _sort_dimensions(y₁::AbstractArray, y₂::AbstractArray, mask::AbstractVector)
    
    if mask[1]
        res = reshape(y₁[1,:],1,size(y₁,2))
        c=2
    else
        res = reshape(y₂[1,:],1,size(y₁,2))
        c=1
    end

    for (i,b) in enumerate(mask[2:end])
        if b
            res = vcat(res, reshape(y₁[c,:],1,size(y₁,2)))
            c+=1
        else
            res = vcat(res, reshape(y₂[i+1,:],1,size(y₂,2)))
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
