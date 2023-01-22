# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT)

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
        nn3, nn4 = _get_nns(n_dims, K, hidden, device, length(mask1))
        
        d+=1
        i+=1

        push!(trafos, CouplingRQS(nn1, nn2, mask1, mask2))
        push!(trafos, CouplingRQS(nn3, nn4, mask2, mask1))

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

function scale_shift_norm(x::AbstractMatrix)
    n_smpls = size(x,2)    
    stds = Float64[]
    for i in axes(x, 1)
        std_tmp  = std(x[i,:])
        x[i,:] .*= 1/std_tmp
        append!(stds, std_tmp)

        mean_tmp = mean(x[i,:])
        x[i,:] .-= mean_tmp
    end

    #calculate logabsdetjacobian to track volume change thats introduced by the scaling 
    ladj_scs = sum(log.(abs.(1 ./ stds)))
    ladj = fill(ladj_scs, 1, n_smpls)

    return x, ladj
end

export scale_shift_norm

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
