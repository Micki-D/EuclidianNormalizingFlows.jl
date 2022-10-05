# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

function get_flow(n_dims::Integer, K::Integer=10, hidden::Integer=20)
    d = floor(Int, n_dims/2) 
    i = 1
    all_dims = [1:n_dims...]
    trafos = Function[]
    
    while d <= n_dims
        mask1 = [i:d...]
        mask2 = all_dims[.![(el in all_dims && el in mask1) for el in all_dims]]
        nns1 = _get_nns(n_dims, K, hidden)
        nns2 = _get_nns(n_dims, K, hidden)
        
        d+=1
        i+=1

        push!(trafos, CouplingRQS(nns1, mask1, mask2))
        push!(trafos, CouplingRQS(nns2, mask2, mask1))
    end

    return fchain(trafos)
end 

export get_flow

function _get_nns(n_dims::Integer, K::Integer, hidden::Integer)
    d = floor(Int, n_dims/2)
    nns = Chain[]

    for i in 1:(n_dims-d)
        nn_tmp = Chain(Dense(d => hidden, relu),
                       Dense(hidden => hidden, relu),
                       Dense(hidden => 3K-1)
        )
        push!(nns, nn_tmp)
    end

    return nns
end

function get_params(θ::AbstractMatrix, N::Integer, K::Integer)

    w = _cumsum(_softmax(θ[1:K,:]))
    h = _cumsum(_softmax(θ[K+1:2K,:]))
    d = _softplus(θ[2K+1:end,:])

    w = vcat(repeat([-5,], 1, N), w)
    h = vcat(repeat([-5,], 1, N), h)
    d = vcat(repeat([1,], 1, N), d)
    d = vcat(d, repeat([1,], 1, N))

    return w, h, d
end

export get_params

function _sort_dimensions(x::AbstractMatrix, y::AbstractMatrix, mask1::AbstractVector)
    
    if 1 in mask1
        res = reshape(x[1,:],1,size(x,2))
        c1 = 2
        c2 = 1
    else
        res = reshape(y[1,:],1,size(x,2))
        c1 = 1
        c2 = 2
    end

    for i in 2:(size(x,1)+size(y,1))
        if i in mask1
            res = vcat(res, reshape(x[c1,:],1,size(x,2)))
            c1+=1
        else
            res = vcat(res, reshape(y[c2,:],1,size(y,2)))
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

    val = hcat([_softmax(i) for i in eachcol(x)]...)

    return val 
end


function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return hcat([_cumsum(i) for i in eachcol(x)]...)
end


function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    return log.(exp.(x) .+ 1) 
end

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