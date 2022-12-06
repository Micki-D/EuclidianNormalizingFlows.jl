
export Conv1x1

struct Conv1x1 <: Function
    k::Integer
    v1::AbstractMatrix
    v2::AbstractMatrix
    v3::AbstractMatrix
    freeze::Bool
end

@functor Conv1x1

function Conv1x1(v1, v2, v3; freeze=false)
    k = length(v1)
    return Conv1x1(k, v1, v2, v3, freeze)
end

function Conv1x1(k;freeze=false)
    v1 = reshape(Flux.glorot_uniform(k), 1, k)
    v2 = reshape(Flux.glorot_uniform(k), 1, k)
    v3 = reshape(Flux.glorot_uniform(k), 1, k)

    return Conv1x1(k, v1, v2, v3, freeze)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::Conv1x1,
    x::AbstractMatrix{<:Real}
)
    return forward(f, x)
end

(f::Conv1x1)(x::AbstractArray) = forward(f,x)[1]

function InverseFunctions.inverse(f::Conv1x1)
    return Conv1x1Inv(f.k, f.v1, f.v2, f.v3, f.freeze)
end

struct Conv1x1Inv <: Function
    k::Integer
    v1::AbstractMatrix
    v2::AbstractMatrix
    v3::AbstractMatrix
    freeze::Bool
end

@functor Conv1x1Inv

function ChangesOfVariables.with_logabsdet_jacobian(
    f::Conv1x1Inv,
    x::AbstractMatrix{<:Real}
)
    return inverse(f, x)
end

(f::Conv1x1Inv)(x::AbstractMatrix) = inverse(f,x)[1]

function InverseFunctions.inverse(f::Conv1x1Inv)
    return Conv1x1(f.k, f.v1, f.v2, f.v3, f.freeze)
end


# Forward pass
function forward(C::Conv1x1, X::AbstractMatrix{T}) where T

    Y = apply_conv(eltype(C.v1).(X), C.v1, C.v2, C.v3)
    return Y, fill(eltype(Y)(0), 1, size(Y,2)) # logdet always 0
end

# Inverse pass
function inverse(C::Conv1x1Inv, Y::AbstractArray{T, N}) where {T, N}
    X = apply_conv(eltype(C.v1).(Y), C.v3, C.v2, C.v1)
    return X, fill(eltype(X)(0), 1, size(X,2)) # logdet always 0
end


# for 1x1 Conv
#gemm_outer!(out::Matrix{T},  v::Matrix{T}, tmp::Matrix{T}) where T = LinearAlgebra.BLAS.gemm!('T', 'N', T(1), v, tmp, T(1), out)
#gemm_outer!(out::CuMatrix{T}, v::CuMatrix{T}, tmp::CuVector{T}) where T = CUDA.CUBLAS.gemm!('T', 'N', T(1), v, tmp, T(1), out)

function apply_conv(x::AbstractMatrix{T}, v::Vararg{AbstractMatrix{T}, N}) where {T, N}

    # tmp = fill(zero(T), 1, size(x,2))
    # Y = deepcopy(x)
    # for v=vi
    #     n = -2/norm(v)^2
    #     mul!(tmp, v, Y)
    #     rmul!(tmp, n)
    #     gemm_outer!(Y, v, tmp)
    # end

    Y1 = -2/norm(v[1])^2 * v[1]' * v[1] * x + x
    Y2 = -2/norm(v[2])^2 * v[2]' * v[2] * Y1 + Y1
    Y3 = -2/norm(v[3])^2 * v[3]' * v[3] * Y2 + Y2

    return Y3 
end

# X = Float32.(vcat(fill(1,1,10), fill(2,1,10), fill(3,1,10), fill(4,1,10)))
