# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


struct ScaleShiftTrafo{T<:Union{Real,AbstractVector{<:Real}}} <: Function
    a::T
    b::T
end

@functor ScaleShiftTrafo

Base.:(==)(a::ScaleShiftTrafo, b::ScaleShiftTrafo) = a.a == b.a && a.b == b.b
Base.isequal(a::ScaleShiftTrafo, b::ScaleShiftTrafo) = isequal(a.a, b.a) && isequal(a.b, b.b)
Base.hash(x::ScaleShiftTrafo, h::UInt) = hash(x.a, hash(x.b, hash(:JohnsonTrafoInv, hash(:ScaleShiftTrafo, h))))

(f::ScaleShiftTrafo{<:Real})(x::Real) = muladd(x, f.a, f.b)
(f::ScaleShiftTrafo)(x) = muladd.(x, f.a, f.b)
(f::ScaleShiftTrafo)(vs::AbstractValueShape) = vs

function ChangesOfVariables.with_logabsdet_jacobian(
    f::ScaleShiftTrafo{<:AbstractVector{<:Real}},
    x::AbstractMatrix{<:Real}
)
    ladj = sum(log.(abs.(f.a)))
    f(x), similar_fill(ladj, x, (size(x, 2),))'
end

function InverseFunctions.inverse(f::ScaleShiftTrafo)
    a_inv = inv.(f.a)
    b_inv = - a_inv .* f.b
    ScaleShiftTrafo(a_inv, b_inv)
end


@with_kw mutable struct AdaptiveScaleShift <: Function
    a::AbstractArray = []
    b::AbstractArray = []
    initiated::Bool = false
end

@functor AdaptiveScaleShift
export AdaptiveScaleShift

(f::AdaptiveScaleShift)(x) = f.initiated ? muladd(f.a, x, f.b) : init_scale_shift(f, x) 
(f::AdaptiveScaleShift)(vs::AbstractValueShape) = vs

function ChangesOfVariables.with_logabsdet_jacobian(
    f::AdaptiveScaleShift,
    x::Any
)
    return f(x), fill(sum(log.(abs.(diag(f.a)))), 1, size(x,2))
end

function InverseFunctions.inverse(f::AdaptiveScaleShift)
    a_inv = inv(f.a)
    b_inv = vec(f.b .* diag(-a_inv))
    return AdaptiveScaleShift(a_inv, b_inv, f.initiated)
end

function init_scale_shift(f::AdaptiveScaleShift, x::AbstractArray)
    a = inv.(vec(std(x, dims = 2)))
    b = vec(mean(x, dims =2).*a)
    f.a = Diagonal(a) 
    f.b = -b 
    f.initiated = true
    return muladd(f.a, x, f.b)
end
