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

struct AdaptiveScaleShift <: Function
    trafo::ScaleShiftTrafo
end



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
