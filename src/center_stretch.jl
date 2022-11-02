# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


function center_stretch(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    exp_abs_bx = exp(abs(b * x))
    convert(R, sign(x) * log((sqrt((1 - exp_abs_bx)^2 * exp(2 * b*a) + 4* exp_abs_bx) - (1 - exp_abs_bx) * exp(b*a))/2)/b + c)
end


function center_contract(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    x_unshifted = x - c
    convert(R, (log(1 + exp(b * (x_unshifted-a))) - log(1 + exp(-b * (x_unshifted+a)))) / b)
end

function center_contract_ladj(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    x_unshifted = x - c
    dy_dx = 1/(1 + exp(-b * (x_unshifted - a))) + 1/(1 + exp(b * (x_unshifted + a)))
    log(abs(dy_dx))
end


@with_kw struct CenterStretch{T<:Union{Real,AbstractVector{<:Real}}} <: Function
    a::T = 0.0
    b::T = 1.0
    c::T = 0.0
end

@functor CenterStretch

Base.:(==)(a::CenterStretch, b::CenterStretch) = a.a == b.a && a.b == b.b && a.c == b.c
Base.isequal(a::CenterStretch, b::CenterStretch) = isequal(a.a, b.a) && isequal(a.b, b.b) && isequal(a.c, b.c)
Base.hash(x::CenterStretch, h::UInt) = hash(x.c, hash(x.b, hash(x.a, hash(:CenterStretch, hash(:EuclidianNormalizingFlows, h)))))

(f::CenterStretch)(x::Union{Real,AbstractVecOrMat{<:Real}}) = center_stretch.(x, f.a, f.b, f.c)

function with_logabsdet_jacobian(f::CenterStretch, x::Union{Real,AbstractVecOrMat{<:Real}})
    y = f(x)

    neg_ladjs = center_contract_ladj.(y, f.a, f.b, f.c)
    (y, - sum_ladjs(neg_ladjs))
end

inverse(f::CenterStretch) = CenterContract(f.a, f.b, f.c)



@with_kw struct CenterContract{T<:Union{Real,AbstractVector{<:Real}}} <:Function
    a::T = 0.0
    b::T = 1.0
    c::T = 0.0
end

@functor CenterContract

Base.:(==)(a::CenterContract, b::CenterContract) = a.a == b.a && a.b == b.b && a.c == b.c
Base.isequal(a::CenterContract, b::CenterContract) = isequal(a.a, b.a) && isequal(a.b, b.b) && isequal(a.c, b.c)
Base.hash(x::CenterContract, h::UInt) = hash(x.c, hash(x.b, hash(x.a, hash(:CenterContract, hash(:EuclidianNormalizingFlows, h)))))

(f::CenterContract)(x::Union{Real,AbstractVecOrMat{<:Real}}) = center_contract.(x, f.a, f.b, f.c)

function with_logabsdet_jacobian(f::CenterContract, x::Union{Real,AbstractVecOrMat{<:Real}})
    y = f(x)
    # println("C")
    # println(size(x))
    # println(size(y))
    ladjs = center_contract_ladj.(x, f.a, f.b, f.c)
    (y, sum_ladjs(ladjs))
end

inverse(f::CenterContract) = CenterStretch(f.a, f.b, f.c)
