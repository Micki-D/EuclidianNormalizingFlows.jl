# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQSBlock <: Function
    nn::Chain
    mask::AbstractVector
end

export CouplingRQSBlock
@functor CouplingRQSBlock

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSBlock,
    x::Any
)
    return rqs_trafo!(f, x)
end

(f::CouplingRQSBlock)(x::Any) = rqs_trafo!(f, x)[1]
(f::CouplingRQSBlock)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CouplingRQSBlock)
    return CouplingRQSBlockInverse(f.nn, f.mask)
end


struct CouplingRQSBlockInverse <: Function
    nn::Chain
    mask::AbstractVector
end

export CouplingRQSBlockInverse
@functor CouplingRQSBlockInverse

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSBlockInverse,
    x::Any
)
    return rqs_trafo!(f, x)
end

(f::CouplingRQSBlockInverse)(x::Any) = rqs_trafo!(f, x)[1]
(f::CouplingRQSBlockInverse)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CouplingRQSBlockInverse)
    return CouplingRQSBlock(f.nn, f.mask)
end


function rqs_trafo!(trafo::Union{CouplingRQSBlock, CouplingRQSBlockInverse}, x::Any)

    spline = trafo isa CouplingRQSBlock ? RQSpline : RQSplineInv
    dims_tt = sum(trafo.mask)

    #Experimental disintegration theorem approach: 
    #t_dim = findall(x->x, trafo.mask)[1]
    #input_mask = [i < t_dim ? trafo.mask[i] : ~trafo.mask[i] for i in 1:length(trafo.mask)]
    #pure musketeer:
    input_mask = .~trafo.mask 
    y, ladj = with_logabsdet_jacobian(spline(get_params(trafo.nn(x[input_mask,:]), dims_tt)...), x[trafo.mask,:])   

    return _sort_dimensions(y, x, trafo.mask), ladj
end

export rqs_trafo!
