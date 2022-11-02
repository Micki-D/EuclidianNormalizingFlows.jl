# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    EuclidianNormalizingFlows

Euclidian normalizing flows.
"""
module EuclidianNormalizingFlows

using LinearAlgebra
using Random
using Statistics

using ArgCheck
using ArraysOfArrays
using ChainRulesCore
using ChangesOfVariables
using Distributions
using DocStringExtensions
using ElasticArrays
using Flux
using ForwardDiffPullbacks
using FunctionChains
using Functors
using InverseFunctions
using Optim
using Optimisers
using Parameters
using SpecialFunctions
using StatsBase
using ValueShapes
using KernelAbstractions
using KernelAbstractions: @atomic

import Zygote
import ZygoteRules

using Distributions: log2π

import InverseFunctions.inverse
import ChainRulesCore.rrule
import ChainRulesCore.ignore_derivatives
import ChangesOfVariables.with_logabsdet_jacobian
using Base.Threads: @threads
using CUDA
using CUDAKernels
using HypothesisTests


include("abstract_trafo.jl")
include("optimize_whitening.jl")
include("householder_trafo.jl")
include("scale_shift_trafo.jl")
include("center_stretch.jl")
include("johnson_trafo.jl")
include("spline_trafo.jl")
include("coupling_rqs.jl")
include("utils.jl")
#include("test_spline_pullback.jl")

end # module
