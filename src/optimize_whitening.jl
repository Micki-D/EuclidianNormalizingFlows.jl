# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2


function mvnormal_negll_trafo(trafo::Function, X::AbstractMatrix{<:Real})
    nsamples = size(X, 2) # normalize by number of samples to be independent of batch size:
    
    Y, ladj = with_logabsdet_jacobian(trafo, X)
    #ref_ll = sum(sum(std_normal_logpdf.(Y), dims = 1) .+ ladj) / nsamples
    # Faster:
    ll = (sum(std_normal_logpdf.(Y)) + sum(ladj)) / nsamples
    #@assert ref_ll ≈ ll
    return -ll
end

function mvnormal_negll_trafo_with_logdensity(trafo::Function, X::AbstractMatrix{<:Real}, logdensity_x::AbstractVector)
    nsamples = size(X, 2) # normalize by number of samples to be independent of batch size:
    
    Y, ladj = with_logabsdet_jacobian(trafo, X)

    ll = (sum(std_normal_logpdf.(Y)) + sum(ladj) - sum(logdensity_x)) / nsamples

    return -ll
end

function mvnormal_negll_trafograd(trafo::Function, X::AbstractMatrix{<:Real})
    negll, back = Zygote.pullback(mvnormal_negll_trafo, trafo, X)
    d_trafo = back(one(eltype(X)))[1]
    return negll, d_trafo
end

function mvnormal_negll_trafograd_withld(trafo::Function, X::AbstractMatrix{<:Real}, ld::AbstractVector)
    negll, back = Zygote.pullback(mvnormal_negll_trafo_with_logdensity, trafo, X, ld)
    d_trafo = back(one(eltype(X)))[1]
    return negll, d_trafo
end


function optimize_whitening(
    smpls::VectorOfSimilarVectors{<:Real}, initial_trafo::Function, optimizer;
    nbatches::Integer = 100, nepochs::Integer = 100,
    optstate = Optimisers.setup(optimizer, deepcopy(initial_trafo)),
    negll_history = Vector{Float64}(),
    shuffle_samples::Bool = false
)
    batchsize = round(Int, length(smpls) / nbatches)
    batches = collect(Iterators.partition(smpls, batchsize))
    trafo = deepcopy(initial_trafo)
    state = deepcopy(optstate)
    negll_hist = Vector{Float64}()
    for i in 1:nepochs
        for batch in batches
            X = gpu(flatview(batch))
            negll, d_trafo = mvnormal_negll_trafograd(trafo, X)
            state, trafo = Optimisers.update(state, trafo, d_trafo)
            push!(negll_hist, negll)
        end
        if shuffle_samples
            shuffled_smpls = shuffle(smpls)
            batches = collect(Iterators.partition(shuffled_smpls, batchsize))
        end
    end
    (result = trafo, optimizer_state = state, negll_history = vcat(negll_history, negll_hist))
end
export optimize_whitening

function optimize_whitening_annealing(
    smpls, initial_trafo::Function, 
    nbatches::Integer, 
    nepochs::Integer, 
    learn_start::Real, 
    learn_max::Real, 
    learn_end::Real, 
    phase_durations::AbstractVector;
    negll_history = Vector{Float64}()
)
    batchsize = round(Int, length(smpls) / nbatches)
    trafo = deepcopy(initial_trafo)
    negll_hist = Vector{Float64}()
    state = nothing 

    niterations = nepochs * nbatches

    phase_durations *= niterations 
    phase_durations = round.(Int, phase_durations)

    phase1 = fill(learn_start, phase_durations[1])
    phase2 = [learn_start+i/phase_durations[2]*(learn_max-learn_start) for i in 1:phase_durations[2]]
    phase3 = fill((learn_max-learn_start), phase_durations[3])
    phase4 = [(learn_max-learn_start)-i/phase_durations[4]*learn_end for i in 1:phase_durations[4]]

    learning_rates = vcat(phase1,phase2,phase3,phase4)

    for i in 1:nepochs
        if i == 1 
            state = Optimisers.setup(Optimisers.Adam(learn_start), trafo)
        else
            state = Optimisers.adjust(state, learning_rates[i])
        end

        if i == phase_durations[3]
            trafo.fs[1].freeze_conv = true
            trafo.fs[2].freeze_conv = true
        end

        #println("Training in epoch $i now.")

        shuffled_smpls = shuffle(smpls)
        batches = collect(Iterators.partition(shuffled_smpls, batchsize))
        for batch in batches 
            X = gpu(flatview(unshaped.(batch.v)))
            ld = batch.logd
            negll, d_trafo = EuclidianNormalizingFlows.mvnormal_negll_trafograd_withld(trafo, X, ld)
            state, trafo = Optimisers.update(state, trafo, d_trafo)
            push!(negll_hist, negll)
        end
    
    end

    (result = trafo, optimizer_state = state, negll_history = vcat(negll_history, negll_hist))
end
export optimize_whitening_annealing


function optimize_whitening_stationary(
    rand_dist::Function, 
    initial_trafo::Function, optimizer;
    nbatches::Integer = 2, batchsize::Integer = 2, max_nepochs::Integer = 2, stationary_p_val::Real = 1e-3,
    optstate = Optimisers.setup(optimizer, deepcopy(initial_trafo)),
    negll_history = Vector{Float64}(), wanna_use_GPU::Bool = false
)
    
    trafo = deepcopy(initial_trafo)
    state = deepcopy(optstate)
    negll_hist = Vector{Float64}()
    stationary_flag = true
    nepochs = 0
    
    ### WARMUP ###
    X = rand_dist(2)
    mvnormal_negll_trafograd(trafo, X)
    ##############
    
    while stationary_flag && nepochs < max_nepochs
        @time for i in 1:nbatches
            X = rand_dist(batchsize)
            negll, d_trafo = mvnormal_negll_trafograd(trafo, X)
            state, trafo = Optimisers.update(state, trafo, d_trafo)
            push!(negll_hist, negll)
        end
        nepochs += 1
        p_val = pvalue(ADFTest(cpu(negll_hist)[end-nbatches+1:end], :constant, 1))
        if !(0 <= p_val <= 1)
            println("p_val is $(p_val)")
            println(cpu(negll_hist)[end-nbatches+1:end])
            break
        end
        if p_val < stationary_p_val
            stationary_flag = false
        end
        println("Done epoch $(nepochs), p_val = $(p_val)")
    end
    (result = trafo, optimizer_state = state, negll_history = vcat(negll_history, negll_hist))
end

export optimize_trafo

