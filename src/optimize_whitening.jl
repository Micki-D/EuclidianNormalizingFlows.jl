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


function mvnormal_negll_trafograd(trafo::Function, X::AbstractMatrix{<:Real})
    negll, back = Zygote.pullback(mvnormal_negll_trafo, trafo, X)
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
            X = flatview(batch)
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


function optimize_whitening_ann(
    smpls::VectorOfSimilarVectors{<:Real}, initial_trafo::Function, annealing_steps::Vector;
    nbatches::Integer = 100, nepochs::Integer = 100,
    negll_history = Vector{Float64}(),
    shuffle_samples::Bool = false    
)
    batchsize = round(Int, length(smpls) / nbatches)
    batches = collect(Iterators.partition(smpls, batchsize))
    trafo = deepcopy(initial_trafo)
    optstate = Optimisers.setup(Optimisers.Adam(annealing_steps[1]), deepcopy(initial_trafo))
    state = deepcopy(optstate)
    negll_hist = Vector{Float64}()

    negll_hist_mean = Vector{Float64}()

    step_count = 2

    for i in 1:nepochs
        # if i >3
        #     mc_diff_1 = negll_hist_mean[end-3] - negll_hist_mean[end-2]
        #     mc_diff_2 = negll_hist_mean[end-2] - negll_hist_mean[end-1]

        #     if mc_diff_1/mc_diff_2<1.5
        #         optstate = Optimisers.setup(Optimisers.Adam(annealing_steps[step_count]), deepcopy(initial_trafo))
        #         state = deepcopy(optstate)
        #         step_count+=1
        #     end
        # end
        if i%50==0 && step_count <= length(annealing_steps)
            optstate = Optimisers.setup(Optimisers.Adam(annealing_steps[step_count]), trafo)
            state = deepcopy(optstate)
            println("using $(annealing_steps[step_count]) for stepsize")
            step_count+=1
        end

        for batch in batches
            X = flatview(batch)
            negll, d_trafo = mvnormal_negll_trafograd(trafo, X)
            state, trafo = Optimisers.update(state, trafo, d_trafo)
            push!(negll_hist, negll)
        end

        # mean_cost = mean(negll_hist[end-nbatches+1:end])
        # push!(negll_hist_mean, mean_cost)

        if shuffle_samples
            shuffled_smpls = shuffle(smpls)
            batches = collect(Iterators.partition(shuffled_smpls, batchsize))
        end
    end
    (result = trafo, optimizer_state = state, negll_history = vcat(negll_history, negll_hist), negll_hist_mean = negll_hist_mean)
end
export optimize_whitening_ann
