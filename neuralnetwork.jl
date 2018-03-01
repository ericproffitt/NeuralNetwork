sigmoid(z::Float64) = 1 / (1 + exp(-z))
sigmoidPrime(z::Float64) = sigmoid(z) * (1 - sigmoid(z))


### Types ###

abstract type AbstractNode end

mutable struct Edge
    source::AbstractNode
    target::AbstractNode
    weight::Float64
    derivative::Float64
    augmented::Bool

    Edge(source::AbstractNode, target::AbstractNode) = new(source, target, randn(1,1)[1], 0.0, false)
end

mutable struct Node <: AbstractNode
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64
    activationPrime::Float64

    Node() = new([], [], -1.0, -1.0)
end

mutable struct InputNode <: AbstractNode
    index::Int
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64

    InputNode(index::Int) = new(index, [], [], -1.0)
end

mutable struct BiasNode <: AbstractNode
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64

    BiasNode() = new([], [], 1.0)
end

struct Network
    inputNodes::Vector{InputNode}
    hiddenNodes::Vector{Node}
    outputNodes::Vector{Node}

    function Network{T<:Integer}(sizes::Vector{T}, bias::Bool=true)
        inputNodes = [InputNode(i) for i in 1:sizes[1]]
        hiddenNodes = [Node() for _ in 1:sizes[2]]
        outputNodes = [Node() for _ in 1:sizes[3]]

        for inputNode in inputNodes
            for node in hiddenNodes
                edge = Edge(inputNode, node);
                push!(inputNode.outgoingEdges, edge)
                push!(node.incomingEdges, edge)
            end
        end

        for node in hiddenNodes
            for outputNode in outputNodes
                edge = Edge(node, outputNode);
                push!(node.outgoingEdges, edge)
                push!(outputNode.incomingEdges, edge)
            end
        end

        if bias == true
            biasNode = BiasNode()
            for node in hiddenNodes
                edge = Edge(biasNode, node);
                push!(biasNode.outgoingEdges, edge)
                push!(node.incomingEdges, edge)
            end
        end

        new(inputNodes, hiddenNodes, outputNodes)
    end
end


### Methods ###

function evaluate(node::Node, inputVector::Vector{Float64})
    if node.activation > -0.5
        return node.activation
    else
        weightedSum = sum([d.weight * evaluate(d.source, inputVector) for d in node.incomingEdges])
        node.activation = sigmoid(weightedSum)
        node.activationPrime = sigmoidPrime(weightedSum)

        return node.activation
    end
end

function evaluate(node::InputNode, inputVector::Vector{Float64})
    node.activation = inputVector[node.index]
    return node.activation
end

function evaluate(node::BiasNode, inputVector::Vector{Float64})
    node.activation = 1.0
    return node.activation
end

function updateWeights(node::AbstractNode, learningRate::Real)
    for d in node.incomingEdges
        if d.augmented == false
            d.augmented = true
            d.weight -= learningRate * d.derivative
            updateWeights(d.source, learningRate)
            d.derivative = 0.0
        end
    end
end

function compute{T<:Real}(network::Network, inputVector::Vector{T})
    inputVector = float(inputVector)
    
    output = [evaluate(node, inputVector) for node in network.outputNodes]
    for node in network.outputNodes
        clear(node)
    end
    return output
end

function clear(node::AbstractNode)
    for d in node.incomingEdges
        node.activation = -1.0
        node.activationPrime = -1.0
        d.augmented = false
        clear(d.source)
    end
end

function propagateDerivatives(node::AbstractNode, err::Real)
    for d in node.incomingEdges
        if d.augmented == false
            d.augmented = true
            d.derivative += err * node.activationPrime * d.source.activation
            propagateDerivatives(d.source, err * d.weight * node.activationPrime)
        end
    end
end

function backpropagation(network::Network, example::Vector{Vector{Float64}})
    output = [evaluate(node, example[1]) for node in network.outputNodes]
    errors = output - example[2]
    for (node, err) in zip(network.outputNodes, errors)
        propagateDerivatives(node, err)
    end

    for node in network.outputNodes
        clear(node)
    end
end

function train{T<:Real}(network::Network, labeledExamples::Vector{Vector{Vector{T}}}, learningRate::Real=0.7, iterations::Integer=10000)
    labeledExamples = [[float(ex[1]), float(ex[2])] for ex in labeledExamples]

    for _ in 1:iterations
        for ex in labeledExamples
            backpropagation(network, ex)
        end

        for node in network.outputNodes
            updateWeights(node, learningRate)
        end

        for node in network.outputNodes
            clear(node)
        end
    end
    nothing
end


### Test ###

function test()
    labeledExamples =  [[[0,0,0], [0]],
                        [[0,0,1], [1]],
                        [[0,1,0], [0]],
                        [[0,1,1], [1]],
                        [[1,0,0], [0]],
                        [[1,0,1], [1]],
                        [[1,1,0], [1]],
                        [[1,1,1], [0]]]

    neuralnetwork = Network([3,4,1])
    train(neuralnetwork, labeledExamples)
    println("training accuracy = ", sum([round.(compute(neuralnetwork, ex[1])) == ex[2] for ex in labeledExamples]) / length(labeledExamples))
    nothing
end
