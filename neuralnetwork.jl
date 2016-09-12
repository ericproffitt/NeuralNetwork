sigmoid(z::Float64) = 1/(1 + exp(-z))
sigmoidPrime(z::Float64) = sigmoid(z) * (1 - sigmoid(z))

### Types ###

abstract AbstractNode

type Edge
    source::AbstractNode
    target::AbstractNode
    weight::Float64
    derivative::Float64
    augmented::Bool

    Edge(source::AbstractNode, target::AbstractNode) = new(source, target, randn(1,1)[1], 0.0, false)
end

type Node <: AbstractNode
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64
    activationPrime::Float64

    Node() = new([], [], -1.0, -1.0)
end

type InputNode <: AbstractNode
    index::Int
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64

    InputNode(index::Int) = new(index, [], [], -1.0)
end

type BiasNode <: AbstractNode
    incomingEdges::Vector{Edge}
    outgoingEdges::Vector{Edge}
    activation::Float64

    BiasNode() = new([], [], 1.0)
end

type Network
    inputNodes::Vector{InputNode}
    hiddenNodes::Vector{Node}
    outputNodes::Vector{Node}

    function Network(sizes::Array, bias::Bool=true)
        inputNodes = [InputNode(i) for i in 1:sizes[1]];
        hiddenNodes = [Node() for _ in 1:sizes[2]];
        outputNodes = [Node() for _ in 1:sizes[3]];

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

function evaluate(obj::Node, inputVector::Array)
    if obj.activation > -0.5
        return obj.activation
    else
        weightedSum = sum([d.weight * evaluate(d.source, inputVector) for d in obj.incomingEdges])
        obj.activation = sigmoid(weightedSum)
        obj.activationPrime = sigmoidPrime(weightedSum)

        return obj.activation
    end
end

function evaluate(obj::InputNode, inputVector::Array)
    obj.activation = inputVector[obj.index]
    return obj.activation
end

function evaluate(obj::BiasNode, inputVector::Array)
    obj.activation = 1.0
    return obj.activation
end

function updateWeights(obj::AbstractNode, learningRate::Float64)
    for d in obj.incomingEdges
        if d.augmented == false
            d.augmented = true
            d.weight -= learningRate * d.derivative
            updateWeights(d.source, learningRate)
            d.derivative = 0.0
        end
    end
end

function compute(obj::Network, inputVector::Array)
    output = [evaluate(node, inputVector) for node in obj.outputNodes]
    for node in obj.outputNodes
        clear(node)
    end
    return output
end

function clear(obj::AbstractNode)
    for d in obj.incomingEdges
        obj.activation = -1.0
        obj.activationPrime = -1.0
        d.augmented = false
        clear(d.source)
    end
end

function propagateDerivatives(obj::AbstractNode, error::Float64)
    for d in obj.incomingEdges
        if d.augmented == false
            d.augmented = true
            d.derivative += error * obj.activationPrime * d.source.activation
            propagateDerivatives(d.source, error * d.weight * obj.activationPrime)
        end
    end
end

function backpropagation(obj::Network, example::Array)
    output = [evaluate(node, example[1]) for node in obj.outputNodes]
    error = output - example[2]
    for (node, err) in zip(obj.outputNodes, error)
        propagateDerivatives(node, err)
    end

    for node in obj.outputNodes
        clear(node)
    end
end

function train(obj::Network, labeledExamples::Array, learningRate::Float64=0.7, iterations::Int=10000)
    for _ in 1:iterations
        for ex in labeledExamples
            backpropagation(obj, ex)
        end

        for node in obj.outputNodes
            updateWeights(node, learningRate)
        end

        for node in obj.outputNodes
            clear(node)
        end
    end
end


labeledExamples = Array[Array[[0,0,0], [0]],
                        Array[[0,0,1], [1]],
                        Array[[0,1,0], [0]],
                        Array[[0,1,1], [1]],
                        Array[[1,0,0], [0]],
                        Array[[1,0,1], [1]],
                        Array[[1,1,0], [1]],
                        Array[[1,1,1], [0]]];

neuralnetwork = Network([3,4,1])
@time train(neuralnetwork, labeledExamples)
