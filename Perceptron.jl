module Perceptron

type Neuron{T<:AbstractFloat}
  eta::T
  n_epoch::Int
  weight::Vector{T}
end

function Neuron(size_input::Int, eta=0.01, n_epoch=10)
  Neuron(eta, n_epoch, zeros(1 + size_input))
end

function _input{T<:AbstractFloat}(model::Neuron, xi::Vector{T})
  dot(xi, model.weight[2:end]) + model.weight[1]
end

function predict{T<:AbstractFloat}(model::Neuron, xi::Vector{T})
  _input(model, xi) >= 0.0 ? 1 : -1
end

function fit(model::Neuron, X::Matrix{Float32}, y::Vector{Int})
  #=
    X: training dataset matrix.
       size is (n_sample, n_feature)
    y: label data vector
  =#
  errors = []
  for epoch in 1:model.n_epoch
    error = 0
    for i in 1:size(X)[1]
      xi = vec(X[i, :])
      target = y[i]
      update = model.eta * (target - predict(model, xi))
      model.weight[2:end] += update * xi
      model.weight[1] += update
      error = (update != 0.0) ? error + 1 : error
    end
    println("epoch $epoch: error = $error")
    append!(errors, [error])
  end
  return errors
end

function load_iris()
  table_iris = readcsv("iris.csv")
  X = table_iris[:, 1:4]
  y = table_iris[:, 5]
  return Array{Float32}(X), y
end

end
