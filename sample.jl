using Gadfly
include("Perceptron.jl")

function plot_data(X, y)
  plt = plot(layer(x=X[1:50, 1], y=X[1:50, 3], Geom.point,
                   Theme(default_color=colorant"blue")),
             layer(x=X[50:end, 1], y=X[50:end, 3], Geom.point,
                   Theme(default_color=colorant"red")),
             Guide.XLabel("sepal length [cm]"),
             Guide.YLabel("petal length [cm]"))
  draw(SVG("iris.svg", 6inch, 3inch), plt)
  nothing
end

X, y = Perceptron.load_iris()
X = X[1:100, :]
y = [yi == "Iris-setosa" ? 1 : -1 for yi in y[1:100]]

# plot_data(X, y)

model = Perceptron.Neuron(size(X)[2], 0.1, 10)
errors = Perceptron.fit(model, X, y)

#=
plt = plot(x=Array(1:size(errors)[1]), y=errors, Geom.line,
           Guide.XLabel("epoch"),
           Guide.YLabel("error"))
draw(SVG("error.svg", 6inch, 3inch), plt)
=#
