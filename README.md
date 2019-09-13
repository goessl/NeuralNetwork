# NeuralNetwork

Very basic neural network implementation to understand & try out machine
learning.
In this repository you can find two neural network classes,
[one without](NeuralNetwork.java) &
[one with biases](NeuralNetworkWithBias.java). Both work mostly the same and
support the same methods so they will referred to as *the neural network*.
I recommend using the bias version for better results.

The neural network class is only used for construction, forward & backward
propagation! To train the network
[sebig3000/Optimization](https://github.com/sebig3000/Optimization) is needed!

## Usage

Construct a new neural network by calling its constructor and specifying its
parameters. These include the number of inputs, the number of nodes in every
layer and the activation functions of every layer & their derivatives.
Common activation functions & their derivatives are already implemented in
[ActivationFunction](ActivationFunction.java).
Now you can already forward datasets through the network & do back propagation.

Input and output is stored in matricies in which every row represents an
dataset and every column gets feed into an node (e.g. element[1][2] is a
parameter of a dataset with index [1] and gets feed into the node with
index [2])

To train the network you create a
[new NeuralNetworkOptimizer](NeuralNetworkOptimizer.java) and give it your
neural network. hen call whatever optimization algorithm you like with your
input & output sets and let the optimizer train the network.
(Adam is recommended)

## Getting Started

**This repository depends on
[sebig3000/Matrix](https://github.com/sebig3000/Matrix)!**
So make sure you have already installed it.
Then you can simply download this repository and add it to your project as a
new package! Done!

## Contributors

The two people who inspired me to try making my own machine learning project
are Brandon Rohrer and Stephen Welch.
Both make awesome YouTube videos that explain how machine learning works.

Stephen Welch:
- YouTube: https://www.youtube.com/user/Taylorns34
- Homepage: http://www.welchlabs.com/
- GitHub: https://github.com/stephencwelch

Brandon Rohrer:
- YouTube https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ
- Blog: https://brohrer.github.io/blog.html
- GitHub: https://github.com/brohrer

## TODO

 - [ ] Fix dependencies
 - [ ] Remove optimization wrappers
 - [ ] Argument validation
 - [x] Documentation

## License (MIT)

MIT License

Copyright (c) 2019 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
