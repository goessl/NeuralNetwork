/*
 * MIT License
 * 
 * Copyright (c) 2019 Sebastian Gössl
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */



package neuralnetwork;

import java.util.ArrayList;
import matrix.Matrix;
import java.util.Arrays;
import java.util.List;
import java.util.PrimitiveIterator;
import java.util.function.Consumer;
import optimization.Optimizable;



/**
 * Wrapper class for NeuralNetwork to be optimizable.
 * Wraps & unwraps the weight matricies to parameter arrays and parameter
 * arrays to the weight matricies.
 * 
 * @author Sebastian Gössl
 * @version 1.1 12.9.2019
 */
public class NeuralNetworkOptimizable implements Optimizable<double[]> {
    
    /**
     * NeuralNetwork to wrap.
     */
    private final NeuralNetwork net;
    
    
    
    /**
     * Constructs a new NeuralNetworkOptimizable with the given NeuralNetwork.
     * 
     * @param net NeuralNetwork to wrap
     */
    public NeuralNetworkOptimizable(NeuralNetwork net) {
        this.net = net;
    }
    
    
    
    /**
     * Unwraps the elements of a matrix array to a primitive double array.
     * 
     * @param matricies matricies to be unwrapped
     * @return elements of the matricies in an primitive double array
     */
    private double[] matriciesToArray(Matrix[] matricies) {
        final List<Double> list = new ArrayList<>();
        for(Matrix matrix : matricies) {
            matrix.forEach((Consumer<Double>)list::add);
        }
        
        return list.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    
    
    @Override
    public double[] getParameters() {
        return matriciesToArray(net.getWeights());
    }
    
    @Override
    public void setParameters(double[] params) {
        final PrimitiveIterator.OfDouble iterator =
                Arrays.stream(params).iterator();
        
        for(Matrix weight : net.getWeights()) {
            weight.set(() -> (iterator.nextDouble()));
        }
    }
    
    
    @Override
    public double cost(double[][] input, double[][] output) {
        return net.cost(new Matrix(input), new Matrix(output));
    }
    
    @Override
    public double[] costPrime(double[][] input, double[][] output) {
        return matriciesToArray(
                net.costPrime(new Matrix(input), new Matrix(output)));
    }
}
