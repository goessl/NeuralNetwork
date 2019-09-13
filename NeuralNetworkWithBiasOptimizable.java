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
import optimization.Optimizable;



/**
 * Wrapper class for NeuralNetworkWithBias to be optimizable.
 * Wraps & unwraps the weight & bias matricies to parameter arrays and
 * parameter arrays to the weight & bias matricies.
 * 
 * @author Sebastian Gössl
 * @version 1.2 13.9.2019
 */
public class NeuralNetworkWithBiasOptimizable
        implements Optimizable<double[]> {
    
    /**
     * NeuralNetworkWithBias to wrap.
     */
    private final NeuralNetworkWithBias net;
    /**
     * References to the weight & bias matricies in alternating order.
     * weights[0], biases[0], weights[1], ...
     */
    private final Matrix[] matricies;
    
    
    
    /**
     * Constructs a new NeuralNetworkWithBiasOptimizable with the given
     * NeuralNetworkWithBias.
     * 
     * @param net NeuralNetworkWithBias to wrap
     */
    public NeuralNetworkWithBiasOptimizable(NeuralNetworkWithBias net) {
        this.net = net;
        
        
        matricies = new Matrix[2 * net.getNumberOfLayers()];
        
        for(int i=0; i<net.getNumberOfLayers(); i++) {
            matricies[2*i] = net.getWeights(i);
            matricies[2*i + 1] = net.getBiases(i);
        }
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
            matrix.forEach(list::add);
        }
        
        return list.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    
    
    @Override
    public double[] getParameters() {
        return matriciesToArray(matricies);
    }
    
    @Override
    public void setParameters(double[] params) {
        final PrimitiveIterator.OfDouble iterator =
                Arrays.stream(params).iterator();
        
        for(Matrix matrix : matricies) {
            matrix.set(() -> iterator.nextDouble());
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
