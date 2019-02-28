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

import java.util.function.DoubleUnaryOperator;



/**
 * Common activation functions & their derivatives used in neural networks.
 * 
 * @author Sebastian Gössl
 * @version 1.0 3.3.2019
 */
public enum ActivationFunction {
  
  /**
   * Enums.
   */
  IDENTITY, TANH, SIGMOID, RELU, SOFTPLUS, RELU_LEAKY;
  
  
  
  /**
   * Leaky ReLU leakage
   */
  private static final double RELU_LEAKY_LEAKAGE = 0.01;
  
  
  /**
   * Functions.
   */
  private static DoubleUnaryOperator[] foos = new DoubleUnaryOperator[] {
    //Identity
    (x) -> (x),
    //Tanh
    (x) -> (Math.tanh(x)),
    //Sigmoid
    (x) -> (1 / (1 + Math.exp(-x))),
    //ReLU
    (x) -> {
      if(x >= 0) {
        return x;
      } else {
        return 0.0;
      }},
    //SoftPlus
    (x) -> (Math.log(1 + Math.exp(x))),
    //Leaky ReLU
    (x) -> {
      if(x >= 0) {
        return x;
      } else {
        return RELU_LEAKY_LEAKAGE * x;
      }}
  };
  
  /**
   * Derivatives.
   */
  private static DoubleUnaryOperator[] primes = new DoubleUnaryOperator[] {
    //Identity
    (x) -> (1),
    //Tanh
    (x) -> (1 - Math.tanh(x) * Math.tanh(x)),
    //Sigmoid
    (x) -> (Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)))),
    //ReLU
    (x) -> {
      if(x >= 0) {
        return 1.0;
      } else {
        return 0.0;
      }},
    //Softplus
    (x) -> (1 / (1 + Math.exp(-x))),
    //Leaky ReLU
    (x) -> {
      if(x >= 0) {
        return  1.0;
      } else {
        return RELU_LEAKY_LEAKAGE;
      }}
  };
  
  /**
   * Names.
   */
  private static final String[] names = {
    "Identity",
    "Hyperbolic tangent",
    "Sigmoid",
    "Rectified linear unit",
    "SoftPlus",
    "Leaky rectified linear unit"
  };
  
  
  
  /**
   * Returns the function as DoubleUnaryOperator.
   * 
   * @return function as DoubleUnaryOperator;
   */
  public DoubleUnaryOperator get() {
    return foos[ordinal()];
  }
  
  /**
   * Returns the derivative as DoubleUnaryOperator.
   * 
   * @return derivative as DoubleUnaryOperator;
   */
  public DoubleUnaryOperator getPrime() {
    return primes[ordinal()];
  }
  
  
  
  @Override
  public String toString() {
    return names[ordinal()];
  }
}
