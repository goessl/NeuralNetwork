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



package optimization;



/**
 * Interface used for mathematical optimizable classes.
 * 
 * @author Sebastian Gössl
 * @version 1.0 3.3.2019
 * @param <T> Input & output type
 */
public interface Optimizable<T> {
  
  /**
   * Returns all parameters arranged to an array.
   * 
   * @return parameters arranged to an array
   */
  double[] getParameters();
  
  /**
   * Replaces all parameters with the given ones.
   * 
   * @param params new parameters
   */
  void setParameters(double[] params);
  
  
  /**
   * Returns the error of this objects outputs compared to the given outputs.
   * 
   * @param input input
   * @param output output
   * @return error of this objects outputs compared to the given outputs
   */
  double cost(T[] input, T[] output);
  
  /**
   * Returns the derivative of the cost with respect to every parameter.
   * 
   * @param input input
   * @param output wanted output
   * @return derivative of the cost with respect to every parameter
   */
  double[] costPrime(T[] input, T[] output);
}
