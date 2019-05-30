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

package matrix;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;

/**
 * Matrix class used to store and operate on matricies.
 * The indices of the elements are zero indexed.
 * 
 * @author Sebastian Gössl
 * @version 1.0 21.1.2019
 */
public class Matrix implements Iterable<Double> {
  
  /**
   * Dimensions of the matrix.
   * Height: Number of rows
   * Width: Number of columns
   */
  private final int height, width;
  /**
   * Elements of the matrix.
   */
  private final double[][] data;

  /**
   * Constructs a copy of the given matrix.
   * 
   * @param other matrix to copy
   */
  public Matrix(Matrix other) {
    this(other.getHeight(), other.getWidth());
    final PrimitiveIterator.OfDouble iterator = other.iterator();
    set(iterator::nextDouble);
  }
  
  /**
   * Constructs a new matrix with the content of the given array.
   * 
   * @param array data to be stored into the matrix
   */
  public Matrix(double[][] array) {
    this(array.length, array[0].length, (y, x) -> (array[y][x]));
  }
  
  /**
   * Constructs a new matrix with <code>height</code> rows and
   * <code>width</code> columns and fills it with the elements returned from
   * the given supplier.
   * 
   * @param height number of rows
   * @param width number of columns
   * @param supplier supplier to fill the matrix with values
   */
  public Matrix(int height, int width, DoubleSupplier supplier) {
    this(height, width);
    set(supplier);
  }
  
  /**
   * Constructs a new matrix with <code>height</code> rows and
   * <code>width</code> columns and fills the elements with the given
   * function.
   * The function receives the row and column indices of the current element
   * to calculate.
   * 
   * @param height number of rows
   * @param width number of columns
   * @param function function that recieves the indices of the element it
   * shall calculate
   */
  public Matrix(int height, int width,
          ToDoubleBiFunction<Integer, Integer> function) {
    this(height, width);
    set(function);
  }
  
  /**
   * Constructs a new matrix with <code>height</code> rows and
   * <code>width</code> columns.
   * All elements are initialized to zero.
   * 
   * @param height number of rows
   * @param width number of columns
   */
  public Matrix(int height, int width) {
    this.height = height;
    this.width = width;
    
    data = new double[height][width];
  }
  
  
  
  /**
   * Returns the number of rows.
   * 
   * @return number of rows
   */
  public int getHeight() {
    return height;
  }
  
  /**
   * Returns the number of columns.
   * 
   * @return number of columns
   */
  public int getWidth() {
    return width;
  }
  
  /**
   * Returns the element at the specified position.
   * 
   * @param row row of the element to return
   * @param column column of the element to return
   * @return the element at the specified position
   */
  public double get(int row, int column) {
    return data[row][column];
  }
  
  /**
   * Replaces the element at the specified position with the specified
   * element.
   * 
   * @param row row of the element to set
   * @param column column of the element to set
   * @param value element to be stored at the specified position
   * @return element previously at the specified position
   */
  public double set(int row, int column, double value) {
    final double oldElement = data[row][column];
    data[row][column] = value;
    return oldElement;
  }
  
  /**
   * Replaces all elements of the matrix with the specified elements.
   * 
   * @param value element to replace all elements
   */
  public void set(double value) {
    set(() -> (value));
  }
  
  /**
   * Replaces all elements with the values returned from the given supplier.
   * 
   * @param supplier supplier to supply new values for all elements
   */
  public void set(DoubleSupplier supplier) {
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(j, i, supplier.getAsDouble());
      }
    }
  }
  
  /**
   * Replaces all elements with the values returned from the given function.
   * It recieves the position (row and column indices) of the element to
   * replace as arguments.
   * 
   * @param function function to calculate new values for all elements
   */
  public void set(ToDoubleBiFunction<Integer, Integer> function) {
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(j, i, function.applyAsDouble(j, i));
      }
    }
  }
  
  
  /**
   * Adds the given matrix to this matrix elementwise and returns the result.
   * 
   * @param operand other summand
   * @return sum
   */
  public Matrix add(Matrix operand) {
    return apply(operand, Double::sum);
  }
  
  /**
   * Subtracts the given matrix from this matrix elementwise and returns the
   * result.
   * 
   * @param operand subtrahend
   * @return difference
   */
  public Matrix subtract(Matrix operand) {
    return apply(operand, (x, y) -> (x - y));
  }
  
  /**
   * Multiplies every element of this matrix with the given value and returns
   * the result.
   * Scalar multiplication.
   * 
   * @param factor scalar factor
   * @return product
   */
  public Matrix multiply(double factor) {
    return apply((x) -> (factor * x));
  }
  
  /**
   * Matrix multiplies this matrix with the given matrix and returns the
   * result.
   * 
   * @param operand second factor
   * @return product
   */
  public Matrix multiply(Matrix operand) {
    final Matrix result = new Matrix(getHeight(), operand.getWidth(),
            (y, x) -> {
              double sum = 0;
              for(int i=0; i<getWidth() && i<operand.getHeight(); i++) {
                sum += get(y, i) * operand.get(i, x);
              }
              return sum;
            });
    
    return result;
  }
  
  /**
   * Multiplies this matrix with the given matrix elementwise and returns the
   * result.
   * 
   * @param operand factor
   * @return product
   */
  public Matrix multiplyElementwise(Matrix operand) {
    return apply(operand, (x, y) -> (x * y));
  }
  
  /**
   * Divides this matrix by the given matrix elementwise and returns the
   * result.
   * 
   * @param operand divisor
   * @return quotient
   */
  public Matrix divideElementwise(Matrix operand) {
    return apply(operand, (x, y) -> (x / y));
  }
  
  /**
   * Returns the transpose of this matrix.
   * 
   * @return Transpose of this matrix.
   */
  public Matrix transpose() {
    final Matrix result = new Matrix(getWidth(), getHeight(),
            (y, x) -> (get(x, y)));
    
    return result;
  }
  
  
  /**
   * Applies the given operator on every element of this matrix.
   * 
   * @param operator operator to apply on every element of this matrix
   */
  public void forEach(DoubleUnaryOperator operator) {
    final PrimitiveIterator.OfDouble iterator = iterator();
    set(() -> (operator.applyAsDouble(iterator.nextDouble())));
  }
  
  /**
   * Applies the given operator elementwise on every element of this matrix
   * and the given one.
   * 
   * @param operand second operand
   * @param operator operator to apply on every element of this matrix
   */
  public void forEach(Matrix operand, DoubleBinaryOperator operator) {
    final PrimitiveIterator.OfDouble i1 = iterator();
    final PrimitiveIterator.OfDouble i2 = operand.iterator();
    set(() -> (operator.applyAsDouble(i1.nextDouble(), i2.nextDouble())));
  }
  
  /**
   * Applies the given operator on every element of this matrix and returns
   * the result.
   * 
   * @param operator operator to apply on every element of this matrix
   * @return result of the operation
   */
  public Matrix apply(DoubleUnaryOperator operator) {
    final PrimitiveIterator.OfDouble iterator = iterator();
    final Matrix result = new Matrix(getHeight(), getWidth(),
            () -> (operator.applyAsDouble(iterator.nextDouble())));
    
    return result;
  }
  
  /**
   * Applies the given operator elementwise on every element of this matrix
   * and the given one and returns the result.
   * 
   * @param operand second operand
   * @param operator operator to apply on every element of the matricies
   * @return result of the operation
   */
  public Matrix apply(Matrix operand, DoubleBinaryOperator operator) {
    final PrimitiveIterator.OfDouble i1 = iterator();
    final PrimitiveIterator.OfDouble i2 = operand.iterator();
    final Matrix result = new Matrix(getHeight(), getWidth(),
            () -> (operator.applyAsDouble(i1.nextDouble(), i2.nextDouble())));
    
    return result;
  }
  
  /**
   * Applies the given operator on every element of this matrix and the given
   * matrix elementwise wrapping around and returns the result.
   * The result has as many rows and the matrix with more rows and as many
   * columns as the matrix with more columns.
   * 
   * @param operand second operand
   * @param operator operator to apply on every element of the matricies
   * @return result of the operation
   */
  public Matrix applyDifSize(Matrix operand, DoubleBinaryOperator operator) {
    final Matrix result = new Matrix(
            Math.max(getHeight(), operand.getHeight()),
            Math.max(getWidth(), operand.getWidth()),
            (y, x) -> {
              final double value1 = get(y % getHeight(), x % getWidth());
              final double value2 = operand.get(
                      y % operand.getHeight(), x % operand.getWidth());
              return operator.applyAsDouble(value1, value2);
            });
    
    return result;
  }

  /**
   * Fills the matrix with random values
   */
  public void rand() {
    rand(new Random(), -Double.MAX_VALUE, Double.MAX_VALUE);
  }

  /**
   * Fills the matrix with random values,
   * from minimum (inclusive) to maximum (exclusive),
   * given by the random number generator
   * @param rand Random number generator
   * @param minimum Minimum value (inclusive)
   * @param maximum Maximum value (exclusive)
   */
  public void rand(Random rand, double minimum, double maximum) {
    final double range = maximum - minimum;

    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(j, i, range*rand.nextDouble() + minimum);
      }
    }
  }
  
  /**
   * Returns and iterator which iterates over all elements column-row wise.
   * 
   * @return iterator which iterates over all elements column-row wise
   */
  @Override
  public PrimitiveIterator.OfDouble iterator() {
    return new PrimitiveIterator.OfDouble() {
      /**
       * Current position of the iterator.
       */
      private int i=0, j=0;
      
      @Override
      public boolean hasNext() {
        return j<getHeight() && i<getWidth();
      }
      
      @Override
      public double nextDouble() {
        if(!hasNext()) {
          throw new NoSuchElementException();
        }
        
        
        final double element = get(j, i);
        
        if(++i >= getWidth()) {
          i = 0;
          j++;
        }
        
        return element;
      }
    };
  }
  
  /**
   * Returns a copy of this matrix in 2 dimensional array form
   * 
   * @return copy of this matrix in 2 dimensional array form
   */
  public double[][] toArray() {
    final double[][] array = new double[getHeight()][getWidth()];
    
    for(int i=0; i<array.length; i++) {
      System.arraycopy(data[i], 0, array[i], 0, array[i].length);
    }
    
    return array;
  }
  
  /**
   * Returns a string representation of the contents of this matrix.
   * 
   * @return string representation of the contents of this matrix
   */
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    
    for(double[] array : data) {
      builder.append(Arrays.toString(array)).append("\n");
    }
    
    return builder.toString();
  }
}
