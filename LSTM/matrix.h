// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>
  #include <stdarg.h>

  #define NOT_USED(x) ((void)x)

  #define sum(n, matrix1, ...)      fold(n, add, matrix1, __VA_ARGS__)
  #define product(n, matrix1, ...)  fold(n, multiply, matrix1, __VA_ARGS__)
  #define subtract(n, matrix1, ...) fold(n, minus, matrix1, __VA_ARGS__)

  typedef struct {
    unsigned int rows, columns;
    long double **matrix;
  } matrix_type;

  matrix_type *make_matrix(unsigned int, unsigned int);
  void destroy_matrix(matrix_type *);
  long double random_long_double(long double);
  long double zero(long double);
  long double one(long double);
  void matrix_for_each(long double (*)(long double), matrix_type *);
  matrix_type *dot_product(matrix_type *, matrix_type *);
  matrix_type *matrix_copy_shape(matrix_type *);
  matrix_type *matrix_copy(matrix_type *);
  long double sigmoid(long double);
  matrix_type *matrix_sigmoid(matrix_type *);
  matrix_type *matrix_tanh(matrix_type *);
  long double sigmoid_derivative_helper(long double);
  long double tanh_derivative_helper(long double);
  matrix_type *sigmoid_derivative(matrix_type *);
  matrix_type *tanh_derivative(matrix_type *);
  matrix_type *broadcast_function(long double (*)(long double, long double), matrix_type *, matrix_type *);
  matrix_type *fold(unsigned int, long double (*)(long double, long double), matrix_type *, ...);
  long double add(long double, long double);
  long double multiply(long double, long double);
  long double minus(long double, long double);
  matrix_type *transpose(matrix_type *);
#endif
