// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>
  #include <stdarg.h>

  #define NOT_USED(x) ((void)x)

  #define sum(x, y)      broadcast_function(add,      x, y)
  #define product(x, y)  broadcast_function(multiply, x, y)
  #define subtract(x, y) broadcast_function(minus,    x, y)

  typedef struct {
    unsigned int rows, columns;
    long double **matrix;
  } matrix_type;

  matrix_type *make_matrix(unsigned int, unsigned int);
  void destroy_matrix(matrix_type *);
  long double random_long_double(long double);
  void matrix_for_each(long double (*)(long double), matrix_type *);
  matrix_type *dot_product(matrix_type *, matrix_type *);
  matrix_type *matrix_copy(matrix_type *);
  long double sigmoid(long double);
  matrix_type *matrix_sigmoid(matrix_type *);
  long double sigmoid_derivative_helper(long double);
  matrix_type *sigmoid_derivative(matrix_type *);
  matrix_type *broadcast_function(long double (*)(long double, long double), matrix_type *, matrix_type *);
  long double add(long double, long double);
  long double multiply(long double, long double);
  long double minus(long double, long double);
  matrix_type *transpose(matrix_type *);
  void matrix_push_all(matrix_type *, long double *);
#endif
