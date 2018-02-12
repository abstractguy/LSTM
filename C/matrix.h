// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>
  #include <stdarg.h>

  #define NOT_USED(x) ((void)x)

/*
  #define sum(n, matrix1, ...) \
    fold_vertically(n, add, matrix1, __VA_ARGS__)
  #define product(n, matrix1, ...) \
    fold_vertically(n, multiply, matrix1, __VA_ARGS__)
  #define subtract(n, matrix1, ...) \
    fold_vertically(n, minus, matrix1, __VA_ARGS__)
*/

  #define sum(n, matrix1, ...) \
    fold(n, add, matrix1, __VA_ARGS__)
  #define product(n, matrix1, ...) \
    fold(n, multiply, matrix1, __VA_ARGS__)
  #define subtract(n, matrix1, ...) \
    fold(n, minus, matrix1, __VA_ARGS__)

  typedef struct {
    unsigned int rows, columns;
    long double **matrix;
  } matrix;

  matrix *make_matrix(unsigned int, unsigned int);
  matrix *destroy_matrix(matrix *);
  long double random_long_double(long double);
  long double zero(long double);
  long double one(long double);
  void matrix_for_each(long double (*)(long double), matrix *);
  matrix *dot_product(matrix *, matrix *);
  matrix *matrix_copy(matrix *);
  matrix *matrix_initialize(long double (*)(long double), unsigned int, unsigned int);
  matrix *matrix_initialize_from_matrix(long double (*)(long double), matrix *);
  long double sigmoid(long double);
  matrix *matrix_sigmoid(matrix *);
  matrix *matrix_tanh(matrix *);
  long double sigmoid_derivative_helper(long double);
  long double tanh_derivative_helper(long double);
  matrix *sigmoid_derivative(matrix *);
  matrix *tanh_derivative(matrix *);
  matrix *broadcast_function(long double (*)(long double, long double), matrix *, matrix *);
  matrix *fold(unsigned int, long double (*)(long double, long double), matrix *, ...);
  long double add(long double, long double);
  long double multiply(long double, long double);
  long double minus(long double, long double);
  matrix *transpose(matrix *);
  //matrix *apply_vertically(long double (*)(long double, long double), matrix *);
  //matrix *fold_vertically(unsigned int, long double (*)(long double, long double), matrix *, ...);
  void matrix_push_all(matrix *, long double *);

#endif
