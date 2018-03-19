// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdio.h>
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>

  #define sum(x, y)      broadcast(add,      x, y)
  #define product(x, y)  broadcast(multiply, x, y)
  #define subtract(x, y) broadcast(minus,    x, y)

  typedef struct {
    unsigned int rows, columns;
    long double **matrix;
  } matrix_type;

  matrix_type *make_matrix(unsigned int, unsigned int);
  void destroy_matrix(matrix_type *);
  matrix_type *dot_product(matrix_type *, matrix_type *);
  matrix_type *matrix_copy(matrix_type *);
  long double sigmoid_double(long double);
  matrix_type *sigmoid(matrix_type *);
  matrix_type *sigmoid_derivative(matrix_type *);
  matrix_type *broadcast(long double (*)(long double, long double), 
                         matrix_type *, matrix_type *);
  long double add(long double, long double);
  long double multiply(long double, long double);
  long double minus(long double, long double);
  matrix_type *transpose(matrix_type *);
  void print_matrix(char *, matrix_type *);
  void matrix_push_all(char *, matrix_type *, long double *);
#endif
