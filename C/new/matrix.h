// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>
  #include <stdarg.h>

  #define NOT_USED(x) ((void)x)

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
  matrix *matrix_copy_shape(matrix *);
  matrix *matrix_copy(matrix *);
  matrix *matrix_initialize(long double (*)(long double), matrix *);
  matrix *broadcast_vertical(matrix *, matrix *);
  matrix *broadcast_horizontal(matrix *, matrix *);
  matrix *broadcast_function(long double (*)(long double, long double), matrix *, matrix *);
  matrix *fold(long double (*)(long double, long double), long double (*)(long double), unsigned int, matrix *, ...);
  long double sum(long double, long double);
  long double product(long double, long double);

#endif
