// matrix.h
#ifndef MATRIX_H
  #define MATRIX_H
  #include <stdlib.h>
  #include <assert.h>
  #include <math.h>

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
  matrix *matrix_map2(long double (*)(long double, long double), matrix *, matrix *);
  matrix *dot_product(matrix *, matrix *);
  matrix *matrix_copy_shape(matrix *);
  matrix *matrix_copy(matrix *);
  matrix *matrix_initialize(long double (*)(long double), matrix *);
  long double add(long double, long double);
  long double multiply(long double, long double);

#endif
