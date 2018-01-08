// matrix.c
#include "matrix.h"

matrix *make_matrix(unsigned int rows, unsigned int columns) {
  matrix *matrix1 = NULL;
  assert((matrix1 = calloc(1, sizeof(matrix))));
  matrix1->rows = rows;
  matrix1->columns = columns;
  assert((matrix1->matrix = calloc(rows, sizeof(long double *))));
  for (unsigned int row = 0; row < rows; row++) {
    assert((matrix1->matrix[row] = calloc(columns, sizeof(long double))));
  } return matrix1;
}

matrix *destroy_matrix(matrix *matrix1) {
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    free(matrix1->matrix[row]);
    matrix1->matrix[row] = NULL;
  } free(matrix1->matrix);
    matrix1->matrix = NULL;
    free(matrix1);
    return NULL;
}

long double random_long_double(long double x) {
  return 2.0 * (((long double)rand()) / ((long double)RAND_MAX)) - 1.0;
}

long double zero(long double x) {return 0.0;}

long double one(long double x) {return 1.0;}

void matrix_for_each(long double (*f)(long double), matrix *matrix1) {
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      matrix1->matrix[row][column] = f(matrix1->matrix[row][column]);
    }
  }
}
