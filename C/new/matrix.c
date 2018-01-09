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
  NOT_USED(x);
  return 2.0 * (((long double)rand()) / ((long double)RAND_MAX)) - 1.0;
}

long double zero(long double x) {NOT_USED(x); return 0.0;}

long double one(long double x) {NOT_USED(x); return 1.0;}

void matrix_for_each(long double (*f)(long double), matrix *matrix1) {
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      matrix1->matrix[row][column] = f(matrix1->matrix[row][column]);
    }
  }
}

matrix *dot_product(matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = make_matrix(matrix1->columns, matrix2->rows);
  assert(matrix1->rows == matrix2->columns);
  matrix_for_each(zero, matrix3);
  for (unsigned int row1 = 0; row1 < matrix1->rows; row1++) {
    for (unsigned int column2 = 0; column2 < matrix2->columns; column2++) {
      for (unsigned int column1 = 0; column1 < matrix1->columns; column1++) {
        matrix3->matrix[row1][column2] += matrix1->matrix[row1][column1] * matrix2->matrix[column1][column2];
      }
    }
  } return matrix3;
}
