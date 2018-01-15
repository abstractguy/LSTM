// matrix.c
#include "matrix.h"

matrix *make_matrix(unsigned int rows, unsigned int columns) {
  matrix *matrix1 = NULL;
  matrix1 = calloc(1, sizeof(matrix));
  assert(matrix1);
  matrix1->rows = rows;
  matrix1->columns = columns;
  matrix1->matrix = calloc(rows, sizeof(long double *));
  assert(matrix1->matrix);
  for (unsigned int row = 0; row < rows; row++) {
    matrix1->matrix[row] = calloc(columns, sizeof(long double));
    assert(matrix1->matrix[row]);
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

matrix *matrix_copy_shape(matrix *matrix1) {
  return make_matrix(matrix1->rows, matrix1->columns);
}

matrix *matrix_copy(matrix *matrix1) {
  matrix *matrix2 = matrix_copy_shape(matrix1);

  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      matrix2->matrix[row][column] = matrix1->matrix[row][column];
    }
  } return matrix2;
}

matrix *matrix_initialize(long double (*init)(long double), unsigned int x, unsigned int y) {
  matrix *matrix1 = make_matrix(x, y);
  matrix_for_each(init, matrix1);
  return matrix1;
}

matrix *matrix_initialize_from_matrix(long double (*init)(long double), matrix *matrix1) {
  matrix *matrix2 = matrix_copy(matrix1);
  matrix_for_each(init, matrix2);
  return matrix2;
}

long double sigmoid(long double x) {
  return 1.0 / (1.0 + (long double)expl(-x));
}

matrix *matrix_sigmoid(matrix *matrix1) {
  matrix_for_each(sigmoid, matrix1);
  return matrix1;
}

matrix *matrix_tanh(matrix *matrix1) {
  matrix_for_each(tanhl, matrix1);
  return matrix1;
}

long double sigmoid_derivative(long double x) {
  long double y = sigmoid(x);
  return y * (1.0 - y);
}

long double tanh_derivative(long double x) {
  long double y = tanhl(x);
  return 1.0 - y * y;
}
