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

matrix *matrix_map2(long double (*f)(long double, long double), matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = make_matrix(matrix1->rows, matrix1->columns);
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      matrix3->matrix[row][column] = f(matrix1->matrix[row][column], matrix2->matrix[row][column]);
    }
  } return matrix3;
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

matrix *matrix_initialize(long double (*init)(long double), matrix *matrix1) {
  matrix *matrix2 = matrix_copy(matrix1);
  matrix_for_each(init, matrix2);
  return matrix2;
}

matrix *broadcast_vertical(matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = NULL;

  if ((matrix1->rows == 1) && (matrix2->rows > 1)) {
    matrix3 = make_matrix(matrix2->rows, matrix1->columns);

    for (unsigned int row = 0; row < matrix2->rows; row++) {
      for (unsigned int column = 0; column < matrix1->columns; column++) {
        matrix3->matrix[row][column] = matrix1->matrix[0][column];
      }
    } matrix1 = destroy_matrix(matrix1);
      return matrix3;
  }   else return matrix1;
}

matrix *broadcast_horizontal(matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = NULL;
  if ((matrix1->columns == 1) && (matrix2->columns > 1)) {
    matrix3 = make_matrix(matrix1->rows, matrix2->columns);
    for (unsigned int row = 0; row < matrix1->rows; row++) {
      for (unsigned int column = 0; column < matrix2->columns; column++) {
        matrix3->matrix[row][column] = matrix1->matrix[row][0];
      }
    } matrix1 = destroy_matrix(matrix1);
      return matrix3;
  }   else return matrix1;
}

matrix *broadcast_function(long double (*f)(long double, long double), matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = matrix1;
  matrix *matrix4 = matrix2;
  matrix *matrix5 = NULL;

  //assert((matrix3->columns == 1) || (matrix4->columns == 1) || (matrix3->rows == 1) || (matrix4->rows == 1) || (matrix3->columns == matrix4->columns) || (matrix3->rows == matrix4->rows));

  matrix3 = broadcast_horizontal(matrix3, matrix4);
  matrix4 = broadcast_horizontal(matrix4, matrix3);
  matrix3 = broadcast_vertical(matrix3, matrix4);
  matrix4 = broadcast_vertical(matrix4, matrix3);

  matrix5 = matrix_map2(f, matrix3, matrix4);

  matrix3 = destroy_matrix(matrix3);
  matrix4 = destroy_matrix(matrix4);

  return matrix5;
}

matrix *fold(long double (*f)(long double, long double), long double (*init)(long double), unsigned int times, matrix *matrix1, ...) {
  matrix *matrix2 = matrix_initialize(init, matrix1);
  unsigned int time = times - 1;

  va_list args;
  va_start(args, matrix1);

  matrix2 = broadcast_function(f, matrix2, matrix1);

  for (unsigned int t = 0; t < time; t++) {
    matrix2 = broadcast_function(f, matrix2, va_arg(args, matrix *));
  }

  va_end(args);

  return matrix2;
}

long double sum(long double x, long double y) {
  return x + y;
}

long double product(long double x, long double y) {
  return x * y;
}
