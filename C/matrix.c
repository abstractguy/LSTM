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
  matrix *matrix3 = make_matrix(matrix1->rows, matrix2->columns);
  assert(matrix1->columns == matrix2->rows);
  matrix_for_each(zero, matrix3);

  for (unsigned int row1 = 0; row1 < matrix1->rows; row1++) {
    for (unsigned int column2 = 0; column2 < matrix2->columns; column2++) {
      for (unsigned int column1 = 0; column1 < matrix1->columns; column1++) {
        matrix3->matrix[row1][column2] += matrix1->matrix[row1][column1] * matrix2->matrix[column1][column2];
      }
    }
  } matrix1 = destroy_matrix(matrix1);
    matrix2 = destroy_matrix(matrix2);
    return matrix3;
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

long double sigmoid_derivative_helper(long double x) {
  long double y = sigmoid(x);
  return y * (1.0 - y);
}

long double tanh_derivative_helper(long double x) {
  long double y = tanhl(x);
  return 1.0 - y * y;
}

matrix *sigmoid_derivative(matrix *matrix1) {
  matrix_for_each(sigmoid_derivative_helper, matrix1);
  return matrix1;
}

matrix *tanh_derivative(matrix *matrix1) {
  matrix_for_each(tanh_derivative_helper, matrix1);
  return matrix1;
}

matrix *broadcast_function(long double (*f)(long double, long double), matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = NULL;
  unsigned int rows1    = matrix1->rows,
               columns1 = matrix1->columns,
               rows2    = matrix2->rows,
               columns2 = matrix2->columns,
               rows3, columns3, A, B, C, D;

  A = rows1    == 1;
  B = columns1 == 1;
  C = rows2    == 1;
  D = columns2 == 1;

  assert(rows1    == rows2    || A || C);
  assert(columns1 == columns2 || B || D);

  rows3    = A ? rows2    : rows1;
  columns3 = B ? columns2 : columns1;
  matrix3  = make_matrix(rows3, columns3);

  for (unsigned int row = 0; row < rows3; row++) {
    for (unsigned int column = 0; column < columns3; column++) {
      matrix3->matrix[row][column] =
        f(matrix1->matrix[A ? 0 : row][B ? 0 : column],
          matrix2->matrix[C ? 0 : row][D ? 0 : column]);
    }
  }

  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);

  return matrix3;
}

long double add(long double x, long double y) {return x + y;}
long double multiply(long double x, long double y) {return x * y;}
long double minus(long double x, long double y) {return x - y;}

matrix *fold(unsigned int time, long double (*f)(long double, long double), matrix *matrix1, ...) {
  va_list args;
  if (time > 1) {
    va_start(args, matrix1);
    for (unsigned int n = 1; n < time; n++) {
      matrix1 = broadcast_function(f, matrix1, va_arg(args, matrix *));
    } va_end(args);
  }   return matrix1;
}

matrix *transpose(matrix *matrix1) {
  matrix *matrix2 = make_matrix(matrix1->columns, matrix1->rows);

  for (unsigned int row = 0; row < matrix1->columns; row++) {
    for (unsigned int column = 0; column < matrix1->rows; column++) {
      matrix2->matrix[row][column] = matrix1->matrix[column][row];
    }
  }

  matrix1 = destroy_matrix(matrix1);
  return matrix2;
}

matrix *apply_vertically(long double (*f)(long double, long double), matrix *matrix1) {
  matrix *matrix2 = NULL;
  if (matrix1->columns == 1) return matrix1;
  else {
    matrix2 = make_matrix(matrix1->rows, 1);
    for (unsigned int row = 0; row < matrix1->rows; row++) {
      matrix2->matrix[row][0] = matrix1->matrix[row][0];
    }
    for (unsigned int column = 1; column < matrix1->columns; column++) {
      for (unsigned int row = 0; row < matrix1->rows; row++) {
        matrix2->matrix[row][0] = f(matrix2->matrix[row][0], matrix1->matrix[row][column]);
      }
    } matrix1 = destroy_matrix(matrix1);
      return matrix2;
  }
}

matrix *fold_vertically(unsigned int time, long double (*f)(long double, long double), matrix *matrix1, ...) {
  matrix *matrix2 = NULL;
  va_list args;
  matrix1 = apply_vertically(f, matrix1);
  if (time > 1) {
    va_start(args, matrix1);
    for (unsigned int n = 1; n < time; n++) {
      matrix2 = apply_vertically(f, va_arg(args, matrix *));
      for (unsigned int row = 0; row < matrix1->rows; row++) {
        matrix1->matrix[row][0] =
          f(matrix1->matrix[row][0],
            matrix2->matrix[row][0]);
      } matrix2 = destroy_matrix(matrix2);
    }   va_end(args);
  }     return matrix1;
}

void matrix_push_all(matrix *matrix1, long double *array) {
  unsigned int rows    = matrix1->rows,
               columns = matrix1->columns;
  for (unsigned int row = 0; row < rows; row++) {
    for (unsigned int column = 0; column < columns; column++) {
      matrix1->matrix[row][column] = array[row * columns + column];
    }
  }
}
