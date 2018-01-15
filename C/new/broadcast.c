// broadcast.c
#include "broadcast.h"

matrix *matrix_map2(long double (*f)(long double, long double), matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = make_matrix(matrix1->rows, matrix1->columns);
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      matrix3->matrix[row][column] = f(matrix1->matrix[row][column], matrix2->matrix[row][column]);
    }
  } return matrix3;
}

matrix *broadcast_vertical(matrix *matrix1, matrix *matrix2) {
  matrix *matrix3 = NULL;

  if (matrix1->rows == 1 && matrix2->rows > 1) {
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
  if (matrix1->columns == 1 && matrix2->columns > 1) {
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

  return matrix5;
}

matrix *fold(long double (*f)(long double, long double), long double (*init)(long double), unsigned int times, matrix *matrix1, va_list args) {
  matrix *matrix2 = matrix_initialize_from_matrix(init, matrix1);
  //matrix *matrix3 = NULL;
  unsigned int time = times - 1;

  matrix2 = broadcast_function(f, matrix2, matrix1);

  for (unsigned int t = 0; t < time; t++) {
    //matrix3 = va_arg(args, matrix *);
    //matrix2 = broadcast_function(f, matrix2, matrix3);
    //matrix3 = destroy_matrix(matrix3);
    matrix2 = broadcast_function(f, matrix2, va_arg(args, matrix *));
  }

  //matrix1 = destroy_matrix(matrix1);
  return matrix2;
}

long double add(long double x, long double y) {return x + y;}
long double multiply(long double x, long double y) {return x * y;}

matrix *sum(unsigned int n, matrix *matrix1, ...) {
  matrix *matrix2 = NULL;
  va_list args;
  va_start(args, matrix1);
  matrix2 = fold(add, zero, n, matrix1, args);
  va_end(args);
  return matrix2;
}

matrix *product(unsigned int n, matrix *matrix1, ...) {
  matrix *matrix2 = NULL;
  va_list args;
  va_start(args, matrix1);
  matrix2 = fold(multiply, one, n, matrix1, args);
  va_end(args);
  return matrix2;
}
