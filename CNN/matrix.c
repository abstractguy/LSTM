// matrix.c
#include "matrix.h"

matrix_type *make_matrix(unsigned int rows, unsigned int columns) {
  matrix_type *matrix   = malloc(sizeof(matrix_type));
  matrix->rows          = rows;
  matrix->columns       = columns;
  matrix->matrix        = malloc(sizeof(long double *) * rows);
  for (unsigned int row = 0; row < rows; row++)
    matrix->matrix[row] = malloc(sizeof(long double) * columns);
  return matrix;
}

void destroy_matrix(matrix_type *matrix) {
  for (unsigned int row = 0; row < matrix->rows; row++)
    free(matrix->matrix[row]);
  free(matrix->matrix);
  free(matrix);
}

matrix_type *dot_product(matrix_type *matrix1, matrix_type *matrix2) {
  matrix_type *matrix3 = make_matrix(matrix1->rows, matrix2->columns);

  for (unsigned int row1 = 0; row1 < matrix1->rows; row1++)
    for (unsigned int column2 = 0; column2 < matrix2->columns; column2++) {
      matrix3->matrix[row1][column2] = 0.0;
      for (unsigned int column1 = 0; column1 < matrix1->columns; column1++)
        matrix3->matrix[row1][column2] += 
          matrix1->matrix[row1][column1] * matrix2->matrix[column1][column2];
    }
  destroy_matrix(matrix1);
  destroy_matrix(matrix2);
  return matrix3;
}

matrix_type *matrix_copy(matrix_type *matrix1) {
  matrix_type *matrix2 = make_matrix(matrix1->rows, matrix1->columns);
  for (unsigned int row = 0; row < matrix1->rows; row++)
    for (unsigned int column = 0; column < matrix1->columns; column++)
      matrix2->matrix[row][column] = matrix1->matrix[row][column];
  return matrix2;
}

long double sigmoid_double(long double x) {
  return 1.0 / (1.0 + expl(-x));
}

matrix_type *sigmoid(matrix_type *matrix) {
  for (unsigned int row = 0; row < matrix->rows; row++)
    for (unsigned int column = 0; column < matrix->columns; column++)
      matrix->matrix[row][column] = sigmoid_double(matrix->matrix[row][column]);
  return matrix;
}

matrix_type *sigmoid_derivative(matrix_type *matrix) {
  long double y;
  for (unsigned int row = 0; row < matrix->rows; row++)
    for (unsigned int column = 0; column < matrix->columns; column++) {
      y = sigmoid_double(matrix->matrix[row][column]);
      matrix->matrix[row][column] = y * (1.0 - y);
    }
  return matrix;
}

matrix_type *broadcast(long double (*f)(long double, long double), 
                       matrix_type *matrix1, matrix_type *matrix2) {
  matrix_type *matrix3 = NULL;
  unsigned int
    rows    = matrix1->rows    == 1 ? matrix2->rows    : matrix1->rows,
    columns = matrix1->columns == 1 ? matrix2->columns : matrix1->columns;

  matrix3 = make_matrix(rows, columns);

  for (unsigned int row = 0; row < rows; row++)
    for (unsigned int column = 0; column < columns; column++)
      matrix3->matrix[row][column] =
        f(matrix1->matrix[matrix1->rows    == 1 ? 0 : row]
                         [matrix1->columns == 1 ? 0 : column],
          matrix2->matrix[matrix2->rows    == 1 ? 0 : row]
                         [matrix2->columns == 1 ? 0 : column]);

  destroy_matrix(matrix1);
  destroy_matrix(matrix2);

  return matrix3;
}

long double add(long double x, long double y) {return x + y;}
long double multiply(long double x, long double y) {return x * y;}
long double minus(long double x, long double y) {return x - y;}

matrix_type *transpose(matrix_type *matrix1) {
  matrix_type *matrix2 = make_matrix(matrix1->columns, matrix1->rows);

  for (unsigned int row = 0; row < matrix1->columns; row++)
    for (unsigned int column = 0; column < matrix1->rows; column++)
      matrix2->matrix[row][column] = matrix1->matrix[column][row];

  destroy_matrix(matrix1);
  return matrix2;
}

void print_matrix(char *string, matrix_type *matrix) {
  puts(string);
  for (unsigned int row = 0; row < matrix->rows; row++) {
    for (unsigned int column = 0; column < matrix->columns; column++) {
      printf("%+5.4Lf ", matrix->matrix[row][column]);
    } putchar('\n');
  }
}

void matrix_push_all(char *string, matrix_type *matrix, long double *array) {
  for (unsigned int row = 0; row < matrix->rows; row++)
    for (unsigned int column = 0; column < matrix->columns; column++)
      matrix->matrix[row][column] = array[row * matrix->columns + column];
  print_matrix(string, matrix);
}
