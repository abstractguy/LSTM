// CNN.c    -- NAND gate
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define EPOCH 60000

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
matrix_type *product(matrix_type *, matrix_type *);
matrix_type *subtract(matrix_type *, matrix_type *);
matrix_type *sum(matrix_type *, matrix_type *);
matrix_type *transpose(matrix_type *);
void print_matrix(char *, matrix_type *);
void matrix_push_all(char *, matrix_type *, long double *);

int main(void) {
  matrix_type *in       = make_matrix(4, 4), *out    = make_matrix(4, 1),
              *synapses = make_matrix(4, 1), *errors = NULL, *answer = NULL;
  long double input[4][4]   = {{0.0, 0.0, 1.0, 1.0},
                               {0.0, 1.0, 1.0, 1.0},
                               {1.0, 0.0, 1.0, 1.0},
                               {1.0, 1.0, 1.0, 1.0}};
  long double output[4][1]  = {{1.0},
                               {1.0},
                               {1.0},
                               {0.0}};
  for (unsigned int row = 0; row < synapses->rows; row++)
    for (unsigned int column = 0; column < synapses->columns; column++)
      synapses->matrix[row][column] = (2.0 * rand() / RAND_MAX) - 1.0;

  matrix_push_all("Input:",  in,  (long double *)input);
  matrix_push_all("Output:", out, (long double *)output);

  for (unsigned int epoch = 0; epoch < EPOCH; epoch++) {
    answer   = sigmoid(dot_product(matrix_copy(in), matrix_copy(synapses)));
    if (epoch == EPOCH-1) print_matrix("Feedforward:", answer);

    errors   = product(sigmoid_derivative(answer), subtract(out, answer));
    if (epoch == EPOCH-1) print_matrix("Feedback:", errors);

    synapses = sum(synapses, dot_product(transpose(in), errors));
    if (epoch == EPOCH-1) print_matrix("New synapses:", synapses);

  } destroy_matrix(in); destroy_matrix(out); destroy_matrix(synapses);
    return 0;
}

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

matrix_type *product(matrix_type *matrix1, matrix_type *matrix2) {
  matrix_type *matrix3 = make_matrix(matrix2->rows, matrix2->columns);

  for (unsigned int row = 0; row < matrix2->rows; row++)
    for (unsigned int column = 0; column < matrix2->columns; column++)
      matrix3->matrix[row][column] = 
        matrix1->matrix[row][column] * matrix2->matrix[row][column];

  destroy_matrix(matrix1);
  destroy_matrix(matrix2);

  return matrix3;
}

matrix_type *subtract(matrix_type *matrix1, matrix_type *matrix2) {
  matrix_type *matrix3 = make_matrix(matrix2->rows, matrix2->columns);

  for (unsigned int row = 0; row < matrix2->rows; row++)
    for (unsigned int column = 0; column < matrix2->columns; column++)
      matrix3->matrix[row][column] =
        matrix1->matrix[row][column] - matrix2->matrix[row][column];

  return matrix3;
}

matrix_type *sum(matrix_type *matrix1, matrix_type *matrix2) {
  matrix_type *matrix3 = make_matrix(matrix2->rows, matrix2->columns);

  for (unsigned int row = 0; row < matrix2->rows; row++)
    for (unsigned int column = 0; column < matrix2->columns; column++)
      matrix3->matrix[row][column] =
        matrix1->matrix[row][column] + 
          matrix2->matrix[row][matrix2->columns == 1 ? 0 : column];

  destroy_matrix(matrix1);
  destroy_matrix(matrix2);

  return matrix3;
}

matrix_type *transpose(matrix_type *matrix1) {
  matrix_type *matrix2 = make_matrix(matrix1->columns, matrix1->rows);

  for (unsigned int row = 0; row < matrix1->columns; row++)
    for (unsigned int column = 0; column < matrix1->rows; column++)
      matrix2->matrix[row][column] = matrix1->matrix[column][row];

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
