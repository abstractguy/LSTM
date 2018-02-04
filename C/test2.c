// CNN.c
#include "matrix.h"
#include "printing.h"

void matrix_push_all(matrix *, long double *);

int main(void) {
  matrix *in      = make_matrix(4, 4),
         *out     = make_matrix(4, 1),
         *weights = matrix_initialize(random_long_double, 4, 1),
         *errors  = NULL, *updates = NULL, *answer  = NULL;
  long double input[4][4] = {
    {0.0, 0.0, 1.0, 1.0},
    {0.0, 1.0, 1.0, 1.0},
    {1.0, 0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0, 1.0}
  };
  long double output[4][1] = {
    {1.0},
    {1.0},
    {1.0},
    {0.0}
  };
  matrix_push_all(in,  (long double *)input);
  matrix_push_all(out, (long double *)output);

  puts("\nInput:");
  print_matrix(in);
  puts("\nOutput:");
  print_matrix(out);
  puts("\nWeights:");
  print_matrix(weights);

  puts("\nFeedforward once:");
  answer = 
    matrix_sigmoid(
      dot_product(
        matrix_copy(in), 
        matrix_copy(weights)));
  print_matrix(answer);

  puts("\nFeedback once:");
  errors = 
    fold(2, multiply, 
      fold(2, minus, out, matrix_copy(answer)), 
      sigmoid_derivative(answer));
  print_matrix(errors);

  puts("\nUpdates:");
  updates = dot_product(transpose(in), errors);
  print_matrix(updates);

  puts("\nNew weights");
  weights = fold(2, multiply, weights, updates);
  print_matrix(weights);

  weights = destroy_matrix(weights);
  return 0;
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

