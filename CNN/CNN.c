// CNN.c    -- NAND gate
#include "matrix.h"
#include "printing.h"
#define EPOCH 60000
int main(void) {
  matrix_type *in      = make_matrix(4, 4), *out    = make_matrix(4, 1),
              *weights = make_matrix(4, 1), *errors = NULL, *answer = NULL;
  long double input[4][4]   = {{0.0, 0.0, 1.0, 1.0},
                               {0.0, 1.0, 1.0, 1.0},
                               {1.0, 0.0, 1.0, 1.0},
                               {1.0, 1.0, 1.0, 1.0}};
  long double output[4][1]  = {{1.0},
                               {1.0},
                               {1.0},
                               {0.0}};
  for (unsigned int row = 0; row < weights->rows; row++)
    for (unsigned int column = 0; column < weights->columns; column++)
      weights->matrix[row][column] = 
        2.0 * (((long double)rand()) / ((long double)RAND_MAX)) - 1.0;
  matrix_push_all(in,  (long double *)input);
  matrix_push_all(out, (long double *)output);
  print_matrix("Input:",   in);
  print_matrix("Output:",  out);
  for (unsigned int epoch = 0; epoch < EPOCH; epoch++) {
    answer  = sigmoid(dot_product(matrix_copy(in),
                                  matrix_copy(weights)));
    if (epoch == EPOCH-1) print_matrix("Feedforward:", answer);

    errors  = product(sigmoid_derivative(answer),
                      subtract(matrix_copy(out), 
                               matrix_copy(answer)));
    if (epoch == EPOCH-1) print_matrix("Feedback:", errors);

    weights = sum(weights, dot_product(transpose(matrix_copy(in)), errors));
    if (epoch == EPOCH-1) print_matrix("New weights:", weights);

  } destroy_matrix(in); destroy_matrix(out); destroy_matrix(weights);
    return 0;
}
