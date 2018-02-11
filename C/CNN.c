// CNN.c    -- NAND gate
#include "matrix.h"
#include "printing.h"
int main(void) {
  matrix *in = make_matrix(4, 4), *out = make_matrix(4, 1), *weights = make_matrix(4, 1),
         *derivative = NULL, *delta = NULL, *errors  = NULL, *answer = NULL;
  long double input[4][4]   = {{0.0, 0.0, 1.0, 1.0},
                               {0.0, 1.0, 1.0, 1.0},
                               {1.0, 0.0, 1.0, 1.0},
                               {1.0, 1.0, 1.0, 1.0}};
  long double output[4][1]  = {{1.0},
                               {1.0},
                               {1.0},
                               {0.0}};
  long double w[4][1]       = {{1.0},
                               {1.0},
                               {1.0},
                               {1.0}};
  matrix_push_all(in, (long double *)input); matrix_push_all(out, (long double *)output); matrix_push_all(weights, (long double *)w);
  puts("Input:"); print_matrix(in);          puts("Output:"); print_matrix(out);          puts("Weights:"); print_matrix(weights);
  for (unsigned int epoch = 0; epoch < 60000; epoch++) {
    answer     = matrix_sigmoid(dot_product(matrix_copy(in), matrix_copy(weights)));
    derivative = sigmoid_derivative(matrix_copy(answer));
    delta      = fold(2, minus,    matrix_copy(out),        matrix_copy(answer));
    errors     = fold(2, multiply, matrix_copy(derivative), matrix_copy(delta));
    weights    = fold(2, add,      weights,                 dot_product(transpose(matrix_copy(in)), matrix_copy(errors)));
    if (epoch == 59999) {
      puts("Feedforward once:"); print_matrix(answer);
      puts("Derivative:");       print_matrix(derivative);
      puts("delta:");            print_matrix(delta);
      puts("Feedback once:");    print_matrix(errors);
      puts("New weights:");      print_matrix(weights);
    } answer = destroy_matrix(answer); derivative = destroy_matrix(derivative); delta = destroy_matrix(delta); errors = destroy_matrix(errors);
  }   in = destroy_matrix(in); out = destroy_matrix(out); weights = destroy_matrix(weights);
      return 0;
}
