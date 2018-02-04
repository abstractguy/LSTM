// CNN.c    -- NAND gate
#include "matrix.h"
#include "printing.h"
int main(void) {
  matrix *in      = make_matrix(4, 4), *out = make_matrix(4, 1),
         *weights = matrix_initialize(random_long_double, 4, 1),
         *errors  = NULL, *updates = NULL, *answer  = NULL;
  long double input[4][4]  = {{0.0, 0.0, 1.0, 1.0},
                              {0.0, 1.0, 1.0, 1.0},
                              {1.0, 0.0, 1.0, 1.0},
                              {1.0, 1.0, 1.0, 1.0}};
  long double output[4][1] = {{1.0},
                              {1.0},
                              {1.0},
                              {0.0}};
  matrix_push_all(in,  (long double *)input);
  matrix_push_all(out, (long double *)output);
  puts("Input:");            print_matrix(in);
  puts("Output:");           print_matrix(out);
  puts("Weights:");          print_matrix(weights);
  answer  = matrix_sigmoid(dot_product(matrix_copy(in), 
                                       matrix_copy(weights)));
  puts("Feedforward once:"); print_matrix(answer);
  errors  = fold(2, multiply, fold(2, minus, out, matrix_copy(answer)), 
                              sigmoid_derivative(answer));
  puts("Feedback once:");    print_matrix(errors);
  updates = dot_product(transpose(in), errors);
  puts("Updates:");          print_matrix(updates);
  weights = fold(2, add, weights, 
                         fold(2, multiply, matrix_copy(weights), updates));
  puts("New weights:");      print_matrix(weights);
  weights = destroy_matrix(weights);
  return 0;
}
