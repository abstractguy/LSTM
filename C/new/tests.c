// tests.c
#include "printing.h"

// First test!
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  matrix *matrix = NULL;
  NOT_USED(argc);
  NOT_USED(argv);
  srand(1);
  print_LSTM(LSTM);
  matrix = dot_product(LSTM->tensor[Xt_i].matrix[LSTM->tensor[Xt_i].time - 1], LSTM->tensor[Wi_iota].matrix[0]);
  matrix = destroy_matrix(matrix);
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
