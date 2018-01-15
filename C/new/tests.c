// tests.c
#include "printing.h"
int main(int argc, char *argv[]) {
  // TEST 1:
  LSTM_type *LSTM = make_LSTM(4, 1);

  // TEST 3:
  matrix *matrix1 = NULL;
  matrix *matrix2 = NULL;
  matrix *matrix3 = NULL;

  // TEST 1:
  NOT_USED(argc);
  NOT_USED(argv);

  // TEST 2:
  //push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));

  // TEST 3:
  matrix1 =
    product(2, 
      matrix_tanh(first(LSTM, At_c)), 
      first(LSTM, Bt_iota));
  matrix2 =
    product(2, 
      second(LSTM, St_c), 
      first(LSTM, Bt_phi));
  matrix3 = sum(2, matrix1, matrix2);
  push(LSTM, St_c, matrix3);

  // TEST 1:
  print_LSTM(LSTM);

  // TEST 3:
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);

  // TEST 1:
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
