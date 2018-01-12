// tests.c
#include "printing.h"
int main(int argc, char *argv[]) {
  // TEST 1:
  LSTM_type *LSTM = make_LSTM(4, 1);

  // TEST 3a:
  matrix *matrix1 = make_matrix(4, 1);
  matrix *matrix2 = make_matrix(4, 1);
  matrix *matrix3 = NULL;
  matrix *matrix4 = NULL;
  matrix_for_each(zero, matrix1);
  matrix_for_each(one, matrix2);

  // TEST 1:
  NOT_USED(argc);
  NOT_USED(argv);

  // TEST 2:
  //push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));

  // TEST 3:
  matrix3 =
    fold(multiply, one, 2, 
      matrix_tanh(first(LSTM, At_c)), 
      first(LSTM, Bt_iota));
  matrix4 =
    fold(multiply, one, 2, 
      second(LSTM, St_c), 
      first(LSTM, Bt_phi));
  push(LSTM, St_c, fold(add, zero, 2, matrix3, matrix4));

  //TEST 1:
  print_LSTM(LSTM);

  // TEST 3a:
  //matrix3 = fold(add, zero, 2, matrix1, matrix2);
  //print_matrix(matrix3);

  // TEST 3:
  matrix3 = destroy_matrix(matrix3);
  matrix4 = destroy_matrix(matrix4);

  LSTM = destroy_LSTM(LSTM);
  return 0;
}
