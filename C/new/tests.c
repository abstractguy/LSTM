// tests.c
#include "printing.h"
int main(int argc, char *argv[]) {
  // TEST 1:
  LSTM_type *LSTM = make_LSTM(4, 1);

  // TEST 1:
  NOT_USED(argc);
  NOT_USED(argv);

  // TEST 2:
  //push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));

  // TEST 3:
  push(LSTM, St_c, 
    sum(2, 
      product(2, 
        matrix_tanh(first(LSTM, At_c)), 
        first(LSTM, Bt_iota)), 
      product(2, 
        second(LSTM, St_c), 
        first(LSTM, Bt_phi))));

  // TEST 1:
  print_LSTM(LSTM);

  // TEST 1:
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
