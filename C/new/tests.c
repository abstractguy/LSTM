// tests.c
#include "printing.h"
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  NOT_USED(argc);
  NOT_USED(argv);

  //push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));

  push(LSTM, St_c, 
    fold(add, zero, 2, 
      fold(multiply, one, 2, 
        matrix_tanh(first(LSTM, At_c)), 
        first(LSTM, Bt_iota)), 
      fold(multiply, one, 2, 
        second(LSTM, St_c), 
        first(LSTM, Bt_phi))));

  print_LSTM(LSTM);
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
