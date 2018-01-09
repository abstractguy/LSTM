// tests.c
#include "printing.h"
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  NOT_USED(argc);
  NOT_USED(argv);
  push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));
  print_LSTM(LSTM);
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
