// test1.c
#include "printing.h"

// First test!
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  srand(1);
  print_LSTM(LSTM);
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
