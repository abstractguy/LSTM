// test2.c
#include "printing.h"
#include "feedforward.h"

// First test!
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  srand(1);
  print_LSTM(LSTM);
  printf("\nFeedforwarding once...\n");
  feedforward_once(LSTM);
  printf("\n\n");
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
