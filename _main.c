// main.c
#include "_feedforward.h"
#include "_feedback.h"
//#include "_update.h"
#include "_printing.h"

void run_LSTM(LSTM_type *);

int main(void) {
  LSTM_type *LSTM = NULL;

  // NAND inputs (Input & Xt):
  long double input[4][1][1] = {
    {{0.0}}, // NAND(1, 0) = 1
    {{1.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}  // NAND(0, 0) = 1
  };

  // NAND inputs (Xt_reversed):
  long double input_reversed[4][1][1] = {
    {{0.0}}, // NAND(0, 0) = 1
    {{1.0}}, // NAND(0, 1) = 1
    {{1.0}}, // NAND(1, 1) = 0
    {{0.0}}  // NAND(1, 0) = 1
  };

  // NAND outputs (Output & Answer):
  long double output[4][1][1] = {
    {{1.0}}, // NAND(0, 0) = 1
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}, // NAND(1, 1) = 0
    {{1.0}}  // NAND(1, 0) = 1
  };

  LSTM = make_LSTM((long double *)input, (long double *)input_reversed, (long double *)output, 4, 1, 1);
  //for (unsigned int epoch = 0; epoch < 25; epoch++) {
    run_LSTM(LSTM);
  //}

  print_LSTM(LSTM);

  LSTM = destroy_LSTM(LSTM);
  return 0;
}

void run_LSTM(LSTM_type *LSTM) {
  feedforward(LSTM);
  feedback(LSTM);
  //update(LSTM);
}
