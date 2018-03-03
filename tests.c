// tests.c
#include "feedforward.h"
#include "feedback.h"
#include "update.h"
#include "printing.h"

void run_LSTM(LSTM_type *);

int main(void) {
  LSTM_type *LSTM = make_LSTM(4, 1, 1);

  // NAND inputs (Xt_i):
  long double inputs[4][1][1] = {
    {{0.0}}, // NAND(1, 0) = 1
    {{1.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}, // NAND(0, 0) = 1
  };

  // NAND outputs (Yt_k):
  long double outputs[4][1][1] = {
    {{1.0}}, // NAND(1, 0) = 1
    {{0.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{1.0}}, // NAND(0, 0) = 1
  };

  push_all(LSTM, Xt_i, (long double *)inputs);
  push_all(LSTM, Yt_k, (long double *)outputs);

  for (unsigned int epoch = 0; epoch < 1000; epoch++) {
    run_LSTM(LSTM);
  }

  print_LSTM
(LSTM);

  LSTM = destroy_LSTM(LSTM);
  return 0;
}

void run_LSTM(LSTM_type *LSTM) {
  for (unsigned int epoch = 0; epoch < 4; epoch++) {
    feedforward_once(LSTM);
    feedback_once(LSTM);
    update_forward_once(LSTM);
  } update_backward_once(LSTM);
}
