// tests.c
#include "feedforward.h"
#include "feedback.h"
#include "printing.h"

int main(void) {
  // TEST 1:
  LSTM_type *LSTM = make_LSTM(5, 1, 1);

  // TEST 4/5:
  // XOR inputs (Xt_i):
  long double inputs[5][1][1] = {
    {{0.0}}, // NAND(1, 0) = 1
    {{1.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}, // NAND(0, 0) = 0
    {{0.0}}  // Dummy input
  };

  // XOR outputs (Yt_k):
  long double outputs[5][1][1] = {
    {{1.0}}, // NAND(1, 0) = 1
    {{0.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}, // NAND(0, 0) = 0
    {{0.0}}  // Dummy output
  };

  /* TEST 2:
  push(LSTM, At_iota, dot_product(first(LSTM, Xt_i), first(LSTM, Wi_iota)));
  */

  /* TEST 3:
  push(LSTM, St_c, 
    sum(2, 
      product(2, 
        matrix_tanh(first(LSTM, At_c)), 
        first(LSTM, Bt_iota)), 
      product(2, 
        second(LSTM, St_c), 
        first(LSTM, Bt_phi))));
  */

  // TEST 4/5:
  push_all(LSTM, Xt_i, (long double *)inputs);
  push_all(LSTM, Yt_k, (long double *)outputs);

  // TEST 4:
  feedforward_once(LSTM);

  // TEST 5:
  feedback_once(LSTM);

  // TEST 1:
  print_LSTM(LSTM);

  // TEST 1:
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
