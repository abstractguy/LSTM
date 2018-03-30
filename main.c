// main.c
#include "feedforward.h"
#include "feedback.h"
#include "update.h"
#include "printing.h"

void run_LSTM(LSTM_type *);

int main(void) {
  LSTM_type *LSTM = NULL;

  // Stochastic mode.
  // TIME_SIZE: 4, WORD_SIZE: 2, BATCH_SIZE: 1, HIDDEN_SIZE: 16
  // Count inputs (Input & Xt):
  long double input[TIME_SIZE][BATCH_SIZE][WORD_SIZE] = {
    {{0.0, 0.0}}, // Count(0, 0) = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 1.0}}, // Count(0, 1) = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0}}, // Count(1, 0) = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 1.0}}  // Count(1, 1) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  };

  // TIME_SIZE: 4, WORD_SIZE: 2, BATCH_SIZE: 1, HIDDEN_SIZE: 16
  // Count inputs (Xt_reversed):
  long double input_reversed[TIME_SIZE][BATCH_SIZE][WORD_SIZE] = {
    {{1.0, 1.0}}, // Count(1, 1) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0}}, // Count(1, 0) = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 1.0}}, // Count(0, 1) = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 0.0}}  // Count(0, 0) = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  };

  // TIME_SIZE: 4, WORD_SIZE: 2, BATCH_SIZE: 1, HIDDEN_SIZE: 16
  // Count outputs (Output & Answer):
  long double output[TIME_SIZE][BATCH_SIZE][HIDDEN_SIZE] = {
    // Count(0, 0) = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(0, 1) = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(1, 0) = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(1, 1) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}
  };

  LSTM = make_LSTM((long double *)input, (long double *)input_reversed, (long double *)output, TIME_SIZE, WORD_SIZE, BATCH_SIZE, HIDDEN_SIZE);

  for (unsigned int epoch = 0; epoch < 500; epoch++) run_LSTM(LSTM);

  print_LSTM(LSTM);

  destroy_LSTM(LSTM);

  return 0;
}

void run_LSTM(LSTM_type *LSTM) {
  feedforward(LSTM);
  feedback(LSTM);
  update(LSTM);
}
