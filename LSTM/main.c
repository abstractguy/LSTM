// main.c
#include "feedforward.h"
#include "feedback.h"
#include "update.h"
#include "printing.h"

int main(void) {
  LSTM_type *LSTM = NULL;

  // Stochastic mode.
  // TIME_SIZE: 6, WORD_SIZE: 4, BATCH_SIZE: 1, HIDDEN_SIZE: 16
  // Count inputs (Xt):
  long double input[TIME_SIZE][BATCH_SIZE][WORD_SIZE] = {
    {{0.0, 0.0, 0.0, 0.0}}, // Dummy input
    {{0.0, 1.0, 0.0, 1.0}}, // Count(0, 0) = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 1.0, 1.0, 0.0}}, // Count(0, 1) = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0, 0.0, 1.0}}, // Count(1, 0) = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0, 1.0, 0.0}}, // Count(1, 1) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 1.0, 1.0, 1.0}}  // Dummy input
  };

  // TIME_SIZE: 6, WORD_SIZE: 4, BATCH_SIZE: 1, HIDDEN_SIZE: 16
  // Count outputs (Yt):
  long double output[TIME_SIZE][BATCH_SIZE][HIDDEN_SIZE] = {
    // Dummy output
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    // Count(0, 0) = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(0, 1) = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(1, 0) = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, 
    // Count(1, 1) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    // Dummy output
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}
  };

  LSTM = make_LSTM((long double *)input, (long double *)output, TIME_SIZE, WORD_SIZE, BATCH_SIZE, HIDDEN_SIZE);

  for (unsigned int epoch = 0; epoch < 300; epoch++) {
    feedforward(LSTM);
    feedback(LSTM);
    update(LSTM);
  }

  print_LSTM(LSTM);

  destroy_LSTM(LSTM);

  return 0;
}
