// main.c
//#include "_feedforward.h"
//#include "_feedback.h"
//#include "_update.h"
#include "_LSTM.h"
#include "_printing.h"

//void run_LSTM(LSTM_type *);

int main(void) {
  LSTM_type *LSTM = NULL;

  // NAND inputs (Input):
  long double input[4][1][1] = {
    {{0.0}}, // NAND(1, 0) = 1
    {{1.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{0.0}}  // NAND(0, 0) = 1
  };

  // NAND outputs (Output):
  long double output[5][1][1] = {
    {{0.0}}, // Dummy output
    {{1.0}}, // NAND(1, 0) = 1
    {{0.0}}, // NAND(1, 1) = 0
    {{1.0}}, // NAND(0, 1) = 1
    {{1.0}}  // NAND(0, 0) = 1
  };

  LSTM = make_LSTM((long double *)input, (long double *)output, 4, 1, 1);
  //for (unsigned int epoch = 0; epoch < 25; epoch++) run_LSTM(LSTM);

  print_LSTM(LSTM);
  LSTM = destroy_LSTM(LSTM);
  return 0;
}

/*
void run_LSTM(LSTM_type *LSTM) {
  for (unsigned int epoch = 0; epoch < 4; epoch++) {
    feedforward_once(LSTM, epoch);
  }
  LSTM_copy_last_matrix_to_beginning(LSTM, GATES_BEGIN, GATES_END);
  LSTM_copy_last_matrix_to_beginning(LSTM, ERRORS_BEGIN, ERRORS_END);
  for (unsigned int epoch = 4; epoch > 0; epoch--) {
    feedback_once(LSTM, epoch);
    update_forward_once(LSTM, epoch);
  } update_backward_once(LSTM);
}
*/
