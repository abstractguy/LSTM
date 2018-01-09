// LSTM.c
#include "LSTM.h"
LSTM_type *make_LSTM(unsigned int rows, unsigned int columns) {
  LSTM_type *LSTM = NULL;
  LSTM = calloc(1, sizeof(LSTM_type));
  assert(LSTM);
  // Empty inputs (Xt_i):
  LSTM_initialize_tensors(LSTM, 0, 1, zero, 1, columns, rows);
  // Empty outputs (Yt_k):
  LSTM_initialize_tensors(LSTM, 1, 2, zero, 1, rows, columns * 2);
  LSTM_initialize_tensors(LSTM, GATES_BEGIN, GATES_END, one, 2, columns, rows);
  LSTM_initialize_tensors(LSTM, INPUT_WEIGHTS_BEGIN, INPUT_WEIGHTS_END, random_long_double, 1, rows, columns);
  LSTM_initialize_tensors(LSTM, HIDDEN_WEIGHTS_BEGIN, HIDDEN_WEIGHTS_END, random_long_double, 1, rows, rows);
  LSTM_initialize_tensors(LSTM, CELL_WEIGHTS_BEGIN, CELL_WEIGHTS_END, random_long_double, 1, rows, columns);
  LSTM_initialize_tensors(LSTM, ERRORS_BEGIN, ERRORS_END, zero, 1, columns, rows);
  LSTM_initialize_tensors(LSTM, INPUT_UPDATES_BEGIN, INPUT_UPDATES_END, zero, 1, rows, columns);
  LSTM_initialize_tensors(LSTM, HIDDEN_UPDATES_BEGIN, HIDDEN_UPDATES_END, zero, 1, rows, rows);
  LSTM_initialize_tensors(LSTM, CELL_UPDATES_BEGIN, CELL_UPDATES_END, zero, 1, rows, columns);
  return LSTM;
}
LSTM_type *destroy_LSTM(LSTM_type *LSTM) {
  for (index tensor = 0; tensor < LSTM_SIZE; tensor++) {
    for (unsigned int time = 0; time < LSTM->tensor[tensor].time; time++) {
      LSTM->tensor[tensor].matrix[time] = destroy_matrix(LSTM->tensor[tensor].matrix[time]);
    } free(LSTM->tensor[tensor].matrix);
      LSTM->tensor[tensor].matrix = NULL;
  }   free(LSTM);
      return NULL;
}
void LSTM_initialize_tensors(LSTM_type *LSTM, index begin, index end, long double (*init)(long double), unsigned int time, unsigned int rows, unsigned int columns) {
  for (index tensor = begin; tensor < end; tensor++) {
    LSTM->tensor[tensor].time = time;
    LSTM->tensor[tensor].matrix = calloc(1, sizeof(matrix *) + sizeof(matrix) * time);
    assert(LSTM->tensor[tensor].matrix);
    for (unsigned int t = 0; t < time; t++) {
      LSTM->tensor[tensor].matrix[t] = make_matrix(rows, columns);
      matrix_for_each(init, LSTM->tensor[tensor].matrix[t]);
    }
  }
}
matrix *first(LSTM_type *LSTM, index tensor) {
  return LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time - 1];
}
matrix *second(LSTM_type *LSTM, index tensor) {
  return LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time - 2];
}
void push(LSTM_type *LSTM, index tensor, matrix *matrix1) {
  LSTM->tensor[tensor].matrix = realloc(LSTM->tensor[tensor].matrix, sizeof(matrix *) + sizeof(matrix) * (LSTM->tensor[tensor].time + 1));
  assert(LSTM->tensor[tensor].matrix);
  LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time] = matrix1;
}
matrix *pop(LSTM_type *LSTM, index tensor) {
  matrix *matrix1 = first(LSTM, tensor);
  LSTM->tensor[tensor].matrix = realloc(LSTM->tensor[tensor].matrix, sizeof(matrix *) + sizeof(matrix) * LSTM->tensor[tensor].time - 1);
  assert(LSTM->tensor[tensor].matrix);
  return matrix1;
}
