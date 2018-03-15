// LSTM.c
#include "LSTM.h"

LSTM_type *make_LSTM(long double *input, long double *input_reversed, long double *output, unsigned int t, unsigned int rows, unsigned int columns) {
  LSTM_type *LSTM = NULL;
  LSTM = malloc(sizeof(LSTM_type));
  assert(LSTM);
  //srand(time(NULL));
  srand(1);

  // Empty input/outputs to initialize (Input, Output):
  LSTM_initialize(LSTM, Input, Output, zero, t, columns, rows);
  LSTM_initialize(LSTM, Output, GATES_BEGIN, zero, t, rows, columns);

  LSTM_initialize(LSTM, GATES_BEGIN, GATES_END, one, 2, columns, rows);
  LSTM_initialize(LSTM, INPUT_WEIGHTS_BEGIN, INPUT_WEIGHTS_END, random_long_double, 1, rows, columns);
  LSTM_initialize(LSTM, HIDDEN_WEIGHTS_BEGIN, HIDDEN_WEIGHTS_END, random_long_double, 1, rows, rows);
  LSTM_initialize(LSTM, CELL_WEIGHTS_BEGIN, CELL_WEIGHTS_END, random_long_double, 1, rows, columns);
  LSTM_initialize(LSTM, BIAS_WEIGHTS_BEGIN, BIAS_WEIGHTS_END, one, 1, rows, columns);
  LSTM_initialize(LSTM, ERRORS_BEGIN, ERRORS_END, zero, 2, columns, rows);
  LSTM_initialize(LSTM, INPUT_UPDATES_BEGIN, INPUT_UPDATES_END, zero, 1, rows, columns);
  LSTM_initialize(LSTM, HIDDEN_UPDATES_BEGIN, HIDDEN_UPDATES_END, zero, 1, rows, rows);
  LSTM_initialize(LSTM, CELL_UPDATES_BEGIN, CELL_UPDATES_END, zero, 1, rows, columns);
  LSTM_initialize(LSTM, BIAS_UPDATES_BEGIN, BIAS_UPDATES_END, one, 1, rows, columns);

  push_all(LSTM, Input,  (long double *)input);
  push_all(LSTM, Output, (long double *)output);
  push_all(LSTM, Input_reversed, (long double *)input_reversed);

  return LSTM;
}

LSTM_type *destroy_LSTM(LSTM_type *LSTM) {
  for (index_type tensor = 0; tensor < LSTM_SIZE; tensor++) {
    for (unsigned int time = 0; time < LSTM->tensor[tensor].time; time++) {
      LSTM->tensor[tensor].matrix[time] = destroy_matrix(LSTM->tensor[tensor].matrix[time]);
    } free(LSTM->tensor[tensor].matrix);
      LSTM->tensor[tensor].matrix = NULL;
  }   free(LSTM);
      return NULL;
}

void LSTM_initialize(LSTM_type *LSTM, index_type begin, index_type end, long double (*init)(long double), unsigned int time, unsigned int rows, unsigned int columns) {
  for (index_type tensor = begin; tensor < end; tensor++) {
    LSTM->tensor[tensor].time = time;
    LSTM->tensor[tensor].matrix = calloc(1, sizeof(matrix_type *) + sizeof(matrix_type) * time);
    assert(LSTM->tensor[tensor].matrix);
    for (unsigned int t = 0; t < time; t++) {
      LSTM->tensor[tensor].matrix[t] = make_matrix(rows, columns);
      matrix_for_each(init, LSTM->tensor[tensor].matrix[t]);
    }
  }
}

matrix_type *LSTM_matrix(LSTM_type *LSTM, index_type tensor, long index) {
  long time = LSTM->tensor[tensor].time;
  assert(index >= -time && index < time);
  return LSTM->tensor[tensor].matrix[index < 0 ? time + index : index];
}

matrix_type *LSTM_read(LSTM_type *LSTM, index_type tensor, long index) {
  return matrix_copy(LSTM_matrix(LSTM, tensor, index));
}

void LSTM_write(LSTM_type *LSTM, index_type tensor, long time, matrix_type *matrix1) {
  matrix_type *matrix2 = LSTM_matrix(LSTM, tensor, time);
  matrix2 = matrix1;
}

void push(LSTM_type *LSTM, index_type tensor, matrix_type *matrix) {
  LSTM->tensor[tensor].matrix = realloc(LSTM->tensor[tensor].matrix, sizeof(matrix_type *) + sizeof(matrix_type) * (LSTM->tensor[tensor].time + 1));
  assert(LSTM->tensor[tensor].matrix);
  LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time++] = matrix;
}

matrix_type *pop(LSTM_type *LSTM, index_type tensor) {
  matrix_type *matrix = LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time - 1];
  LSTM->tensor[tensor].matrix = realloc(LSTM->tensor[tensor].matrix, sizeof(matrix_type *) + sizeof(matrix_type) * --LSTM->tensor[tensor].time);
  assert(LSTM->tensor[tensor].matrix);
  return matrix;
}

void push_all(LSTM_type *LSTM, index_type tensor, long double *steps) {
  unsigned int time = LSTM->tensor[tensor].time, rows, columns;
  for (unsigned int n = 0; n < time; n++) {
    rows = LSTM->tensor[tensor].matrix[n]->rows;
    columns = LSTM->tensor[tensor].matrix[n]->columns;
    for (unsigned int row = 0; row < rows; row++) {
      for (unsigned int column = 0; column < columns; column++) {
        LSTM->tensor[tensor].matrix[n]->matrix[row][column] = steps[n * rows * columns + row * columns + column];
      }
    }
  }
}

void empty_tensor(LSTM_type *LSTM, index_type tensor) {
  matrix_type *matrix = NULL;
  while (LSTM->tensor[tensor].time) {
    matrix = pop(LSTM, tensor);
    matrix = destroy_matrix(matrix);
  }
}

void copy_tensor(LSTM_type *LSTM, index_type tensor1, index_type tensor2) {
  empty_tensor(LSTM, tensor2);
  for (unsigned int time = 0; time < LSTM->tensor[tensor1].time; time++) {
    push(LSTM, tensor2, matrix_copy(LSTM->tensor[tensor1].matrix[time]));
  }
}
