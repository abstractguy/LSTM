// _printing.c
#include "_printing.h"

void print_matrix(matrix_type *matrix) {
  for (unsigned int row = 0; row < matrix->rows; row++) {
    for (unsigned int column = 0; column < matrix->columns; column++) {
      printf("%+5.4Lf ", matrix->matrix[row][column]);
    } putchar('\n');
  }
}

void print_LSTM(LSTM_type *LSTM) {
  for (index_type tensor = 0; tensor < LSTM_SIZE; tensor++) {
    printf("\nType %u:", tensor);
    for (unsigned int time = 0; time < LSTM->tensor[tensor].time; time++) {
      printf("\nTime %u:\n", time);
      print_matrix(LSTM->tensor[tensor].matrix[time]);
    } putchar('\n');
  }
}
