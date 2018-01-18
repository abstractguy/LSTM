// printing.c
#include "printing.h"

void print_matrix(matrix *matrix1) {
  for (unsigned int row = 0; row < matrix1->rows; row++) {
    for (unsigned int column = 0; column < matrix1->columns; column++) {
      printf("%+5.4Lf ", matrix1->matrix[row][column]);
    } putchar('\n');
  }
}

void print_LSTM(LSTM_type *LSTM) {
  for (index tensor = 0; tensor < LSTM_SIZE; tensor++) {
    printf("\nType %u:", tensor);
    for (unsigned int time = 0; time < LSTM->tensor[tensor].time; time++) {
      printf("\nTime %u:\n", time);
      print_matrix(LSTM->tensor[tensor].matrix[time]);
    } putchar('\n');
  }
}
