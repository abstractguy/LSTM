// printing.c
#include "printing.h"

void print_LSTM(LSTM_type *LSTM) {
  for (index tensor = 0; tensor < LSTM_SIZE; tensor++) {
    printf("\nType %u:", tensor);
    for (int time = 0; time < LSTM->tensor[tensor].time; time++) {
      printf("\nTime %u:", time);
      for (int row = 0; row < LSTM->tensor[tensor].matrix[time]->rows; row++) {
        putchar('\n');
        for (int column = 0; column < LSTM->tensor[tensor].matrix[time]->columns; column++) {
          printf("%+5.4Lf ", LSTM->tensor[tensor].matrix[time]->matrix[row][column]);
        }
      }
    }
    putchar('\n');
  }
}
