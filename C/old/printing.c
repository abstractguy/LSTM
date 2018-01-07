// printing.c
#include "printing.h"

void print_LSTM(LSTM_type *LSTM) {
  for (index i = 0; i < LSTM_SIZE; i++) {
    printf("\nType %u:", i);
    for (int z = 0; z < LSTM->LSTM[i]->z; z++) {
      printf("\nTime %u:", z);
      for (int y = 0; y < LSTM->LSTM[i]->y; y++) {
        printf("\n");
        for (int x = 0; x < LSTM->LSTM[i]->x; x++) {
          printf("%+5.4Lf ", LSTM->LSTM[i]->tensor[z][y][x]);
        }
      }
    }
    printf("\n");
  }
}
