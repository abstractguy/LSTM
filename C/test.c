// test.c
#include "LSTM.h"
#include <stdio.h>

// First test!
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  srand(1);
  for (index i = 0; i < LSTM_SIZE; i++) {
    printf("\nType %u:", i);
    for (int z = 0; z < LSTM->LSTM[i]->z; z++) {
      printf("\n\tTime %u:\n", z);
      for (int y = 0; y < LSTM->LSTM[i]->y; y++) {
        printf("\n\t\t");
        for (int x = 0; x < LSTM->LSTM[i]->x; x++) {
          printf("%5.4Lf ", LSTM->LSTM[i]->tensor[z][y][x]);
        }
      }
    }
  }
  printf("\n\n");
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
