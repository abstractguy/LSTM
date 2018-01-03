// test2.c
#include "printing.h"
#include "feedforward.h"

// First test!
int main(int argc, char *argv[]) {
  LSTM_type *LSTM = make_LSTM(4, 1);
  long double inputs[5][4][1] = {
    {{1},{1},{1},{1}},
    {{1},{1},{1},{1}},
    {{1},{0},{1},{1}},
    {{0},{1},{1},{1}},
    {{0},{0},{1},{1}}
  };

  long double outputs[6][1][2] = {
    {{1, 1}},
    {{0, 1}},
    {{0, 1}},
    {{1, 0}},
    {{1, 0}},
    {{0, 1}}
  };

  LSTM->LSTM[Xt_i] = destroy_tensor_3D(LSTM->LSTM[Xt_i]);
  LSTM->LSTM[Yt_k] = destroy_tensor_3D(LSTM->LSTM[Yt_k]);
  LSTM->LSTM[Xt_i] = make_tensor_3D(zero, 1, 4, 5);
  LSTM->LSTM[Yt_k] = make_tensor_3D(zero, 2, 1, 6);

  for (int z = 0; z < 5; z++) {
    for (int y = 0; y < 4; y++) {
      for (int x = 0; x < 1; x++) {
        LSTM->LSTM[Xt_i]->tensor[z][y][x] = inputs[z][y][x];
      }
    }
  }

  for (int z = 0; z < 6; z++) {
    for (int y = 0; y < 1; y++) {
      for (int x = 0; x < 2; x++) {
        LSTM->LSTM[Yt_k]->tensor[z][y][x] = outputs[z][y][x];
      }
    }
  }

  srand(1);
  print_LSTM(LSTM);
  printf("\nFeedforwarding once...\n");
  feedforward_once(LSTM);
  printf("\n\n");
  LSTM = destroy_LSTM(LSTM);
  return 0;
}
