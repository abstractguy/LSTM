// printing.c
#include "printing.h"

char *LSTM_string[49] = {
  "Yt_backup", "Input", "Input_reversed", "Xt", "Xt_reversed", "Output",
  "Answer", "Yt", "Zt", "It", "Ft", "Ot", "Ct", "Wz", "Wi", "Wf", "Wo",
  "Rz", "Ri", "Rf", "Ro", "Pi", "Pf", "Po", "Bz", "Bi", "Bf", "Bo", "DYt",
  "DOt", "DCt", "DFt", "DIt", "DZt", "DWz", "DWi", "DWf", "DWo", "DRz",
  "DRi", "DRf", "DRo", "DPi", "DPf", "DPo", "DBz", "DBi", "DBf", "DBo"
};

void print_matrix(matrix_type *matrix) {
  for (unsigned int row = 0; row < matrix->rows; row++) {
    for (unsigned int column = 0; column < matrix->columns; column++) {
      printf("%+5.4Lf ", matrix->matrix[row][column]);
    } putchar('\n');
  }
}

void print_LSTM(LSTM_type *LSTM) {
  for (index_type tensor = 0; tensor < LSTM_SIZE; tensor++) {
    printf("\nType %s:", LSTM_string[tensor]);
    for (unsigned int time = 0; time < LSTM->tensor[tensor].time; time++) {
      printf("\nTime %u:\n", time);
      print_matrix(LSTM->tensor[tensor].matrix[time]);
    } putchar('\n');
  }
}
