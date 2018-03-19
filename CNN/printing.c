// printing.c
#include "printing.h"

void print_matrix(char *string, matrix_type *matrix) {
  puts(string);
  for (unsigned int row = 0; row < matrix->rows; row++) {
    for (unsigned int column = 0; column < matrix->columns; column++) {
      printf("%+5.4Lf ", matrix->matrix[row][column]);
    } putchar('\n');
  }
}
