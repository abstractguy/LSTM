// file_management.c
#include "file_management.h"

unsigned int count_columns(FILE *fp) {
  unsigned int column = 1;
  char byte = 0;
  do {byte = fgetc(fp);
      if (byte == '\t') column++;
  } while (!feof(fp) && byte != '\r');
  assert(column);
  fseek(fp, 0, SEEK_SET);
  return column;
}

matrix_type *parse_matrix(char *input_file) {
  matrix_type *matrix = NULL;
  FILE *fp = NULL;
  unsigned int column = 0;
  fp = fopen(input_file, "r");
  assert(fp);
  assert(!feof(fp));
  matrix = make_matrix(1, count_columns(fp));
  assert(!parse_newline(fp, matrix, column));
  assert(!parse_tab(fp, matrix, column));
  do {assert(parse_number(fp, matrix, column));
      assert(parse_tab(fp, matrix, column) 
          || parse_newline(fp, matrix, column) 
          || feof(fp));
  } while (!feof(fp));
  fclose(fp);
  return matrix;
}

bool parse_tab(FILE *fp, matrix_type *matrix, unsigned int column) {
  if (fgetc(fp) == '\t') {
    column++;
    return true;
  } else {
    NOT_USED(column);
    NOT_USED(matrix);
    fseek(fp, ftell(fp), -1);
    return false;
  }
}

bool parse_newline(FILE *fp, matrix_type *matrix, unsigned int column) {
  if (fgetc(fp) == '\r') {
    assert(fgetc(fp) == '\n');
    NOT_USED(fgetc(fp));
    if (feof(fp)) {
      NOT_USED(matrix);
      NOT_USED(column);
      return false;
    } else {
      fseek(fp, ftell(fp), -1);
      matrix->matrix = realloc(matrix->matrix, sizeof(long double *) * (matrix->rows + 1));
      for (column = 0; column < matrix->columns; column++) {
        matrix->matrix[matrix->rows] = realloc(matrix->matrix[matrix->rows - 1], sizeof(long double) * matrix->columns);
        matrix->matrix[matrix->rows][column] = 0.0;
      }
      matrix->rows++;
      column = 0;
      return true;
    }
  } else {
    NOT_USED(column);
    NOT_USED(matrix);
    fseek(fp, ftell(fp), -1);
    return false;
  }
}

bool parse_number(FILE *fp, matrix_type *matrix, unsigned int column) {
  char string[20] = {0};
  long int position = ftell(fp);
  assert(column < matrix->columns);
  if (fscanf(fp, "%s", string)) {
    matrix->matrix[matrix->rows - 1][column] = strtold(string, NULL);
    return true;
  } else {
    fseek(fp, position, 0);
    return false;
  }
}
