// file_management.h
#ifndef FILE_MANAGEMENT_H
  #define FILE_MANAGEMENT_H
  #include <stdio.h>
  #include "matrix.h"

  typedef enum {false, true} bool;

  unsigned int count_columns(FILE *);
  matrix_type *parse_matrix(char *);
  bool parse_tab(FILE *, matrix_type *, unsigned int);
  bool parse_newline(FILE *, matrix_type *, unsigned int);
  bool parse_number(FILE *, matrix_type *, unsigned int);
#endif
