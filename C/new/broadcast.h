// broadcast.h
#ifndef BROADCAST_H
  #define BROADCAST_H
  #include "matrix.h"
  #include <stdarg.h>

  matrix *broadcast_vertical(matrix *, matrix *);
  matrix *broadcast_horizontal(matrix *, matrix *);
  matrix *broadcast_function(long double (*)(long double, long double), matrix *, matrix *);
  matrix *fold(long double (*)(long double, long double), long double (*)(long double), unsigned int, matrix *, ...);

#endif
