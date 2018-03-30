// LSTM.h
#ifndef LSTM_H
  #define LSTM_H
  #include "matrix.h"
  //#include <time.h>
  #define TIME_SIZE            4
  #define BATCH_SIZE           1
  #define WORD_SIZE            2
  #define HIDDEN_SIZE          16
  #define LSTM_BEGIN           0
  #define LSTM_SIZE            53
  #define LSTM_END             LSTM_SIZE
  #define GATES_BEGIN          Yt

  #define INPUT_WEIGHTS_BEGIN  Wz
  #define HIDDEN_WEIGHTS_BEGIN Rz
  #define CELL_WEIGHTS_BEGIN   Pi
  #define BIAS_WEIGHTS_BEGIN   Bz
  #define WEIGHTS_BEGIN        INPUT_WEIGHTS_BEGIN

  #define ERRORS_BEGIN         DYt

  #define INPUT_UPDATES_BEGIN  DWz
  #define HIDDEN_UPDATES_BEGIN DRz
  #define CELL_UPDATES_BEGIN   DPi
  #define BIAS_UPDATES_BEGIN   DBz
  #define UPDATES_BEGIN        INPUT_UPDATES_BEGIN

  #define GATES_END            WEIGHTS_BEGIN

  #define WEIGHTS_END          ERRORS_BEGIN
  #define INPUT_WEIGHTS_END    HIDDEN_WEIGHTS_BEGIN
  #define HIDDEN_WEIGHTS_END   CELL_WEIGHTS_BEGIN
  #define CELL_WEIGHTS_END     BIAS_WEIGHTS_BEGIN
  #define BIAS_WEIGHTS_END     WEIGHTS_END

  #define ERRORS_END           UPDATES_BEGIN

  #define UPDATES_END          LSTM_SIZE
  #define INPUT_UPDATES_END    HIDDEN_UPDATES_BEGIN
  #define HIDDEN_UPDATES_END   CELL_UPDATES_BEGIN
  #define CELL_UPDATES_END     BIAS_UPDATES_BEGIN
  #define BIAS_UPDATES_END     UPDATES_END

  typedef enum {
    Yt_backup, Input, Input_reversed, Xt, Xt_reversed, Output, Answer, Yt,
    _Zt, _It, _Ft, _Ot, Zt, It, Ft, Ot, Ct, Wz, Wi, Wf, Wo, Rz, Ri, Rf,
    Ro, Pi, Pf, Po, Bz, Bi, Bf, Bo, DYt, DOt, DCt, DFt, DIt, DZt, DWz,
    DWi, DWf, DWo, DRz, DRi, DRf, DRo, DPi, DPf, DPo, DBz, DBi, DBf, DBo
  } index_type;

  typedef struct {
    struct {
      unsigned int time;
      matrix_type **matrix;
    } tensor[LSTM_SIZE];
  } LSTM_type;

  LSTM_type *make_LSTM(long double *, long double *, long double *, unsigned int, unsigned int, unsigned int, unsigned int);
  void destroy_LSTM(LSTM_type *);
  void LSTM_initialize(LSTM_type *, index_type, index_type, long double (*)(long double), unsigned int, unsigned int, unsigned int);
  unsigned int convert_index(LSTM_type *, index_type, long);
  matrix_type *LSTM_read(LSTM_type *, index_type, long);
  void LSTM_write(LSTM_type *, index_type, long, matrix_type *);
  void push(LSTM_type *, index_type, matrix_type *);
  matrix_type *pop(LSTM_type *, index_type);
  void push_all(LSTM_type *, index_type, long double *);
  void empty_tensor(LSTM_type *, index_type);
  void copy_tensor(LSTM_type *, index_type, index_type);
#endif
