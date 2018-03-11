// _LSTM.h
#ifndef _LSTM_H
  #define _LSTM_H
  #include "matrix.h"
  //#include <time.h>
  #define LSTM_BEGIN           0
  #define LSTM_SIZE            50
  #define LSTM_END             LSTM_SIZE
  #define GATES_BEGIN          Yt

  #define INPUT_WEIGHTS_BEGIN  Wz
  #define HIDDEN_WEIGHTS_BEGIN Rz
  #define CELL_WEIGHTS_BEGIN   Pi
  #define BIAS_WEIGHTS_BEGIN   Bz
  #define WEIGHTS_BEGIN        INPUT_WEIGHTS_BEGIN

  #define ERRORS_BEGIN         DYt

  #define INPUT_UPDATES_BEGIN  DW_z
  #define HIDDEN_UPDATES_BEGIN DR_z
  #define CELL_UPDATES_BEGIN   DPi
  #define BIAS_UPDATES_BEGIN   DB_z
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
    Input, Xt, Output, Answer, Yt, _Zt, _It, _Ft, _Ot, Zt, It, Ft, Ot, Ct, 
    Wz, Wi, Wf, Wo, Rz, Ri, Rf, Ro, Pi, Pf, Po, Bz, Bi, Bf, Bo, DYt, D_Ot, 
    DCt, D_Ft, D_It, D_Zt, DW_z, DW_i, DW_f, DW_o, DR_z, DR_i, DR_f, DR_o, 
    DPi, DPf, DPo, DB_z, DB_i, DB_f, DB_o
  } index_type;

  typedef struct {
    struct {
      unsigned int time;
      matrix_type **matrix;
    } tensor[LSTM_SIZE];
  } LSTM_type;

  LSTM_type *make_LSTM(long double *, long double *, unsigned int, unsigned int, unsigned int);
  LSTM_type *destroy_LSTM(LSTM_type *);
  void LSTM_initialize(LSTM_type *, index_type, index_type, long double (*)(long double), unsigned int, unsigned int, unsigned int);
  matrix_type *first(LSTM_type *, index_type);
  matrix_type *second(LSTM_type *, index_type);
  matrix_type *third(LSTM_type *, index_type);
  void push(LSTM_type *, index_type, matrix_type *);
  matrix_type *pop(LSTM_type *, index_type);
  void push_all(LSTM_type *, index_type, long double *);
  void LSTM_copy_last_matrix_to_beginning(LSTM_type *, index_type, index_type);
  //void copy_tensor(LSTM_type *, index_type, index_type);
#endif
