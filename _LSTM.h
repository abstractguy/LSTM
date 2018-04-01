// _LSTM.h
#ifndef _LSTM_H
  #define _LSTM_H
  #include "matrix.h"
  //#include <time.h>
  #define TIME_SIZE            4
  #define BATCH_SIZE           1
  #define WORD_SIZE            2
  #define HIDDEN_SIZE          16
  #define LSTM_BEGIN           0
  #define LSTM_SIZE            37
  #define LSTM_END             LSTM_SIZE
  #define GATES_BEGIN          Ht

  #define INPUT_WEIGHTS_BEGIN  Wz
  #define HIDDEN_WEIGHTS_BEGIN Rz
  #define CELL_WEIGHTS_BEGIN   Pi
  #define WEIGHTS_BEGIN        INPUT_WEIGHTS_BEGIN

  #define ERRORS_BEGIN         DHt

  #define INPUT_UPDATES_BEGIN  DWz
  #define HIDDEN_UPDATES_BEGIN DRz
  #define CELL_UPDATES_BEGIN   DPi
  #define UPDATES_BEGIN        INPUT_UPDATES_BEGIN

  #define GATES_END            WEIGHTS_BEGIN

  #define WEIGHTS_END          ERRORS_BEGIN
  #define INPUT_WEIGHTS_END    HIDDEN_WEIGHTS_BEGIN
  #define HIDDEN_WEIGHTS_END   CELL_WEIGHTS_BEGIN
  #define CELL_WEIGHTS_END     WEIGHTS_END

  #define ERRORS_END           UPDATES_BEGIN

  #define UPDATES_END          LSTM_SIZE
  #define INPUT_UPDATES_END    HIDDEN_UPDATES_BEGIN
  #define HIDDEN_UPDATES_END   CELL_UPDATES_BEGIN
  #define CELL_UPDATES_END     UPDATES_END

  typedef enum {
    Ht_backup, Xt, Yt, Ht, Zt, It, Ft, Ot, Ct, Wz, Wi, Wf, Wo, Rz, Ri, Rf,
    Ro, Pi, Pf, Po, DHt, DOt, DCt, DFt, DIt, DZt, DWz, DWi, DWf, DWo, DRz,
    DRi, DRf, DRo, DPi, DPf, DPo
  } index_type;

  typedef struct {
    struct {
      unsigned int time;
      matrix_type **matrix;
    } tensor[LSTM_SIZE];
  } LSTM_type;

  LSTM_type *make_LSTM(long double *, long double *, unsigned int, unsigned int, unsigned int, unsigned int);
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
