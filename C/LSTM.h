// LSTM.h
#ifndef LSTM_H
  #define LSTM_H
  #include "matrix.h"
  #define LSTM_SIZE            40
  #define INPUTS_BEGIN         Xt_i
  #define OUTPUTS_BEGIN        Yt_k
  #define OUTPUT_ERRORS_BEGIN  Dt_k
  #define GATES_BEGIN          Bt_h
  #define INPUT_WEIGHTS_BEGIN  Wi_iota
  #define HIDDEN_WEIGHTS_BEGIN Wh_iota
  #define CELL_WEIGHTS_BEGIN   Wc_iota
  #define ERRORS_BEGIN         Dt_omega
  #define INPUT_UPDATES_BEGIN  Ui_iota
  #define HIDDEN_UPDATES_BEGIN Uh_iota
  #define CELL_UPDATES_BEGIN   Uc_iota
  #define INPUTS_END           OUTPUTS_BEGIN
  #define OUTPUTS_END          OUTPUT_ERRORS_BEGIN
  #define OUTPUT_ERRORS_END    GATES_BEGIN
  #define GATES_END            INPUT_WEIGHTS_BEGIN
  #define INPUT_WEIGHTS_END    HIDDEN_WEIGHTS_BEGIN
  #define HIDDEN_WEIGHTS_END   CELL_WEIGHTS_BEGIN
  #define CELL_WEIGHTS_END     ERRORS_BEGIN
  #define ERRORS_END           INPUT_UPDATES_BEGIN
  #define INPUT_UPDATES_END    HIDDEN_UPDATES_BEGIN
  #define HIDDEN_UPDATES_END   CELL_UPDATES_BEGIN
  #define CELL_UPDATES_END     LSTM_SIZE
  #define WEIGHTS_BEGIN        INPUT_WEIGHTS_BEGIN
  #define WEIGHTS_END          CELL_WEIGHTS_END
  #define UPDATES_BEGIN        INPUT_UPDATES_BEGIN
  #define UPDATES_END          CELL_UPDATES_END

  typedef enum {
    Xt_i, Yt_k, Bt_c, St_c, Dt_k, Dt_c, Dt_s, Bt_h, At_iota, Bt_iota,
    At_phi, Bt_phi, At_c, At_omega, Bt_omega, Wi_iota, Wi_phi, Wi_c,
    Wi_omega, Wh_iota, Wh_phi, Wh_c, Wh_omega, Wc_iota, Wc_phi, Wc_omega,
    Dt_omega, Dt_phi, Dt_iota, Ui_iota, Ui_phi, Ui_c, Ui_omega, Uh_iota,
    Uh_phi, Uh_c, Uh_omega, Uc_iota, Uc_phi, Uc_omega
  } index;

  typedef struct {
    struct {
      unsigned int time;
      matrix **matrix;
    } tensor[LSTM_SIZE];
  } LSTM_type;

  LSTM_type *make_LSTM(unsigned int, unsigned int);
  LSTM_type *destroy_LSTM(LSTM_type *);
  void LSTM_initialize(LSTM_type *, index, index, long double (*)(long double), unsigned int, unsigned int, unsigned int);
  matrix *first(LSTM_type *, index);
  matrix *second(LSTM_type *, index);
  matrix *third(LSTM_type *, index);
  void push(LSTM_type *, index, matrix *);
  matrix *pop(LSTM_type *, index);
  void push_all(LSTM_type *, index, long double *);

#endif
