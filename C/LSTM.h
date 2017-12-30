// LSTM.h
#ifndef LSTM_H
  #define LSTM_H
  #include <math.h>
  #include <stdlib.h>
  #include <assert.h>

  #define LSTM_SIZE 40
  #define GATES_BEGIN Bt_h
  #define WEIGHTS_BEGIN Wi_iota
  #define GATES_END WEIGHTS_BEGIN
  #define ERRORS_BEGIN Dt_k
  #define WEIGHTS_END ERRORS_BEGIN
  #define UPDATES_BEGIN Ui_iota
  #define ERRORS_END UPDATES_BEGIN
  #define UPDATES_END LSTM_SIZE

  #define INPUT_WEIGHTS_BEGIN WEIGHTS_BEGIN
  #define HIDDEN_WEIGHTS_BEGIN Wh_iota
  #define INPUT_WEIGHTS_END HIDDEN_WEIGHTS_BEGIN
  #define CELL_WEIGHTS_BEGIN Wc_iota
  #define HIDDEN_WEIGHTS_END CELL_WEIGHTS_BEGIN
  #define CELL_WEIGHTS_END WEIGHTS_END

  #define INPUT_UPDATES_BEGIN UPDATES_BEGIN
  #define HIDDEN_UPDATES_BEGIN Uh_iota
  #define INPUT_UPDATES_END HIDDEN_UPDATES_BEGIN
  #define CELL_UPDATES_BEGIN Uc_iota
  #define HIDDEN_UPDATES_END CELL_UPDATES_BEGIN
  #define CELL_UPDATES_END UPDATES_END

  typedef enum {
    Xt_i, Yt_k, Bt_h, St_c, At_iota, Bt_iota, At_phi, Bt_phi, At_c, Bt_c,
    At_omega, Bt_omega, Wi_iota, Wi_phi, Wi_c, Wi_omega, Wh_iota, Wh_phi,
    Wh_c, Wh_omega, Wc_iota, Wc_phi, Wc_omega, Dt_k, Dt_omega, Dt_s, Dt_c,
    Dt_phi, Dt_iota, Ui_iota, Ui_phi, Ui_c, Ui_omega, Uh_iota, Uh_phi,
    Uh_c, Uh_omega, Uc_iota, Uc_phi, Uc_omega
  } index;

  typedef struct {
    int x, y, z;
    long double ***tensor;
  } tensor_3D;

  typedef struct {
    int x, y;
    tensor_3D *LSTM[LSTM_SIZE];
  } LSTM_type;

  long double sigmoid(long double);
  tensor_3D *matrix_sigmoid(tensor_3D *, int);
  tensor_3D *matrix_tanh(tensor_3D *, int);
  long double sigmoid_derivative(long double), tanh_derivative(long double);
  long double random_long_double(void), zero(void), one(void);
  tensor_3D *matrix_dot_product(tensor_3D *, int, tensor_3D *, int);
  tensor_3D *matrix_map(long double (*)(long double), int, tensor_3D *);
  tensor_3D *make_tensor_3D(long double (*)(void), int, int, int);
  tensor_3D *destroy_tensor_3D(tensor_3D *);
  void copy_time_steps(int, tensor_3D *, int, tensor_3D *, int);
  tensor_3D *tensor_deep_copy(tensor_3D *);
  tensor_3D *append_time_step(tensor_3D *, int, tensor_3D *);
  tensor_3D *drop_time_step(tensor_3D *);
  tensor_3D *sum_time_steps(tensor_3D *);
  LSTM_type *make_LSTM(int, int), *destroy_LSTM(LSTM_type *);
#endif
