// feedforward.c
#include "feedforward.h"

void feedforward_once(LSTM_type *LSTM) {
  matrix *matrix1 = NULL;
  matrix *matrix2 = NULL;
  matrix *matrix3 = NULL;
  matrix *matrix4 = NULL;

  // Input preactivations:
  matrix1 = product(2, first(LSTM, Xt_i),  first(LSTM, Wi_iota));
  matrix2 = product(2, second(LSTM, Bt_h), first(LSTM, Wh_iota));
  matrix3 = product(2, second(LSTM, St_c), first(LSTM, Wc_iota));
  matrix4 = sum(3, matrix1, matrix2, matrix3);
  push(LSTM, At_iota, matrix4);
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);
  matrix3 = destroy_matrix(matrix3);

  // Input activations:
  push(LSTM, Bt_iota, matrix_sigmoid(first(LSTM, At_iota)));

  // Forget preactivations:
  matrix1 = product(2, first(LSTM, Xt_i),  first(LSTM, Wi_phi));
  matrix2 = product(2, second(LSTM, Bt_h), first(LSTM, Wh_phi));
  matrix3 = product(2, second(LSTM, St_c), first(LSTM, Wc_phi));
  matrix4 = sum(3, matrix1, matrix2, matrix3);
  push(LSTM, At_phi, matrix4);
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);
  matrix3 = destroy_matrix(matrix3);

  // Forget activations:
  push(LSTM, Bt_phi, matrix_sigmoid(first(LSTM, At_phi)));

  // Cell preactivations:
  matrix1 = product(2, first(LSTM, Xt_i),  first(LSTM, Wi_c));
  matrix2 = product(2, second(LSTM, Bt_h), first(LSTM, Wh_c));
  matrix3 = sum(2, matrix1, matrix2);
  push(LSTM, At_c, matrix3);
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);

  // Cell activations:
  matrix1 =
    product(2, 
      matrix_tanh(first(LSTM, At_c)), 
      first(LSTM, Bt_iota));
  matrix2 =
    product(2, 
      second(LSTM, St_c), 
      first(LSTM, Bt_phi));
  matrix3 = sum(2, matrix1, matrix2);
  push(LSTM, St_c, matrix3);
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);

  // Output preactivations:
  matrix1 = product(2, first(LSTM, Xt_i),  first(LSTM, Wi_omega));
  matrix2 = product(2, second(LSTM, Bt_h), first(LSTM, Wh_omega));
  matrix3 = product(2, first(LSTM, St_c),  first(LSTM, Wc_omega));
  matrix4 = sum(3, matrix1, matrix2, matrix3);
  push(LSTM, At_omega, matrix4);
  matrix1 = destroy_matrix(matrix1);
  matrix2 = destroy_matrix(matrix2);
  matrix3 = destroy_matrix(matrix3);

  // Output activations:
  push(LSTM, Bt_omega, matrix_sigmoid(first(LSTM, At_omega)));

  // Cell outputs:
  matrix1 =
    product(2,
      matrix_sigmoid(first(LSTM, St_c)),
      first(LSTM, Bt_omega));
  push(LSTM, Bt_c, matrix1);

}
