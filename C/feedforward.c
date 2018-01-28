// feedforward.c
#include "feedforward.h"

void feedforward_once(LSTM_type *LSTM) {
  // Input preactivations:
  push(LSTM, At_iota, 
    transpose(
      sum(3, 
        product(2, 
          transpose(first(LSTM, Xt_i)), 
          first(LSTM, Wi_iota)), 
        product(2, 
          transpose(second(LSTM, Bt_h)), 
          first(LSTM, Wh_iota)), 
        product(2, 
          transpose(second(LSTM, St_c)), 
          first(LSTM, Wc_iota)))));

  // Input activations:
  push(LSTM, Bt_iota, matrix_sigmoid(first(LSTM, At_iota)));

  // Forget preactivations:
  push(LSTM, At_phi, 
    transpose(
      sum(3, 
        product(2, 
          transpose(first(LSTM, Xt_i)), 
          first(LSTM, Wi_phi)), 
        product(2, 
          transpose(second(LSTM, Bt_h)), 
          first(LSTM, Wh_phi)), 
        product(2, 
          transpose(second(LSTM, St_c)), 
          first(LSTM, Wc_phi)))));

  // Forget activations:
  push(LSTM, Bt_phi, matrix_sigmoid(first(LSTM, At_phi)));

  // Cell preactivations:
  push(LSTM, At_c, 
    //transpose(
    sum(2, 
      product(2, 
        transpose(first(LSTM, Xt_i)), 
        first(LSTM, Wi_c)), 
      product(2, 
        transpose(second(LSTM, Bt_h)), 
        first(LSTM, Wh_c))));
    //);

  // Cell activations:
  push(LSTM, St_c, 
    transpose(
      sum(2, 
        product(2, 
          transpose(matrix_tanh(first(LSTM, At_c))), 
          transpose(first(LSTM, Bt_iota))), 
        product(2, 
          transpose(second(LSTM, St_c)), 
          transpose(first(LSTM, Bt_phi))))));

  // Output preactivations:
  push(LSTM, At_omega, 
    sum(3, 
      product(2, 
        transpose(first(LSTM, Xt_i)), 
        first(LSTM, Wi_omega)), 
      product(2, 
        transpose(second(LSTM, Bt_h)), 
        first(LSTM, Wh_omega)), 
      product(2, 
        transpose(first(LSTM, St_c)), 
        first(LSTM, Wc_omega))));

  // Output activations:
  push(LSTM, Bt_omega, matrix_sigmoid(first(LSTM, At_omega)));

  // Cell outputs:
  push(LSTM, Bt_c, 
    transpose(
      product(2, 
        transpose(matrix_sigmoid(first(LSTM, St_c))), 
        transpose(first(LSTM, Bt_omega)))));

}
