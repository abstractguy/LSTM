// update.c
#include "update.h"

void update_forward_once(LSTM_type *LSTM) {
  // Calculate input gradients
  push(LSTM, Ui_iota, 
    dot_product(
      transpose(second(LSTM, Xt_i)), 
      second(LSTM, Dt_iota)));
  push(LSTM, Ui_phi, 
    dot_product(
      transpose(second(LSTM, Xt_i)), 
      second(LSTM, Dt_phi)));
  push(LSTM, Ui_c, 
    dot_product(
      transpose(second(LSTM, Xt_i)), 
      second(LSTM, Dt_c)));
  push(LSTM, Ui_omega, 
    dot_product(
      transpose(second(LSTM, Xt_i)), 
      second(LSTM, Dt_omega)));

  // Calculate recurrent gradients
  push(LSTM, Uh_iota, 
    dot_product(
      transpose(second(LSTM, Yt_k)), 
      first(LSTM, Dt_iota)));
  push(LSTM, Uh_phi, 
    dot_product(
      transpose(second(LSTM, Yt_k)), 
      first(LSTM, Dt_phi)));
  push(LSTM, Uh_c, 
    dot_product(
      transpose(second(LSTM, Yt_k)), 
      first(LSTM, Dt_c)));
  push(LSTM, Uh_omega, 
    dot_product(
      transpose(second(LSTM, Yt_k)), 
      first(LSTM, Dt_omega)));

  // Calculate cell input gradients
  push(LSTM, Uc_iota, 
    dot_product(
      transpose(second(LSTM, St_c)), 
      first(LSTM, Dt_iota)));

  // Calculate cell forget gradients
  push(LSTM, Uc_phi, 
    dot_product(
      transpose(second(LSTM, St_c)), 
      first(LSTM, Dt_phi)));

  // Calculate cell output gradients
  push(LSTM, Uc_omega, 
    dot_product(
      transpose(second(LSTM, St_c)), 
      second(LSTM, Dt_omega)));
}

void update_backward_once(LSTM_type *LSTM) {
  
}
