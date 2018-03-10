// update.c
#include "update.h"

void update_forward_once(LSTM_type *LSTM, unsigned int epoch) {
  // Calculate input gradients
  push(LSTM, Ui_iota, 
    sum(2, pop(LSTM, Ui_iota), 
           dot_product(transpose(matrix_copy(LSTM->tensor[Xt_i].matrix[epoch - 1])), 
                       second(LSTM, Dt_iota))));
  push(LSTM, Ui_phi, 
    sum(2, pop(LSTM, Ui_phi), 
           dot_product(transpose(matrix_copy(LSTM->tensor[Xt_i].matrix[epoch - 1])), 
                       second(LSTM, Dt_phi))));
  push(LSTM, Ui_c, 
    sum(2, pop(LSTM, Ui_c), 
           dot_product(transpose(matrix_copy(LSTM->tensor[Xt_i].matrix[epoch - 1])), 
                       second(LSTM, Dt_c))));
  push(LSTM, Ui_omega, 
    sum(2, pop(LSTM, Ui_omega), 
           dot_product(transpose(matrix_copy(LSTM->tensor[Xt_i].matrix[epoch - 1])), 
                       second(LSTM, Dt_omega))));

  // Calculate recurrent gradients
  push(LSTM, Uh_iota, 
    sum(2, pop(LSTM, Uh_iota), 
           dot_product(transpose(second(LSTM, Bt_c)), 
                       first(LSTM, Dt_iota))));
  push(LSTM, Uh_phi, 
    sum(2, pop(LSTM, Uh_phi), 
           dot_product(transpose(second(LSTM, Bt_c)), 
                       first(LSTM, Dt_phi))));
  push(LSTM, Uh_c, 
    sum(2, pop(LSTM, Uh_c), 
           dot_product(transpose(second(LSTM, Bt_c)), 
                       first(LSTM, Dt_c))));
  push(LSTM, Uh_omega, 
    sum(2, pop(LSTM, Uh_omega), 
           dot_product(transpose(second(LSTM, Bt_c)), 
                       first(LSTM, Dt_omega))));

  // Calculate cell input gradients
  push(LSTM, Uc_iota, 
    sum(2, pop(LSTM, Uc_iota), 
           dot_product(transpose(second(LSTM, St_c)), 
                       first(LSTM, Dt_iota))));

  // Calculate cell forget gradients
  push(LSTM, Uc_phi, 
    sum(2, pop(LSTM, Uc_phi), 
           dot_product(transpose(second(LSTM, St_c)), 
                       first(LSTM, Dt_phi))));

  // Calculate cell output gradients
  push(LSTM, Uc_omega, 
    sum(2, pop(LSTM, Uc_omega), 
           dot_product(transpose(second(LSTM, St_c)), 
                       second(LSTM, Dt_omega))));
}

void update_backward_once(LSTM_type *LSTM) {
  matrix_type *learning_rate = make_matrix(1, 1);
  learning_rate->matrix[0][0] = 0.1;
  // Update input gates
  push(LSTM, Wi_iota, 
    sum(2, pop(LSTM, Wi_iota), 
           product(2, matrix_copy(learning_rate), first(LSTM, Ui_iota))));
  matrix_for_each(zero, LSTM->tensor[Ui_iota].matrix[LSTM->tensor[Ui_iota].time - 1]);
  push(LSTM, Wi_phi, 
    sum(2, pop(LSTM, Wi_phi), 
           product(2, matrix_copy(learning_rate), first(LSTM, Ui_phi))));
  matrix_for_each(zero, LSTM->tensor[Ui_phi].matrix[LSTM->tensor[Ui_phi].time - 1]);
  push(LSTM, Wi_c, 
    sum(2, pop(LSTM, Wi_c), 
           product(2, matrix_copy(learning_rate), first(LSTM, Ui_c))));
  matrix_for_each(zero, LSTM->tensor[Ui_c].matrix[LSTM->tensor[Ui_c].time - 1]);
  push(LSTM, Wi_omega, 
    sum(2, pop(LSTM, Wi_omega), 
           product(2, matrix_copy(learning_rate), first(LSTM, Ui_omega))));
  matrix_for_each(zero, LSTM->tensor[Ui_omega].matrix[LSTM->tensor[Ui_omega].time - 1]);

  // Update recurrent gates
  push(LSTM, Wh_iota, 
    sum(2, pop(LSTM, Wh_iota), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uh_iota))));
  matrix_for_each(zero, LSTM->tensor[Uh_iota].matrix[LSTM->tensor[Uh_iota].time - 1]);
  push(LSTM, Wh_phi, 
    sum(2, pop(LSTM, Wh_phi), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uh_phi))));
  matrix_for_each(zero, LSTM->tensor[Uh_phi].matrix[LSTM->tensor[Uh_phi].time - 1]);
  push(LSTM, Wh_c, 
    sum(2, pop(LSTM, Wh_c), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uh_c))));
  matrix_for_each(zero, LSTM->tensor[Uh_c].matrix[LSTM->tensor[Uh_c].time - 1]);
  push(LSTM, Wh_omega, 
    sum(2, pop(LSTM, Wh_omega), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uh_omega))));
  matrix_for_each(zero, LSTM->tensor[Uh_omega].matrix[LSTM->tensor[Wh_omega].time - 1]);

  // Update cell gates
  push(LSTM, Wc_iota, 
    sum(2, pop(LSTM, Wc_iota), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uc_iota))));
  matrix_for_each(zero, LSTM->tensor[Uc_iota].matrix[LSTM->tensor[Uc_iota].time - 1]);
  push(LSTM, Wc_phi, 
    sum(2, pop(LSTM, Wc_phi), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uc_phi))));
  matrix_for_each(zero, LSTM->tensor[Uc_phi].matrix[LSTM->tensor[Uc_phi].time - 1]);
  push(LSTM, Wc_omega, 
    sum(2, pop(LSTM, Wc_omega), 
           product(2, matrix_copy(learning_rate), first(LSTM, Uc_omega))));
  matrix_for_each(zero, LSTM->tensor[Uc_omega].matrix[LSTM->tensor[Uc_omega].time - 1]);

  learning_rate = destroy_matrix(learning_rate);
}
