// feedback.c
#include "feedback.h"

void feedback_once(LSTM_type *LSTM) {
  matrix *net_error = 
    subtract(2, second(LSTM, Yt_k), 
                second(LSTM, Bt_c));

  // Output errors:
  push(LSTM, Dt_k, 
    sum(5, matrix_copy(net_error), 
           dot_product(first(LSTM, Dt_omega), 
                       transpose(first(LSTM, Wh_omega))), 
           dot_product(first(LSTM, Dt_c), 
                       transpose(first(LSTM, Wh_c))), 
           dot_product(first(LSTM, Dt_phi), 
                       transpose(first(LSTM, Wh_phi))), 
           dot_product(first(LSTM, Dt_iota), 
                       transpose(first(LSTM, Wh_iota)))));

  net_error = destroy_matrix(net_error);

  // Output gate errors:
  push(LSTM, Dt_omega, 
    product(2, sigmoid_derivative(second(LSTM, At_omega)), 
               dot_product(first(LSTM, Dt_k), 
                           transpose(matrix_tanh(second(LSTM, St_c))))));

  // Cell state errors:
  push(LSTM, Dt_s, 
    sum(5, product(3, second(LSTM, Bt_omega), 
                      sigmoid_derivative(second(LSTM, St_c)), 
                      first(LSTM, Dt_k)), 
           product(2, first(LSTM, Bt_phi), 
                      second(LSTM, Dt_s)), 
           product(2, transpose(first(LSTM, Wc_iota)), 
                      second(LSTM, Dt_iota)), 
           product(2, transpose(first(LSTM, Wc_phi)), 
                      second(LSTM, Dt_phi)), 
           product(2, transpose(first(LSTM, Wc_omega)), 
                      first(LSTM, Dt_omega))));

  // Cell output errors:
  push(LSTM, Dt_c, 
    product(3, second(LSTM, Bt_iota), 
               tanh_derivative(second(LSTM, At_c)), 
               first(LSTM, Dt_s)));

  // Forget gate errors:
  push(LSTM, Dt_phi, 
    product(2, sigmoid_derivative(second(LSTM, At_phi)), 
               dot_product(first(LSTM, Dt_s), 
                           transpose(third(LSTM, St_c)))));

  // Input gate errors:
  push(LSTM, Dt_iota, 
    product(2, sigmoid_derivative(second(LSTM, At_iota)), 
               dot_product(first(LSTM, Dt_s), 
                           transpose(matrix_tanh(second(LSTM, At_c))))));
}

/*
void feedback_once(LSTM_type *LSTM) {
  matrix *Bt_plus_1_c     = pop(LSTM, Bt_c), 
         *St_plus_1_c     = pop(LSTM, St_c), 
         *At_plus_1_iota  = pop(LSTM, At_iota), 
         *Bt_plus_1_iota  = pop(LSTM, Bt_iota), 
         *At_plus_1_phi   = pop(LSTM, At_phi), 
         *Bt_plus_1_phi   = pop(LSTM, Bt_phi), 
         *At_plus_1_c     = pop(LSTM, At_c), 
         *At_plus_1_omega = pop(LSTM, At_omega), 
         *Bt_plus_1_omega = pop(LSTM, Bt_omega), 
         *Dt_plus_1_k     = pop(LSTM, Dt_k), 
         *Dt_plus_1_c     = pop(LSTM, Dt_c),
         *Dt_plus_1_s     = pop(LSTM, Dt_s), 
         *Dt_plus_1_omega = pop(LSTM, Dt_omega), 
         *Dt_plus_1_phi   = pop(LSTM, Dt_phi), 
         *Dt_plus_1_iota  = pop(LSTM, Dt_iota);

  // FEEDBACK:
  // Output errors:
  push(LSTM, Dt_k, 
    sum(5, subtract(2, first(LSTM, Yt_k), 
                       first(LSTM, Bt_c)), 
           dot_product(matrix_copy(Dt_plus_1_omega), 
                       transpose(first(LSTM, Wh_omega))), 
           dot_product(matrix_copy(Dt_plus_1_c), 
                       transpose(first(LSTM, Wh_c))), 
           dot_product(matrix_copy(Dt_plus_1_phi), 
                       transpose(first(LSTM, Wh_phi))), 
           dot_product(matrix_copy(Dt_plus_1_iota), 
                       transpose(first(LSTM, Wh_iota)))));

  // Output gate errors:
  push(LSTM, Dt_omega, 
    product(2, sigmoid_derivative(first(LSTM, At_omega)), 
               dot_product(matrix_copy(Dt_plus_1_k), 
                           transpose(matrix_tanh(first(LSTM, St_c))))));

//<---- END OF TRANSLATION
//-- TRANSLATE THE REST --
//TRANSLATE THIS -------->

  // Cell state errors:
  push(LSTM, Dt_s, 
    sum(5, product(3, second(LSTM, Bt_omega), 
                      sigmoid_derivative(second(LSTM, St_c)), 
                      first(LSTM, Dt_k)), 
           product(2, first(LSTM, Bt_phi), 
                      second(LSTM, Dt_s)), 
           product(2, transpose(first(LSTM, Wc_iota)), 
                      second(LSTM, Dt_iota)), 
           product(2, transpose(first(LSTM, Wc_phi)), 
                      second(LSTM, Dt_phi)), 
           product(2, transpose(first(LSTM, Wc_omega)), 
                      first(LSTM, Dt_omega))));

  // Cell output errors:
  push(LSTM, Dt_c, 
    product(3, second(LSTM, Bt_iota), 
               tanh_derivative(second(LSTM, At_c)), 
               first(LSTM, Dt_s)));

  // Forget gate errors:
  push(LSTM, Dt_phi, 
    product(2, sigmoid_derivative(second(LSTM, At_phi)), 
               dot_product(first(LSTM, Dt_s), 
                           transpose(third(LSTM, St_c)))));

  // Input gate errors:
  push(LSTM, Dt_iota, 
    product(2, sigmoid_derivative(second(LSTM, At_iota)), 
               dot_product(first(LSTM, Dt_s), 
                           transpose(matrix_tanh(second(LSTM, At_c))))));

  // UPDATES:
  // Calculate input gradients
  push(LSTM, Ui_iota, 
    sum(2, pop(LSTM, Ui_iota), 
           dot_product(transpose(second(LSTM, Xt_i)), 
                       second(LSTM, Dt_iota))));
  push(LSTM, Ui_phi, 
    sum(2, pop(LSTM, Ui_phi), 
           dot_product(transpose(second(LSTM, Xt_i)), 
                       second(LSTM, Dt_phi))));
  push(LSTM, Ui_c, 
    sum(2, pop(LSTM, Ui_c), 
           dot_product(transpose(second(LSTM, Xt_i)), 
                       second(LSTM, Dt_c))));
  push(LSTM, Ui_omega, 
    sum(2, pop(LSTM, Ui_omega), 
           dot_product(transpose(second(LSTM, Xt_i)), 
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

  Bt_plus_1_c     = destroy_matrix(Bt_plus_1_c);
  St_plus_1_c     = destroy_matrix(St_plus_1_c);
  At_plus_1_iota  = destroy_matrix(At_plus_1_iota);
  Bt_plus_1_iota  = destroy_matrix(Bt_plus_1_iota);
  At_plus_1_phi   = destroy_matrix(At_plus_1_phi);
  Bt_plus_1_phi   = destroy_matrix(Bt_plus_1_phi);
  At_plus_1_c     = destroy_matrix(At_plus_1_c);
  At_plus_1_omega = destroy_matrix(At_plus_1_omega);
  Bt_plus_1_omega = destroy_matrix(Bt_plus_1_omega);
  Dt_plus_1_k     = destroy_matrix(Dt_plus_1_k);
  Dt_plus_1_c     = destroy_matrix(Dt_plus_1_c);
  Dt_plus_1_s     = destroy_matrix(Dt_plus_1_s);
  Dt_plus_1_omega = destroy_matrix(Dt_plus_1_omega);
  Dt_plus_1_phi   = destroy_matrix(Dt_plus_1_phi);
  Dt_plus_1_iota  = destroy_matrix(Dt_plus_1_iota);
}

void feedback(LSTM_type *LSTM, unsigned int epoch) {
  matrix *temp = NULL;
  unsigned int i = epoch;

  for (index tensor = GATES_BEGIN; tensor < GATES_END; tensor++) {
    temp = matrix_copy_shape(LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time - 1]);
    matrix_for_each(zero, temp);
    push(LSTM, tensor, temp);
  }

  for (index tensor = ERRORS_BEGIN; tensor < ERRORS_END; tensor++) {
    temp = matrix_copy_shape(LSTM->tensor[tensor].matrix[LSTM->tensor[tensor].time - 1]);
    matrix_for_each(zero, temp);
    push(LSTM, tensor, temp);
  }

  while (i--) feedback_once(LSTM);
}
*/
