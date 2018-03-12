// _feedback.c
#include "_feedback.h"

void feedback(LSTM_type *LSTM) {
  matrix_type *DZt_plus_1 = first(LSTM, DZt), 
              *DIt_plus_1 = first(LSTM, DIt), 
              *DFt_plus_1 = first(LSTM, DFt), 
              *DOt_plus_1 = first(LSTM, DOt), 
              *DCt_plus_1 = first(LSTM, DCt), 
              *Ft_plus_1  = first(LSTM, Ft);

  matrix_for_each(zero, DZt_plus_1);
  matrix_for_each(zero, DIt_plus_1);
  matrix_for_each(one,  DFt_plus_1);
  matrix_for_each(zero, DOt_plus_1);
  matrix_for_each(zero, DCt_plus_1);
  matrix_for_each(zero, Ft_plus_1);

  copy_tensor(LSTM, Output, Answer);

  

  DZt_plus_1 = destroy_matrix(DZt_plus_1), 
  DIt_plus_1 = destroy_matrix(DIt_plus_1), 
  DFt_plus_1 = destroy_matrix(DFt_plus_1), 
  DOt_plus_1 = destroy_matrix(DOt_plus_1), 
  DCt_plus_1 = destroy_matrix(DCt_plus_1), 
  Ft_plus_1  = destroy_matrix(Ft_plus_1);
}

/*
void feedback_once(LSTM_type *LSTM, unsigned int epoch) {
  matrix_type *net_error = 
    subtract(2, matrix_copy(LSTM->tensor[Yt_k].matrix[epoch - 1]), 
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
*/
