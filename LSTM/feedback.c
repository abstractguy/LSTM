// feedback.c
#include "feedback.h"

void feedback(LSTM_type *LSTM) {
  matrix_type *DZt_plus_1 = LSTM_read(LSTM, DZt, -1), 
              *DIt_plus_1 = LSTM_read(LSTM, DIt, -1), 
              *DFt_plus_1 = LSTM_read(LSTM, DFt, -1), 
              *DOt_plus_1 = LSTM_read(LSTM, DOt, -1), 
              *DCt_plus_1 = LSTM_read(LSTM, DCt, -1), 
              *Ft_plus_1  = LSTM_read(LSTM, Ft,  -1);
  long time = LSTM->tensor[Yt].time;

  matrix_for_each(zero, DZt_plus_1);
  matrix_for_each(zero, DIt_plus_1);
  matrix_for_each(zero, DFt_plus_1);
  matrix_for_each(zero, DOt_plus_1);
  matrix_for_each(zero, DCt_plus_1);
  matrix_for_each(zero, Ft_plus_1);

  for (long t = 0; t < time; t++) {
    // FEEDBACK PART:
    // Block output errors:
    push(LSTM, DHt, 
      sum(5, 
        pop(LSTM, DHt), 
        dot_product(
          matrix_copy(DZt_plus_1), 
          transpose(LSTM_read(LSTM, Rz, 0))),
        dot_product(
          matrix_copy(DIt_plus_1), 
          transpose(LSTM_read(LSTM, Ri, 0))),
        dot_product(
          matrix_copy(DFt_plus_1), 
          transpose(LSTM_read(LSTM, Rf, 0))),
        dot_product(
          matrix_copy(DOt_plus_1), 
          transpose(LSTM_read(LSTM, Ro, 0)))));

    // Output gate errors:
    push(LSTM, DOt, 
      product(3, 
        LSTM_read(LSTM, DHt, -1), 
        matrix_tanh(LSTM_read(LSTM, Ct, -1)), 
        sigmoid_derivative(LSTM_read(LSTM, Ot, -1))));

    // Cell memory errors:
    push(LSTM, DCt, 
      sum(5, 
        product(3, 
          pop(LSTM, DHt), 
          pop(LSTM, Ot), 
          tanh_derivative(LSTM_read(LSTM, Ct, -1))), 
        product(2, 
          LSTM_read(LSTM, Po, -1), 
          LSTM_read(LSTM, DOt, -1)), 
        product(2, 
          LSTM_read(LSTM, Pi, -1), 
          matrix_copy(DIt_plus_1)), 
        product(2, 
          LSTM_read(LSTM, Pf, -1), 
          matrix_copy(DFt_plus_1)), 
        product(2, 
          DCt_plus_1, 
          Ft_plus_1)));

    // Forget gate errors:
    push(LSTM, DFt, 
      product(3, 
        LSTM_read(LSTM, DCt, -1), 
        LSTM_read(LSTM, Ct, -2), 
        sigmoid_derivative(LSTM_read(LSTM, Ft, -1))));

    // Input gate errors:
    push(LSTM, DIt, 
      product(3, 
        LSTM_read(LSTM, DCt, -1), 
        LSTM_read(LSTM, Zt, -1), 
        sigmoid_derivative(LSTM_read(LSTM, It, -1))));

    // Block input errors:
    push(LSTM, DZt, 
      product(3, 
        LSTM_read(LSTM, DCt, -1), 
        pop(LSTM, It), 
        tanh_derivative(pop(LSTM, Zt))));

    // UPDATE PART:
    // Block input input weight updates:
    push(LSTM, DWz, 
      sum(2, 
        pop(LSTM, DWz), 
        dot_product(
          transpose(LSTM_read(LSTM, Xt, -t - 1)), 
          LSTM_read(LSTM, DZt, -1))));

    // Input gate input weight updates:
    push(LSTM, DWi, 
      sum(2, 
        pop(LSTM, DWi), 
        dot_product(
          transpose(LSTM_read(LSTM, Xt, -t - 1)), 
          LSTM_read(LSTM, DIt, -1))));

    // Forget gate input weight updates:
    push(LSTM, DWf, 
      sum(2, 
        pop(LSTM, DWf), 
        dot_product(
          transpose(LSTM_read(LSTM, Xt, -t - 1)), 
          LSTM_read(LSTM, DFt, -1))));

    // Output gate input weight updates:
    push(LSTM, DWo, 
      sum(2, 
        pop(LSTM, DWo), 
        dot_product(
          transpose(LSTM_read(LSTM, Xt, -t - 1)), 
          LSTM_read(LSTM, DOt, -1))));

    // Block input recurrent weight updates:
    push(LSTM, DRz, 
      sum(2, 
        pop(LSTM, DRz), 
        dot_product(
          transpose(LSTM_read(LSTM, Ht, -1)), 
          DZt_plus_1)));

    // Input gate recurrent weight updates:
    push(LSTM, DRi, 
      sum(2, 
        pop(LSTM, DRi), 
        dot_product(
          transpose(LSTM_read(LSTM, Ht, -1)), 
          matrix_copy(DIt_plus_1))));

    // Forget gate recurrent weight updates:
    push(LSTM, DRf, 
      sum(2, 
        pop(LSTM, DRf), 
        dot_product(
          transpose(LSTM_read(LSTM, Ht, -1)), 
          matrix_copy(DFt_plus_1))));

    // Block output recurrent weight updates:
    push(LSTM, DRo, 
      sum(2, 
        pop(LSTM, DRo), 
        dot_product(
          transpose(pop(LSTM, Ht)), 
          DOt_plus_1)));

    // Input gate peephole weight updates:
    push(LSTM, DPi, 
      sum(2, 
        pop(LSTM, DPi), 
        product(2, 
          LSTM_read(LSTM, Ct, -1), 
          DIt_plus_1)));

    // Forget gate peephole weight updates:
    push(LSTM, DPf, 
      sum(2, 
        pop(LSTM, DPf), 
        product(2, 
          LSTM_read(LSTM, Ct, -1), 
          DFt_plus_1)));

    // Output gate peephole weight updates:
    push(LSTM, DPo, 
      sum(2, 
        pop(LSTM, DPo), 
        product(2, 
          pop(LSTM, Ct), 
          LSTM_read(LSTM, DOt, -1))));

    // Prepare next iteration:
    if (t < time - 1) {
      DZt_plus_1 = pop(LSTM, DZt);
      DIt_plus_1 = pop(LSTM, DIt);
      DFt_plus_1 = pop(LSTM, DFt);
      DOt_plus_1 = pop(LSTM, DOt);
      DCt_plus_1 = pop(LSTM, DCt);
      Ft_plus_1  = pop(LSTM, Ft);
    }
  }
  destroy_matrix(pop(LSTM, Ft));
  destroy_matrix(pop(LSTM, DOt));
  destroy_matrix(pop(LSTM, DCt));
  destroy_matrix(pop(LSTM, DFt));
  destroy_matrix(pop(LSTM, DIt));
  destroy_matrix(pop(LSTM, DZt));
}
