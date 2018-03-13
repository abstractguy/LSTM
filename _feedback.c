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

  copy_tensor(LSTM, Input_reversed, Xt_reversed);
  copy_tensor(LSTM, Output, Answer);

  while (LSTM->tensor[Answer].time) {
    // FEEDBACK PART:
    // Block output errors:
    push(LSTM, DYt, 
      sum(5, 
        subtract(2, 
          pop(LSTM, Answer), 
          first(LSTM, Yt)), 
        dot_product(
          transpose(first(LSTM, Rz)),
          matrix_copy(DZt_plus_1)), 
        dot_product(
          transpose(first(LSTM, Ri)),
          matrix_copy(DIt_plus_1)), 
        dot_product(
          transpose(first(LSTM, Rf)),
          matrix_copy(DFt_plus_1)), 
        dot_product(
          transpose(first(LSTM, Ro)),
          matrix_copy(DOt_plus_1))));

    // Output gate errors:
    push(LSTM, DOt, 
      product(3, 
        first(LSTM, DYt), 
        matrix_tanh(first(LSTM, Ct)), 
        sigmoid_derivative(pop(LSTM, _Ot))));

    // Cell memory errors:
    push(LSTM, DCt, 
      sum(5, 
        product(3, 
          pop(LSTM, DYt), 
          pop(LSTM, Ot), 
          tanh_derivative(first(LSTM, Ct))), 
        product(2, 
          first(LSTM, Po), 
          first(LSTM, DOt)), 
        product(2, 
          first(LSTM, Pi), 
          matrix_copy(DIt_plus_1)), 
        product(2, 
          first(LSTM, Pf), 
          matrix_copy(DFt_plus_1)), 
        product(2, 
          DCt_plus_1, 
          Ft_plus_1)));

    // Forget gate errors:
    push(LSTM, DFt, 
      product(3, 
        first(LSTM, DCt), 
        second(LSTM, Ct), 
        sigmoid_derivative(pop(LSTM, _Ft))));

    // Input gate errors:
    push(LSTM, DIt, 
      product(3, 
        first(LSTM, DCt), 
        pop(LSTM, Zt), 
        sigmoid_derivative(pop(LSTM, _It))));

    // Block input errors:
    push(LSTM, DZt, 
      product(3, 
        first(LSTM, DCt), 
        pop(LSTM, It), 
        tanh_derivative(pop(LSTM, _Zt))));

    // UPDATE PART:
    // Block input input weight updates:
    push(LSTM, DWz, 
      sum(2, 
        pop(LSTM, DWz), 
        dot_product(
          transpose(first(LSTM, DZt)), 
          first(LSTM, Xt_reversed))));

    // Input gate input weight updates:
    push(LSTM, DWi, 
      sum(2, 
        pop(LSTM, DWi), 
        dot_product(
          transpose(first(LSTM, DIt)), 
          first(LSTM, Xt_reversed))));

    // Forget gate input weight updates:
    push(LSTM, DWf, 
      sum(2, 
        pop(LSTM, DWf), 
        dot_product(
          transpose(first(LSTM, DFt)), 
          first(LSTM, Xt_reversed))));

    // Output gate input weight updates:
    push(LSTM, DWo, 
      sum(2, 
        pop(LSTM, DWo), 
        dot_product(
          transpose(first(LSTM, DOt)), 
          pop(LSTM, Xt_reversed))));

    // Block input recurrent weight updates:
    push(LSTM, DRz, 
      sum(2, 
        pop(LSTM, DRz), 
        dot_product(
          transpose(DZt_plus_1), 
          first(LSTM, Yt))));

    // Input gate recurrent weight updates:
    push(LSTM, DRi, 
      sum(2, 
        pop(LSTM, DRi), 
        dot_product(
          transpose(matrix_copy(DIt_plus_1)), 
          first(LSTM, Yt))));

    // Forget gate recurrent weight updates:
    push(LSTM, DRf, 
      sum(2, 
        pop(LSTM, DRf), 
        dot_product(
          transpose(matrix_copy(DFt_plus_1)), 
          first(LSTM, Yt))));

    // Block output recurrent weight updates:
    push(LSTM, DRo, 
      sum(2, 
        pop(LSTM, DRo), 
        dot_product(
          transpose(DOt_plus_1), 
          pop(LSTM, Yt))));

    // Input gate peephole weight updates:
    push(LSTM, DPi, 
      sum(2, 
        pop(LSTM, DPi), 
        product(2, 
          first(LSTM, Ct), 
          DIt_plus_1)));

    // Forget gate peephole weight updates:
    push(LSTM, DPf, 
      sum(2, 
        pop(LSTM, DPf), 
        product(2, 
          first(LSTM, Ct), 
          DFt_plus_1)));

    // Output gate peephole weight updates:
    push(LSTM, DPo, 
      sum(2, 
        pop(LSTM, DPo), 
        product(2, 
          pop(LSTM, Ct), 
          first(LSTM, DOt))));

    // Block input bias weight updates:
    push(LSTM, DBz, 
      sum(2, 
        pop(LSTM, DBz), 
        first(LSTM, DZt)));

    // Input gate bias weight updates:
    push(LSTM, DBi, 
      sum(2, 
        pop(LSTM, DBi), 
        first(LSTM, DIt)));

    // Forget gate bias weight updates:
    push(LSTM, DBf, 
      sum(2, 
        pop(LSTM, DBf), 
        first(LSTM, DFt)));

    // Output gate bias weight updates:
    push(LSTM, DBo, 
      sum(2, 
        pop(LSTM, DBo), 
        first(LSTM, DOt)));

    // Prepare next iteration:
    if (LSTM->tensor[Answer].time) {
      DZt_plus_1 = pop(LSTM, DZt), 
      DIt_plus_1 = pop(LSTM, DIt), 
      DFt_plus_1 = pop(LSTM, DFt), 
      DOt_plus_1 = pop(LSTM, DOt), 
      DCt_plus_1 = pop(LSTM, DCt), 
      Ft_plus_1  = pop(LSTM, Ft);
    }
  }
}
