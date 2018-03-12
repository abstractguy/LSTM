// _feedforward.c
#include "_feedforward.h"

void feedforward(LSTM_type *LSTM) {
  matrix_type *input = NULL;
  copy_tensor(LSTM, Input, Xt);

  while (LSTM->tensor[Xt].time) {
    input = pop(LSTM, Xt);
    // Block input preactivations:
    push(LSTM, _Zt, 
      sum(3, 
        dot_product(
          first(LSTM, Wz), 
          matrix_copy(input)), 
        dot_product(
          first(LSTM, Rz), 
          second(LSTM, Yt)), 
        first(LSTM, Bz)));

    // Block input activations:
    push(LSTM, Zt, matrix_tanh(first(LSTM, _Zt)));

    // Input gate preactivations:
    push(LSTM, _It, 
      sum(4, 
        dot_product(
          first(LSTM, Wi), 
          matrix_copy(input)), 
        dot_product(
          first(LSTM, Ri), 
          second(LSTM, Yt)), 
        product(2, 
          first(LSTM, Pi), 
          second(LSTM, Ct)), 
        first(LSTM, Bi)));

    // Input gate activations:
    push(LSTM, It, matrix_sigmoid(first(LSTM, _It)));

    // Forget gate preactivations:
    push(LSTM, _Ft, 
      sum(4, 
        dot_product(
          first(LSTM, Wf), 
          matrix_copy(input)), 
        dot_product(
          first(LSTM, Rf), 
          second(LSTM, Yt)), 
        product(2, 
          first(LSTM, Pf), 
          second(LSTM, Ct)), 
        first(LSTM, Bf)));

    // Forget gate activations:
    push(LSTM, Ft, matrix_sigmoid(first(LSTM, _Ft)));

    // Cell memory:
    push(LSTM, Ct, 
      sum(2, 
        product(2, 
          first(LSTM, Zt), 
          first(LSTM, It)), 
        product(2, 
          second(LSTM, Ct), 
          first(LSTM, Ft))));

    // Output gate preactivations:
    push(LSTM, _Ot, 
      sum(4, 
        dot_product(
          first(LSTM, Wo), 
          input), 
        dot_product(
          first(LSTM, Ro), 
          second(LSTM, Yt)), 
        product(2, 
          first(LSTM, Po), 
          first(LSTM, Ct)), 
        first(LSTM, Bo)));

    // Output gate activations:
    push(LSTM, Ot, matrix_sigmoid(first(LSTM, _Ot)));

    // Block output activations:
    push(LSTM, Yt, 
      product(2, 
        matrix_tanh(first(LSTM, Ct)), 
        first(LSTM, Ot)));
  }
}
