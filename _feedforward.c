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
          matrix_copy(input), 
          LSTM_read(LSTM, Wz, -1)), 
        dot_product(
          LSTM_read(LSTM, Yt, -2), 
          LSTM_read(LSTM, Rz, -1)), 
        LSTM_read(LSTM, Bz, -1)));

    // Block input activations:
    push(LSTM, Zt, matrix_tanh(LSTM_read(LSTM, _Zt, -1)));

    // Input gate preactivations:
    push(LSTM, _It, 
      sum(4, 
        dot_product(
          matrix_copy(input), 
          LSTM_read(LSTM, Wi, 0)), 
        dot_product(
          LSTM_read(LSTM, Yt, -2), 
          LSTM_read(LSTM, Ri, 0)), 
        product(2, 
          LSTM_read(LSTM, Pi, -1), 
          LSTM_read(LSTM, Ct, -2)), 
        LSTM_read(LSTM, Bi, 0)));

    // Input gate activations:
    push(LSTM, It, matrix_sigmoid(LSTM_read(LSTM, _It, -1)));

    // Forget gate preactivations:
    push(LSTM, _Ft, 
      sum(4, 
        dot_product(
          matrix_copy(input), 
          LSTM_read(LSTM, Wf, 0)), 
        dot_product(
          LSTM_read(LSTM, Yt, -2), 
          LSTM_read(LSTM, Rf, 0)), 
        product(2, 
          LSTM_read(LSTM, Pf, -1), 
          LSTM_read(LSTM, Ct, -2)), 
        LSTM_read(LSTM, Bf, 0)));

    // Forget gate activations:
    push(LSTM, Ft, matrix_sigmoid(LSTM_read(LSTM, _Ft, -1)));

    // Cell memory:
    push(LSTM, Ct, 
      sum(2, 
        product(2, 
          LSTM_read(LSTM, Zt, -1), 
          LSTM_read(LSTM, It, -1)), 
        product(2, 
          LSTM_read(LSTM, Ct, -2), 
          LSTM_read(LSTM, Ft, -1))));

    // Output gate preactivations:
    push(LSTM, _Ot, 
      sum(4, 
        dot_product(
          input, 
          LSTM_read(LSTM, Wo, 0)), 
        dot_product(
          LSTM_read(LSTM, Yt, -2), 
          LSTM_read(LSTM, Ro, 0)), 
        product(2, 
          LSTM_read(LSTM, Po, -1), 
          LSTM_read(LSTM, Ct, -1)), 
        LSTM_read(LSTM, Bo, 0)));

    // Output gate activations:
    push(LSTM, Ot, matrix_sigmoid(LSTM_read(LSTM, _Ot, -1)));

    // Block output activations:
    push(LSTM, Yt, 
      product(2, 
        matrix_tanh(LSTM_read(LSTM, Ct, -1)), 
        LSTM_read(LSTM, Ot, -1)));
  } copy_tensor(LSTM, Yt, Yt_backup);
}
