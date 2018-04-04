// feedforward.c
#include "feedforward.h"

void feedforward(LSTM_type *LSTM) {
  unsigned int time = LSTM->tensor[Xt].time;

  for (unsigned int t = 0; t < time; t++) {
    // Block input activations:
    push(LSTM, Zt, 
      matrix_tanh(
        sum(3, 
          dot_product(
            LSTM_read(LSTM, Xt, t), 
            LSTM_read(LSTM, Wz, -1)), 
          dot_product(
            LSTM_read(LSTM, Ht, -2), 
            LSTM_read(LSTM, Rz, -1)), 
          LSTM_read(LSTM, Bz, -1))));

    // Input gate activations:
    push(LSTM, It, 
      matrix_sigmoid(
        sum(4, 
          dot_product(
            LSTM_read(LSTM, Xt, t), 
            LSTM_read(LSTM, Wi, 0)), 
          dot_product(
            LSTM_read(LSTM, Ht, -2), 
            LSTM_read(LSTM, Ri, 0)), 
          product(2, 
            LSTM_read(LSTM, Pi, -1), 
            LSTM_read(LSTM, Ct, -2)), 
          LSTM_read(LSTM, Bi, -1))));

    // Forget gate activations:
    push(LSTM, Ft, 
      matrix_sigmoid(
        sum(4, 
          dot_product(
            LSTM_read(LSTM, Xt, t), 
            LSTM_read(LSTM, Wf, 0)), 
          dot_product(
            LSTM_read(LSTM, Ht, -2), 
            LSTM_read(LSTM, Rf, 0)), 
          product(2, 
            LSTM_read(LSTM, Pf, -1), 
            LSTM_read(LSTM, Ct, -2)), 
          LSTM_read(LSTM, Bf, -1))));

    // Cell memory:
    push(LSTM, Ct, 
      sum(2, 
        product(2, 
          LSTM_read(LSTM, Zt, -1), 
          LSTM_read(LSTM, It, -1)), 
        product(2, 
          LSTM_read(LSTM, Ct, -2), 
          LSTM_read(LSTM, Ft, -1))));

    // Output gate activations:
    push(LSTM, Ot, 
      matrix_sigmoid(
        sum(4, 
          dot_product(
            LSTM_read(LSTM, Xt, t), 
            LSTM_read(LSTM, Wo, 0)), 
          dot_product(
            LSTM_read(LSTM, Ht, -2), 
            LSTM_read(LSTM, Ro, 0)), 
          product(2, 
            LSTM_read(LSTM, Po, -1), 
            LSTM_read(LSTM, Ct, -1)), 
          LSTM_read(LSTM, Bo, -1))));

    // Block output activations:
    push(LSTM, Ht, 
      product(2, 
        matrix_tanh(LSTM_read(LSTM, Ct, -1)), 
        LSTM_read(LSTM, Ot, -1)));

    // Precalculate errors:
    push(LSTM, DHt, 
      subtract(2, 
        LSTM_read(LSTM, Yt, t), 
        LSTM_read(LSTM, Ht, t + 2)));
  }
  copy_tensor(LSTM, Ht, Ht_backup);
}
