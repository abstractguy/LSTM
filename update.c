// update.c
#include "update.h"

void update(LSTM_type *LSTM) {
  push(LSTM, Wz, 
    sum(2, 
      pop(LSTM, Wz), 
      first(LSTM, DWz)));

  matrix_for_each(zero, LSTM->tensor[DWz].matrix[0]);

  push(LSTM, Wi, 
    sum(2, 
      pop(LSTM, Wi), 
      first(LSTM, DWi)));

  matrix_for_each(zero, LSTM->tensor[DWi].matrix[0]);

  push(LSTM, Wf, 
    sum(2, 
      pop(LSTM, Wf), 
      first(LSTM, DWf)));

  matrix_for_each(zero, LSTM->tensor[DWf].matrix[0]);

  push(LSTM, Wo, 
    sum(2, 
      pop(LSTM, Wo), 
      first(LSTM, DWo)));

  matrix_for_each(zero, LSTM->tensor[DWo].matrix[0]);

  push(LSTM, Rz, 
    sum(2, 
      pop(LSTM, Rz), 
      first(LSTM, DRz)));

  matrix_for_each(zero, LSTM->tensor[DRz].matrix[0]);

  push(LSTM, Ri, 
    sum(2, 
      pop(LSTM, Ri), 
      first(LSTM, DRi)));

  matrix_for_each(zero, LSTM->tensor[DRi].matrix[0]);

  push(LSTM, Rf, 
    sum(2, 
      pop(LSTM, Rf), 
      first(LSTM, DRf)));

  matrix_for_each(zero, LSTM->tensor[DRf].matrix[0]);

  push(LSTM, Ro, 
    sum(2, 
      pop(LSTM, Ro), 
      first(LSTM, DRo)));

  matrix_for_each(zero, LSTM->tensor[DRo].matrix[0]);

  push(LSTM, Pi, 
    sum(2, 
      pop(LSTM, Pi), 
      first(LSTM, DPi)));

  matrix_for_each(zero, LSTM->tensor[DPi].matrix[0]);

  push(LSTM, Pf, 
    sum(2, 
      pop(LSTM, Pf), 
      first(LSTM, DPf)));

  matrix_for_each(zero, LSTM->tensor[DPf].matrix[0]);

  push(LSTM, Po, 
    sum(2, 
      pop(LSTM, Po), 
      first(LSTM, DPo)));

  matrix_for_each(zero, LSTM->tensor[DPo].matrix[0]);

  push(LSTM, Bz, 
    sum(2, 
      pop(LSTM, Bz), 
      first(LSTM, DBz)));

  matrix_for_each(zero, LSTM->tensor[DBz].matrix[0]);

  push(LSTM, Bi, 
    sum(2, 
      pop(LSTM, Bi), 
      first(LSTM, DBi)));

  matrix_for_each(zero, LSTM->tensor[DBi].matrix[0]);

  push(LSTM, Bf, 
    sum(2, 
      pop(LSTM, Bf), 
      first(LSTM, DBf)));

  matrix_for_each(zero, LSTM->tensor[DBf].matrix[0]);

  push(LSTM, Bo, 
    sum(2, 
      pop(LSTM, Bo), 
      first(LSTM, DBo)));

  matrix_for_each(zero, LSTM->tensor[DBo].matrix[0]);
}
