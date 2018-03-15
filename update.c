// update.c
#include "update.h"

void update(LSTM_type *LSTM) {
  push(LSTM, Wz, 
    sum(2, 
      pop(LSTM, Wz), 
      LSTM_read(LSTM, DWz, -1)));

  matrix_for_each(zero, LSTM->tensor[DWz].matrix[0]);

  push(LSTM, Wi, 
    sum(2, 
      pop(LSTM, Wi), 
      LSTM_read(LSTM, DWi, -1)));

  matrix_for_each(zero, LSTM->tensor[DWi].matrix[0]);

  push(LSTM, Wf, 
    sum(2, 
      pop(LSTM, Wf), 
      LSTM_read(LSTM, DWf, -1)));

  matrix_for_each(zero, LSTM->tensor[DWf].matrix[0]);

  push(LSTM, Wo, 
    sum(2, 
      pop(LSTM, Wo), 
      LSTM_read(LSTM, DWo, -1)));

  matrix_for_each(zero, LSTM->tensor[DWo].matrix[0]);

  push(LSTM, Rz, 
    sum(2, 
      pop(LSTM, Rz), 
      LSTM_read(LSTM, DRz, -1)));

  matrix_for_each(zero, LSTM->tensor[DRz].matrix[0]);

  push(LSTM, Ri, 
    sum(2, 
      pop(LSTM, Ri), 
      LSTM_read(LSTM, DRi, -1)));

  matrix_for_each(zero, LSTM->tensor[DRi].matrix[0]);

  push(LSTM, Rf, 
    sum(2, 
      pop(LSTM, Rf), 
      LSTM_read(LSTM, DRf, -1)));

  matrix_for_each(zero, LSTM->tensor[DRf].matrix[0]);

  push(LSTM, Ro, 
    sum(2, 
      pop(LSTM, Ro), 
      LSTM_read(LSTM, DRo, -1)));

  matrix_for_each(zero, LSTM->tensor[DRo].matrix[0]);

  push(LSTM, Pi, 
    sum(2, 
      pop(LSTM, Pi), 
      LSTM_read(LSTM, DPi, -1)));

  matrix_for_each(zero, LSTM->tensor[DPi].matrix[0]);

  push(LSTM, Pf, 
    sum(2, 
      pop(LSTM, Pf), 
      LSTM_read(LSTM, DPf, -1)));

  matrix_for_each(zero, LSTM->tensor[DPf].matrix[0]);

  push(LSTM, Po, 
    sum(2, 
      pop(LSTM, Po), 
      LSTM_read(LSTM, DPo, -1)));

  matrix_for_each(zero, LSTM->tensor[DPo].matrix[0]);

  push(LSTM, Bz, 
    sum(2, 
      pop(LSTM, Bz), 
      LSTM_read(LSTM, DBz, -1)));

  matrix_for_each(zero, LSTM->tensor[DBz].matrix[0]);

  push(LSTM, Bi, 
    sum(2, 
      pop(LSTM, Bi), 
      LSTM_read(LSTM, DBi, -1)));

  matrix_for_each(zero, LSTM->tensor[DBi].matrix[0]);

  push(LSTM, Bf, 
    sum(2, 
      pop(LSTM, Bf), 
      LSTM_read(LSTM, DBf, -1)));

  matrix_for_each(zero, LSTM->tensor[DBf].matrix[0]);

  push(LSTM, Bo, 
    sum(2, 
      pop(LSTM, Bo), 
      LSTM_read(LSTM, DBo, -1)));

  matrix_for_each(zero, LSTM->tensor[DBo].matrix[0]);
}
