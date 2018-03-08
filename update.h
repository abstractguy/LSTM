// update.h
#ifndef UPDATE_H
  #define UPDATE_H
  #include "LSTM.h"

  void update_forward_once(LSTM_type *, unsigned int);
  void update_backward_once(LSTM_type *);
#endif
