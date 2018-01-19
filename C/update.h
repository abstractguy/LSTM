// update.h
#ifndef UPDATE_H
  #define UPDATE_H
  #include "LSTM.h"

  void sum_time_steps(LSTM_type *, index);
  void update_forward_once(LSTM_type *);
  void update_backward_once(LSTM_type *);

}
