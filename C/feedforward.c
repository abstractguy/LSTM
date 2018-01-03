// feedforward.c
#include "feedforward.h"

// Will crash at runtime but was written
// this way to avoid bugs after trimming.
// Check for NOT deleting every temp!
void feedforward_once(LSTM_type *LSTM) {
  tensor_3D *temp1 = NULL, *temp2 = NULL;

  // Input preactivations:
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 3);
  matrix_dot_product(LSTM->LSTM[Xt_i], LSTM->LSTM[Xt_i]->z - 1, LSTM->LSTM[Wi_iota], 0, temp1, 0);
  matrix_dot_product(LSTM->LSTM[Bt_h], LSTM->LSTM[Bt_h]->z - 2, LSTM->LSTM[Wh_iota], 0, temp1, 1);
  matrix_dot_product(LSTM->LSTM[St_c], LSTM->LSTM[St_c]->z - 2, LSTM->LSTM[Wc_iota], 0, temp1, 2);
  temp1 = sum_time_steps(temp1);
  LSTM->LSTM[At_iota] = append_time_step(temp1, 0, LSTM->LSTM[At_iota]);
  temp1 = destroy_tensor_3D(temp1);

  // Input activations:
  temp1 = matrix_sigmoid(LSTM->LSTM[At_iota], LSTM->LSTM[At_iota]->z - 1);
  LSTM->LSTM[Bt_iota] = append_time_step(temp1, 0, LSTM->LSTM[Bt_iota]);
  temp1 = destroy_tensor_3D(temp1);

  // Forget preactivations:
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 3);
  matrix_dot_product(LSTM->LSTM[Xt_i], LSTM->LSTM[Xt_i]->z - 1, LSTM->LSTM[Wi_phi], 0, temp1, 0);
  matrix_dot_product(LSTM->LSTM[Bt_h], LSTM->LSTM[Bt_h]->z - 2, LSTM->LSTM[Wh_phi], 0, temp1, 1);
  matrix_dot_product(LSTM->LSTM[St_c], LSTM->LSTM[St_c]->z - 2, LSTM->LSTM[Wc_phi], 0, temp1, 2);
  temp1 = sum_time_steps(temp1);
  LSTM->LSTM[At_phi] = append_time_step(temp1, 0, LSTM->LSTM[At_phi]);
  temp1 = destroy_tensor_3D(temp1);

  // Forget activations:
  temp1 = matrix_sigmoid(LSTM->LSTM[At_phi], LSTM->LSTM[At_phi]->z - 1);
  LSTM->LSTM[Bt_phi] = append_time_step(temp1, 0, LSTM->LSTM[Bt_phi]);
  temp1 = destroy_tensor_3D(temp1);

  // Cell preactivations:
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 2);
  matrix_dot_product(LSTM->LSTM[Xt_i], LSTM->LSTM[Xt_i]->z - 1, LSTM->LSTM[Wi_c], 0, temp1, 0);
  matrix_dot_product(LSTM->LSTM[Bt_h], LSTM->LSTM[Bt_h]->z - 2, LSTM->LSTM[Wh_c], 0, temp1, 1);
  temp1 = sum_time_steps(temp1);
  LSTM->LSTM[At_c] = append_time_step(temp1, 0, LSTM->LSTM[At_c]);
  temp1 = destroy_tensor_3D(temp1);


  // Cell activations:

  // temp1 =
  //   element_wise_multiply(
  //     tanh(first(LSTM, At_c)),
  //     first(LSTM, Bt_iota));
  temp2 = matrix_tanh(LSTM->LSTM[At_c], LSTM->LSTM[At_c]->z - 1);
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 2);
  copy_time_steps(1, temp2, 0, temp1, 0);
  copy_time_steps(1, LSTM->LSTM[Bt_iota], LSTM->LSTM[Bt_iota]->z - 1, temp1, 1);
  temp1 = multiply_time_steps(temp1);
  copy_time_steps(1, temp1, 0, temp2, 1);
  temp1 = destroy_tensor_3D(temp1);

  // temp2 =
  //   element_wise_multiply(
  //     second(LSTM, St_c),
  //     first(LSTM, Bt_phi));
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 2);
  copy_time_steps(1, LSTM->LSTM[St_c], LSTM->LSTM[St_c]->z - 2, temp1, 0);
  copy_time_steps(1, LSTM->LSTM[Bt_phi], LSTM->LSTM[Bt_phi]->z - 1, temp1, 1);
  temp1 = multiply_time_steps(temp1);
  copy_time_steps(1, temp1, 0, temp2, 0);
  temp1 = destroy_tensor_3D(temp1);

  // push(LSTM, St_c, sum(temp1, temp2));
  temp2 = sum_time_steps(temp2);
  LSTM->LSTM[St_c] = append_time_step(temp2, 0, LSTM->LSTM[St_c]);
  temp2 = destroy_tensor_3D(temp2);


  // Output preactivations:
    temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 3);
  matrix_dot_product(LSTM->LSTM[Xt_i], LSTM->LSTM[Xt_i]->z - 1, LSTM->LSTM[Wi_omega], 0, temp1, 0);
  matrix_dot_product(LSTM->LSTM[Bt_h], LSTM->LSTM[Bt_h]->z - 2, LSTM->LSTM[Wh_omega], 0, temp1, 1);
  matrix_dot_product(LSTM->LSTM[St_c], LSTM->LSTM[St_c]->z - 1, LSTM->LSTM[Wc_omega], 0, temp1, 2);
  temp1 = sum_time_steps(temp1);
  LSTM->LSTM[At_omega] = append_time_step(temp1, 0, LSTM->LSTM[At_omega]);
  temp1 = destroy_tensor_3D(temp1);

  // Output activations:
  temp1 = matrix_sigmoid(LSTM->LSTM[At_iota], LSTM->LSTM[At_omega]->z - 1);
  LSTM->LSTM[Bt_omega] = append_time_step(temp1, 0, LSTM->LSTM[Bt_omega]);
  temp1 = destroy_tensor_3D(temp1);

  // Cell outputs:
  temp1 = make_tensor_3D(zero, LSTM->x, LSTM->y, 2);
  temp2 = make_tensor_3D(zero, LSTM->x, LSTM->y, 1);
  copy_time_steps(1, LSTM->LSTM[St_c], LSTM->LSTM[St_c]->z - 1, temp2, 0);
  temp2 = matrix_sigmoid(temp2, 0);
  copy_time_steps(1, temp2, 0, temp1, 0);
  copy_time_steps(1, LSTM->LSTM[Bt_omega], LSTM->LSTM[Bt_omega]->z - 1, temp1, 1);
  temp1 = multiply_time_steps(temp1);
  LSTM->LSTM[Bt_c] = append_time_step(temp1, 0, LSTM->LSTM[Bt_c]);
  temp1 = destroy_tensor_3D(temp1);
  temp2 = destroy_tensor_3D(temp2);
}
