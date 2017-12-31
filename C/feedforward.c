// feedforward.c
#include "feedforward.h"

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
  
}

/*
  ;; Cells
  (join-signals! At_c LSTM
    (list first Xt_i Wi_c)
    (list second Bt_h Wh_c))

  (push! St_c LSTM
    (matrix-map +
      (matrix-map *
        (second (vector-ref LSTM St_c))
        (first
          (vector-ref LSTM Bt_phi)))
      (matrix-map *
        (matrix-tanh
          (first
            (vector-ref LSTM At_c)))
        (first
          (vector-ref LSTM Bt_iota)))))

  ;; Output gates
  (join-signals! At_omega LSTM
    (list first Xt_i Wi_omega)
    (list second Bt_h Wh_omega)
    (list first St_c Wc_omega))

  (push! Bt_omega LSTM
    (matrix-sigmoid
      (first
        (vector-ref LSTM At_omega))))

  ;; Cell outputs
  (push! Bt_c LSTM
    (matrix-map *
      (matrix-sigmoid
        (first (vector-ref LSTM St_c)))
      (first
        (vector-ref LSTM Bt_omega))))
*/
