// LSTM.c
#include "LSTM.h"

long double sigmoid(long double x) {
  return 1.0 / (1.0 + (long double)expl(-x));
}

tensor_3D *matrix_sigmoid(tensor_3D *tensor, int z) {
  return matrix_map(sigmoid, z, tensor);
}

tensor_3D *matrix_tanh(tensor_3D *tensor, int z) {
  return matrix_map(tanhl, z, tensor);
}

long double sigmoid_derivative(long double x) {
  long double y = sigmoid(x);
  return y * (1.0 - y);
}

long double tanh_derivative(long double x) {
  long double y = tanhl(x);
  return 1.0 - y * y;
}

long double random_long_double(void) {
  return 2.0 * (((long double)rand()) / ((long double)RAND_MAX)) - 1.0;
}

long double zero(void) {return 0.0;}

long double one(void) {return 1.0;}

tensor_3D *matrix_dot_product(tensor_3D *tensor1, int z1, tensor_3D *tensor2, int z2) {
  tensor_3D *tensor3 = make_tensor_3D(zero, tensor1->x, tensor2->y, 1);
  for (int y = 0; y < tensor1->x; y++) {
    for (int x = 0; x < tensor2->y; x++) {
      for (int k = 0; k < tensor1->y; k++) {
        tensor3->tensor[0][y][x] += tensor1->tensor[z1][y][k] * tensor2->tensor[z2][k][x];
      }
    }
  }
  return tensor3;
}

tensor_3D *matrix_map(long double (*f)(long double), int z, tensor_3D *tensor1) {
  tensor_3D *tensor2 = make_tensor_3D(zero, tensor1->x, tensor1->y, 1);
  for (int y = 0; y < tensor1->y; y++) {
    for (int x = 0; x < tensor1->x; x++) {
      tensor2->tensor[0][y][x] = f(tensor1->tensor[z][y][x]);
    }
  }
  return tensor2;
}

tensor_3D *make_tensor_3D(long double (*init)(void), int x, int y, int z) {
  tensor_3D *tensor = NULL;
  assert((tensor = calloc(1, sizeof(tensor_3D))));
  tensor->x = x;
  tensor->y = y;
  tensor->z = z;
  assert((tensor->tensor = calloc(z, sizeof(long double **))));
  for (int i = 0; i < z; i++) {
    assert((tensor->tensor[i] = calloc(y, sizeof(long double *))));
    for (int j = 0; j < y; j++) {
      assert((tensor->tensor[i][j] = calloc(x, sizeof(long double))));
      for (int k = 0; k < x; k++) {
        tensor->tensor[i][j][k] = init();
      }
    }
  } return tensor;
}

tensor_3D *destroy_tensor_3D(tensor_3D *tensor) {
  for (int i = 0; i < tensor->z; i++) {
    for (int j = 0; j < tensor->y; j++) {
      free(tensor->tensor[i][j]);
      tensor->tensor[i][j] = NULL;
    } free(tensor->tensor[i]);
      tensor->tensor[i] = NULL;
  }   free(tensor->tensor);
      tensor->tensor = NULL;
      free(tensor);
      return NULL;
}

void copy_time_steps(int n, tensor_3D *tensor1, int z1, tensor_3D *tensor2, int z2) {
  for (int z = 0; z < n; z++) {
    for (int y = 0; y < tensor2->y; y++) {
      for (int x = 0; x < tensor2->x; x++) {
        tensor2->tensor[z + z2][y][x] = tensor1->tensor[z + z1][y][x];
      }
    }
  }
}

tensor_3D *tensor_deep_copy(tensor_3D *tensor1) {
  tensor_3D *tensor2 = make_tensor_3D(zero, tensor1->x, tensor1->y, tensor1->z);
  copy_time_steps(tensor1->z, tensor1, 0, tensor2, 0);
  return tensor2;
}

tensor_3D *append_time_step(tensor_3D *tensor1, int z, tensor_3D *tensor2) {
  tensor_3D *tensor3 = make_tensor_3D(zero, tensor2->x, tensor2->y, tensor2->z + 1);
  copy_time_steps(tensor2->z, tensor2, 0, tensor3, 0);
  copy_time_steps(1, tensor1, z, tensor3, tensor2->z);
  tensor2 = destroy_tensor_3D(tensor2);
  return tensor3;
}

tensor_3D *drop_time_step(tensor_3D *tensor1) {
  tensor_3D *tensor2 = make_tensor_3D(zero, tensor1->x, tensor1->y, tensor1->z - 1);
  copy_time_steps(tensor2->z, tensor1, 0, tensor2, 0);
  tensor1 = destroy_tensor_3D(tensor1);
  return tensor2;
}

LSTM_type *make_LSTM(int x, int y) {
  LSTM_type *LSTM = NULL;
  assert((LSTM = calloc(1, sizeof(LSTM_type))));
  LSTM->x = x;
  LSTM->y = y;

  // Empty inputs (Xt_i):
  LSTM->LSTM[Xt_i] = make_tensor_3D(zero, x, y, 1);

  // Empty outputs (Yt_k):
  LSTM->LSTM[Yt_k] = make_tensor_3D(zero, x, y * 2, 1);

  for (index i = GATES_BEGIN; i < GATES_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(one, x, y, 2);
  }

  for (index i = INPUT_WEIGHTS_BEGIN; i < INPUT_WEIGHTS_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, y, 1);
  }

  for (index i = HIDDEN_WEIGHTS_BEGIN; i < HIDDEN_WEIGHTS_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, x, 1);
  }

  for (index i = CELL_WEIGHTS_BEGIN; i < CELL_WEIGHTS_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, y, 1);
  }

  for (index i = ERRORS_BEGIN; i < ERRORS_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(zero, x, y, 1);
  }

  for (index i = INPUT_UPDATES_BEGIN; i < INPUT_UPDATES_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, y, 1);
  }

  for (index i = HIDDEN_UPDATES_BEGIN; i < HIDDEN_UPDATES_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, x, 1);
  }

  for (index i = CELL_UPDATES_BEGIN; i < CELL_UPDATES_END; i++) {
    LSTM->LSTM[i] = make_tensor_3D(random_long_double, x, y, 1);
  }

  return LSTM;
}

LSTM_type *destroy_LSTM(LSTM_type *LSTM) {
  for (index i = 0; i < LSTM_SIZE; i++) {
    LSTM->LSTM[i] = destroy_tensor_3D(LSTM->LSTM[i]);
  } free(LSTM);
    return NULL;
}
