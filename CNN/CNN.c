// Convolutional neural network emulating an NAND gate.
// Hacker: Samuel Duclos
// Github: https://github.com/abstractguy/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define EPOCH 60000

typedef struct {
  unsigned int rows, columns;
  long double **matrix;
} matrix_type;

matrix_type *make_matrix(unsigned int, unsigned int);
void destroy_matrix(matrix_type *);
long double sigmoid(long double);
void print_matrix(char *, matrix_type *);

int main(void) {
  matrix_type *in       = make_matrix(4, 4), 
              *out      = make_matrix(4, 1), 
              *synapses = make_matrix(4, 1), 
              *errors   = NULL, 
              *answer   = NULL,
              *temp1    = NULL,
              *temp2    = NULL;

  long double input[4][4]   = {{0.0, 0.0, 1.0, 1.0},
                               {0.0, 1.0, 1.0, 1.0},
                               {1.0, 0.0, 1.0, 1.0},
                               {1.0, 1.0, 1.0, 1.0}};

  long double output[4][1]  = {{1.0},
                               {1.0},
                               {1.0},
                               {0.0}};


  // Initialize synapses to random floats between -1 and 1.
  for (unsigned int row = 0; row < synapses->rows; row++)
    for (unsigned int column = 0; column < synapses->columns; column++)
      synapses->matrix[row][column] = (2.0 * rand() / RAND_MAX) - 1.0;


  // Convert input array to in matrix.
  for (unsigned int row = 0; row < in->rows; row++)
    for (unsigned int column = 0; column < in->columns; column++)
      in->matrix[row][column] = input[row][column];
  print_matrix("Input:", in);


  // Convert output array to out matrix.
  for (unsigned int row = 0; row < out->rows; row++)
    for (unsigned int column = 0; column < out->columns; column++)
      out->matrix[row][column] = output[row][column];
  print_matrix("Output:", out);





  // Train network EPOCH times.
  for (unsigned int epoch = 0; epoch < EPOCH; epoch++) {


    // V = RI
    answer = make_matrix(in->rows, synapses->columns);
    for (unsigned int row1 = 0; row1 < in->rows; row1++)
      for (unsigned int column2 = 0; column2 < synapses->columns; column2++) {
        answer->matrix[row1][column2] = 0.0;
        for (unsigned int column1 = 0; column1 < in->columns; column1++)
          answer->matrix[row1][column2] += 
            in->matrix[row1][column1] * synapses->matrix[column1][column2];
      }


    // Neuron transfer function.
    for (unsigned int row = 0; row < answer->rows; row++)
      for (unsigned int column = 0; column < answer->columns; column++)
        answer->matrix[row][column] = 
          sigmoid(answer->matrix[row][column]);

    if (epoch == EPOCH-1) print_matrix("Feedforward:", answer);


    // error = answer - output.
    temp1 = make_matrix(answer->rows, answer->columns);
    for (unsigned int row = 0; row < answer->rows; row++)
      for (unsigned int column = 0; column < answer->columns; column++)
        temp1->matrix[row][column] =
          out->matrix[row][column] - answer->matrix[row][column];


    // Output derivative.
    for (unsigned int row = 0; row < answer->rows; row++)
      for (unsigned int column = 0; column < answer->columns; column++)
        answer->matrix[row][column] = 
          (1.0 - sigmoid(answer->matrix[row][column]) *
                 sigmoid(answer->matrix[row][column]));


    // Error feedback.
    errors = make_matrix(temp1->rows, temp1->columns);
    for (unsigned int row = 0; row < temp1->rows; row++)
      for (unsigned int column = 0; column < temp1->columns; column++)
        errors->matrix[row][column] = 
          answer->matrix[row][column] * temp1->matrix[row][column];
    destroy_matrix(answer);
    destroy_matrix(temp1);

    if (epoch == EPOCH-1) print_matrix("Feedback:", errors);

    temp1 = make_matrix(in->columns, in->rows);


    // Rotate matrix 90°.
    for (unsigned int row = 0; row < in->columns; row++)
      for (unsigned int column = 0; column < in->rows; column++)
        temp1->matrix[row][column] = in->matrix[column][row];





    // V = RI
    temp2 = make_matrix(temp1->rows, errors->columns);
    for (unsigned int row1 = 0; row1 < temp1->rows; row1++)
      for (unsigned int column2 = 0; column2 < errors->columns; column2++) {
        temp2->matrix[row1][column2] = 0.0;
        for (unsigned int column1 = 0; column1 < temp1->columns; column1++)
          temp2->matrix[row1][column2] += 
            temp1->matrix[row1][column1] * errors->matrix[column1][column2];
      }
    destroy_matrix(temp1);
    destroy_matrix(errors);


    // Update synaptic weights.
    temp1 = make_matrix(temp2->rows, temp2->columns);
    for (unsigned int row = 0; row < temp2->rows; row++)
      for (unsigned int column = 0; column < temp2->columns; column++)
        temp1->matrix[row][column] =
          synapses->matrix[row][column] + temp2->matrix[row][0];
    destroy_matrix(synapses);
    destroy_matrix(temp2);
    synapses = temp1;

    if (epoch == EPOCH-1) print_matrix("New synapses:", synapses);

  } destroy_matrix(in); destroy_matrix(out); destroy_matrix(synapses);
    return 0;
}

matrix_type *make_matrix(unsigned int rows, unsigned int columns) {
  matrix_type *matrix   = malloc(sizeof(matrix_type));
  matrix->rows          = rows;
  matrix->columns       = columns;
  matrix->matrix        = malloc(sizeof(long double *) * rows);
  for (unsigned int row = 0; row < rows; row++)
    matrix->matrix[row] = malloc(sizeof(long double) * columns);
  return matrix;
}

void destroy_matrix(matrix_type *matrix) {
  for (unsigned int row = 0; row < matrix->rows; row++)
    free(matrix->matrix[row]);
  free(matrix->matrix);
  free(matrix);
}

long double sigmoid(long double x) {return 1.0 / (1.0 + expl(-x));}

void print_matrix(char *string, matrix_type *matrix) {
  puts(string);
  for (unsigned int row = 0; row < matrix->rows; row++) {
    for (unsigned int column = 0; column < matrix->columns; column++) {
      printf("%+5.4Lf ", matrix->matrix[row][column]);
    } putchar('\n');
  }
}
