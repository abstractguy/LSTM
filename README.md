# LSTM
Author: Samuel Duclos

Licence: 2-clause BSD

Status: Debugging

How to build:

  git clone https://github.com/abstractguy/LSTM/
  
  cd LSTM

  # For convolutional network test (NAND gate):
  gcc -Os -Wall -Wextra -o CNN -lm matrix.c printing.c CNN.c

  ./CNN

  # For LSTM tests:
  gcc -Os -Wall -Wextra -o LSTM -lm matrix.c LSTM.c printing.c feedforward.c feedback.c update.c tests.c

  ./LSTM
