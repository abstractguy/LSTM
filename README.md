# LSTM
Author: Samuel Duclos
Licence: 2-clause BSD
<<<<<<< HEAD
=======

>>>>>>> d7c1e62a7337ef8a14ed9571f1849d1d521081af
Status: Optimizing

How to build:
  # Clone repository and go to it:
  cd ~/Desktop && git clone https://github.com/abstractguy/LSTM/ && cd LSTM

  # For convolutional network test (NAND gate):
  cd CNN && gcc -Os -Wall -Wextra -o CNN -lm CNN.c && ./CNN | less

  # For LSTM tests:
  cd LSTM && gcc -Os -Wall -Wextra -o LSTM -lm matrix.c LSTM.c printing.c feedforward.c feedback.c update.c main.c && ./LSTM >output.txt && less output.txt
