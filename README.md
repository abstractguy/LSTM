# LSTM
Author: Samuel Duclos

Licence: 2-clause BSD

Status: In development

Development branch: LSTM/C/

Development status: Testing everything

To write: Integration of all components in a main loop.

How to build:

  git clone https://github.com/abstractguy/LSTM/
  
  cd LSTM/C/

  gcc matrix.c LSTM.c feedforward.c feedback.c update.c printing.c tests.c -lm -o LSTM

  ./LSTM
