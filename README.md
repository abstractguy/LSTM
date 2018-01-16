# LSTM
Author: Samuel Duclos

Licence: 2-clause BSD

Status: In development

Development branch: LSTM/C/new/

Development status: Validating feedforwarding

To write: Feedback

How to build:

  git clone https://github.com/abstractguy/LSTM/
  
  cd LSTM/C/new/

  gcc matrix.c LSTM.c feedforward.c printing.c tests.c -lm -o LSTM
  
  ./LSTM
