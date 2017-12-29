(import (scheme base) (scheme inexact)
  (srfi 1) (srfi 27)
  (only (gauche base) tanh))

;; Poor man’s hash table
(define X-axis 0) (define Y-axis 1)
(define Xt_i 2) (define Yt_k 3)
(define gates (iota 10 4))
(define weights (iota 11 14))
(define input-weights (iota 4 14))
(define hidden-weights (iota 4 18))
(define cell-weights (iota 3 22))
(define errors (iota 6 25))
(define updates (iota 11 31))
(define input-updates (iota 4 31))
(define hidden-updates (iota 4 35))
(define cell-updates (iota 3 39))

(define-values
  (Bt_h St_c At_iota Bt_iota At_phi
   Bt_phi At_c Bt_c At_omega Bt_omega)
  (apply values gates))

(define-values
  (Wi_iota Wi_phi Wi_c Wi_omega
   Wh_iota Wh_phi Wh_c Wh_omega
   Wc_iota Wc_phi Wc_omega)
  (apply values weights))

(define-values (Dt_k Dt_omega Dt_s
                Dt_c Dt_phi Dt_iota)
  (apply values errors))

(define-values
  (Ui_iota Ui_phi Ui_c Ui_omega
   Uh_iota Uh_phi Uh_c Uh_omega
   Uc_iota Uc_phi Uc_omega)
  (apply values updates))

;; Initializations
(random-source-randomize!
  default-random-source)

(define (random-value x)
  (- (* (random-real) 2) 1))

(define (LSTM-make-tensor
          LSTM type a n f)
  (list-tabulate (length type)
    (lambda (x)
      (list-tabulate a
        (lambda (x)
          (list-tabulate
            (vector-ref LSTM X-axis)
            (lambda (x)
              (list-tabulate
                (* n
                   (vector-ref LSTM
                     Y-axis)) f))))))))

(define (LSTM-make-input-tensor
          LSTM type a f)
  (LSTM-make-tensor LSTM type a 1 f))

(define (LSTM-make-output-tensor
          LSTM type a x)
  (LSTM-make-tensor
    LSTM type a 2 (lambda (y) x)))

(define (LSTM-load! LSTM type xs)
  (for-each (lambda (i x)
              (vector-set! LSTM i x))
    type xs))

(define (LSTM-make-vectors-resetter f)
  (lambda (LSTM type . args)
    (LSTM-load! LSTM type
      (apply f LSTM type
        (append (drop-right args 1)
          (list (let ([x (last args)])
                  (lambda (y) x))))))))

(define (LSTM-make-matrices-resetter f)
  (lambda (LSTM type . args)
    (LSTM-load! LSTM type
      (map first
        (apply f LSTM type args)))))

(define (LSTM-reset-vectors!
          LSTM type x)
  (LSTM-load! LSTM type
    (LSTM-make-input-tensor
      LSTM type 2 (lambda (y) x))))

(define (LSTM-reset-output-vectors!
          LSTM type x)
  (LSTM-load! LSTM type
    (LSTM-make-output-tensor
      LSTM type 2 x)))

(define (LSTM-reset-matrices!
          LSTM type f)
  (LSTM-load! LSTM type
    (map first
      (LSTM-make-input-tensor
        LSTM type 1 f))))

(define (LSTM-reset-input! LSTM)
  (LSTM-reset-matrices! LSTM
    (list Xt_i) (lambda (x) 0)))

(define (LSTM-reset-gates! LSTM)
  (LSTM-reset-vectors! LSTM gates 1)
  (LSTM-reset-output-vectors! LSTM
    (list Bt_c) 1))

(define (LSTM-reset-inputs!
          LSTM type f)
  (LSTM-reset-matrices! LSTM type f))

(define (LSTM-reset-hiddens!
          LSTM type f)
  (LSTM-load! LSTM type
    (map first
      (LSTM-make-tensor LSTM type 1
        (vector-ref LSTM X-axis)
        f))))

(define LSTM-reset-cells!
  LSTM-reset-inputs!)

(define (LSTM-reset-weights! LSTM)
  (LSTM-reset-inputs!
    LSTM input-weights random-value)
  (LSTM-reset-hiddens!
    LSTM hidden-weights random-value)
  (LSTM-reset-cells!
    LSTM cell-weights random-value))

(define (LSTM-reset-updates! LSTM)
  (LSTM-reset-inputs!
    LSTM input-updates (lambda (x) 0))
  (LSTM-reset-hiddens!
    LSTM hidden-updates (lambda (x) 0))
  (LSTM-reset-cells!
    LSTM cell-updates (lambda (x) 0)))

(define (LSTM-reset-errors! LSTM)
  (LSTM-reset-vectors! LSTM errors 1))

(define (make-LSTM m n)
  (let ([LSTM (make-vector 42 #f)])
    (vector-set! LSTM X-axis m)
    (vector-set! LSTM Y-axis n)
    (LSTM-reset-input! LSTM)
    (LSTM-reset-gates! LSTM)
    (LSTM-reset-weights! LSTM)
    (LSTM-reset-errors! LSTM)
    (LSTM-reset-updates! LSTM)
    LSTM))

(define (make-tensor a b m n f)
  (list-tabulate a
    (lambda (x)
      (list-tabulate b
        (lambda (x)
          (list-tabulate m
            (lambda (x)
              (list-tabulate n
                f))))))))

(define (init-output z m n LSTM x)
  (first (make-tensor 1 z m (* n 2)
           (lambda (y) x))))

(define (push-input! LSTM input)
  (vector-set! LSTM Xt_i
    (append (map zip input)
      (list (vector-ref LSTM Xt_i)))))

(define (push-answer! m n LSTM answer)
  (vector-set! LSTM Yt_k
    (append
      (init-output
        (length
          (vector-ref LSTM Xt_i))
        m n LSTM 0)
        answer
        (init-output 1 m n LSTM 1))))

(define (push! k LSTM x)
  (vector-set! LSTM k
    (cons x (vector-ref LSTM k))))

(define (pop! k LSTM)
  (let* ([xyz (vector-ref LSTM k)]
         [x (first xyz)])
    (vector-set! LSTM k (cdr xyz)) x))

(define (transpose m) (apply zip m))

(define (matrix-map f m1 . ms)
  (if (null? ms)
      (map (lambda (x)
             (map (lambda (x) (f x))
               x))
        m1)
      (map (lambda (x)
             (apply map f
               (transpose x)))
        (transpose (cons m1 ms)))))

(define (matrix-multiply A B)
  (let ([B_T (transpose B)])
    (map (lambda (Aj)
           (map (lambda (Bi)
                  (apply +
                    (map * Aj Bi)))
             B_T))
      A)))

(define (matrix-sigmoid matrix)
  (matrix-map
    (lambda (x)
      (/ 1 (+ 1 (exp (- 0 x)))))
    matrix))

(define (matrix-tanh matrix)
  (matrix-map tanh matrix))

(define (sigmoid-derivative x)
  (matrix-map
    (lambda (x) (* x (- 1 x)))
    (matrix-sigmoid x)))

(define (tanh-derivative x)
  (matrix-map
    (lambda (x) (- 1 (* x x)))
    (matrix-tanh x)))

(define (forward-signal LSTM time x y)
  (matrix-multiply
    (time (vector-ref LSTM x))
    (vector-ref LSTM y)))

(define (join-signals! z LSTM . txys)
  (push! z LSTM
    (apply matrix-map +
      (map (lambda (txy)
             (apply forward-signal
               LSTM txy))
        txys))))

;; Feedforwarding
(define (LSTM-feedforward!
          m n LSTM input done?)

  ;; Input gates
  (join-signals! At_iota LSTM
    (list first Xt_i Wi_iota)
    (list second Bt_h Wh_iota)
    (list second St_c Wc_iota))

  (push! Bt_iota LSTM
    (matrix-sigmoid
      (first
        (vector-ref LSTM At_iota))))

  ;; Forget gates
  (join-signals! At_phi LSTM
    (list first Xt_i Wi_phi)
    (list second Bt_h Wh_phi)
    (list second St_c Wc_phi))

  (push! Bt_phi LSTM
    (matrix-sigmoid
      (first
        (vector-ref LSTM At_phi))))

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

  ;; Next input or done.
  (unless
    (done? (vector-ref LSTM Bt_c))
    (LSTM-feedforward! m n LSTM
      (let ([in
             (vector-ref LSTM Xt_i)])
        (if (null? (cdr in)) in
            (pop! Xt_i LSTM)))
      done?)))

;; BackPropagation Through Time.
(define (LSTM-backpropagate!
          LSTM input answer)

  ;; Outputs
  (push! Dt_k LSTM
    (let ([net-error
           (matrix-map -
             (first answer)
             (first (vector-ref
                      LSTM Bt_c)))])
      (matrix-map +
        (matrix-multiply net-error
          (transpose
            (vector-ref LSTM Wc_iota)))
        (matrix-multiply net-error
          (transpose
            (vector-ref LSTM Wc_phi)))
        (matrix-multiply net-error
          (transpose (vector-ref
                       LSTM Wc_omega)))
        (matrix-multiply
          (first
            (vector-ref LSTM Dt_omega))
          (transpose (vector-ref
                       LSTM Wh_omega)))
        (matrix-multiply
          (first
            (vector-ref LSTM Dt_c))
          (transpose
            (vector-ref LSTM Wh_c)))
        (matrix-multiply
          (first
            (vector-ref LSTM Dt_phi))
          (transpose
            (vector-ref LSTM Wh_phi)))
        (matrix-multiply
          (first
            (vector-ref LSTM Dt_iota))
          (transpose (vector-ref LSTM
                       Wh_iota))))))

  ;; Output gates
  (push! Dt_omega LSTM
    (matrix-map *
      (sigmoid-derivative
        (second
          (vector-ref LSTM At_omega)))
      (matrix-multiply
        (first (vector-ref LSTM Dt_k))
        (transpose
          (matrix-tanh
            (second (vector-ref LSTM
                      St_c)))))))

  ;; Cell state
  (push! Dt_s LSTM
    (matrix-map +
      (matrix-map *
        (second (vector-ref LSTM
                  Bt_omega))
        (sigmoid-derivative
          (second (vector-ref LSTM
                    St_c)))
        (first (vector-ref LSTM
                 Dt_k)))
      (matrix-map *
        (first (vector-ref LSTM
                 Bt_phi))
        (second (vector-ref LSTM
                  Dt_s)))
      (matrix-map *
        (vector-ref LSTM Wc_iota)
        (second (vector-ref LSTM
                  Dt_iota)))
      (matrix-map *
        (vector-ref LSTM Wc_phi)
        (second (vector-ref LSTM
                  Dt_phi)))
      (matrix-map *
        (vector-ref LSTM Wc_omega)
        (first (vector-ref LSTM
                 Dt_omega)))))

  ;; Cell outputs
  (push! Dt_c LSTM
    (matrix-map *
      (second (vector-ref LSTM
                Bt_iota))
      (tanh-derivative
        (second (vector-ref LSTM
                  At_c)))
      (first (vector-ref LSTM
               Dt_s))))

  ;; Forget gates
  (push! Dt_phi LSTM
    (matrix-map *
      (sigmoid-derivative
        (second (vector-ref LSTM
                  At_phi)))
      (matrix-multiply
        (first (vector-ref LSTM
                 Dt_s))
        (transpose
          (third (vector-ref LSTM
                   St_c))))))

  ;; Input gates
  (push! Dt_iota LSTM
    (matrix-map *
      (sigmoid-derivative
        (second (vector-ref LSTM
                  At_iota)))
      (matrix-multiply
        (first (vector-ref LSTM Dt_s))
        (transpose
          (matrix-tanh
            (second (vector-ref LSTM
                      At_c)))))))

  ;; Insert update step here! --->
  (until (null? answer)
    (LSTM-backpropagate! LSTM
      (cdr input) (cdr answer))))


;; Tests start here: --->
(define LSTM (make-LSTM 4 1))

(define input
  (list (list 0 0 1 1)
        (list 0 1 1 1)
        (list 1 0 1 1)
        (list 1 1 1 1)))

(define answer
  (list (list (list 0 1) (list 0 1)
              (list 1 0) (list 1 0))
        (list (list 0 1) (list 1 0)
              (list 1 0) (list 1 0))
        (list (list 1 0) (list 0 1)
              (list 1 0) (list 1 0))
        (list (list 1 0) (list 1 0)
              (list 1 0) (list 1 0))
        (list (list 1 1) (list 1 1)
              (list 1 1) (list 1 1))))

(define (feedforward-done? x)
  (any (lambda (x)
         (every (lambda (x) (= x 1))
           x))
    (first x)))

(push-input! LSTM input)

(push-answer! 4 1 LSTM answer)

;;(LSTM-feedforward! 4 1 LSTM input
;;  feedforward-done?)

;; Run once for tests:
(LSTM-feedforward! 4 1 LSTM input
  (lambda (x) #t))

(LSTM-backpropagate!
  LSTM input answer)
