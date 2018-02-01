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

(define (LSTM-make-output-tensor
          LSTM type a x)
  (LSTM-make-tensor
    LSTM type a 2 (lambda (y) x)))

(define (LSTM-init-output LSTM n x)
  (first (LSTM-make-tensor
           LSTM (list 1) n 2
           (lambda (y) x))))

(define (push-answer! LSTM answer)
  (vector-set! LSTM Yt_k
    (append
      (LSTM-init-output LSTM
        (length (vector-ref LSTM Xt_i))
        0)
      answer
      (LSTM-init-output LSTM 1 1))))

(define (push-input! LSTM input)
  (vector-set! LSTM Xt_i
    (append (map zip input)
      (list (vector-ref LSTM Xt_i)))))

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
  (push! At_iota LSTM
    (matrix-map +
      (matrix-multiply
        (first (vector-ref LSTM Xt_i))
        (vector-ref LSTM Wi_iota))
      (matrix-multiply
        (second (vector-ref LSTM Bt_h))
        (vector-ref LSTM Wh_iota))
      (matrix-multiply
        (second (vector-ref LSTM St_c))
        (vector-ref LSTM Wc_iota))))

  (push! Bt_iota LSTM
    (matrix-sigmoid
      (first
        (vector-ref LSTM At_iota))))

  ;; Forget gates
  (push! At_phi LSTM
    (matrix-map +
      (matrix-multiply
        (first (vector-ref LSTM Xt_i))
        (vector-ref LSTM Wi_phi))
      (matrix-multiply
        (second (vector-ref LSTM Bt_h))
        (vector-ref LSTM Wh_phi))
      (matrix-multiply
        (second (vector-ref LSTM St_c))
        (vector-ref LSTM Wc_phi))))

  (push! Bt_phi LSTM
    (matrix-sigmoid
      (first
        (vector-ref LSTM At_phi))))

  ;; Cells
  (push! At_c LSTM
    (matrix-map +
      (matrix-multiply
        (first (vector-ref LSTM Xt_i))
        (vector-ref LSTM Wi_c))
      (matrix-multiply
        (second (vector-ref LSTM Bt_h))
        (vector-ref LSTM Wh_c))))

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
  (push! At_omega LSTM
    (matrix-map +
      (matrix-multiply
        (first (vector-ref LSTM Xt_i))
        (vector-ref LSTM Wi_omega))
      (matrix-multiply
        (second (vector-ref LSTM Bt_h))
        (vector-ref LSTM Wh_omega))
      (matrix-multiply
        (first (vector-ref LSTM St_c))
        (vector-ref LSTM Wc_omega))))

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

(push-answer! LSTM answer)

;;(LSTM-feedforward! 4 1 LSTM input
;;  feedforward-done?)

;; Run once for tests:
(LSTM-feedforward! 4 1 LSTM input
  (lambda (x) #t))

;;(LSTM-backpropagate!
  ;;LSTM input answer)










#| Result:
#(4 1 (((0) (0) (1) (1)) ((0) (1) (1) (1)) ((1) (0) (1) (1)) ((1) (1) (1) (1)) ((0) (0) (0) (0))) (((0 0) (0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0) (0 0)) ((0 0) (0 0) (0 0) (0 0)) ((0 1) (0 1) (1 0) (1 0)) ((0 1) (1 0) (1 0) (1 0)) ((1 0) (0 1) (1 0) (1 0)) ((1 0) (1 0) (1 0) (1 0)) ((1 1) (1 1) (1 1) (1 1)) ((1 1) (1 1) (1 1) (1 1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1.5 0.22327845395528212) (1.5 0.22327845395528212) (1.5552006767616926 0.3561872199322366) (1.5552006767616926 0.3561872199322366)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0 -0.21459208668747043 0.3506126689552902) (0 -0.21459208668747043 0.3506126689552902) (0.9296887704055541 -0.21459208668747043 0.3506126689552902) (0.9296887704055541 -0.21459208668747043 0.3506126689552902)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0.5 0.44655690791056424 0.5867661416544393) (0.5 0.44655690791056424 0.5867661416544393) (0.7170121394899336 0.44655690791056424 0.5867661416544393) (0.7170121394899336 0.44655690791056424 0.5867661416544393)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0 0.4649332644579447 0.15817029987316378) (0 0.4649332644579447 0.15817029987316378) (0.22170640670442276 0.4649332644579447 0.15817029987316378) (0.22170640670442276 0.4649332644579447 0.15817029987316378)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0.5 0.6141838291063382 0.5394603415289573) (0.5 0.6141838291063382 0.5394603415289573) (0.5552006767616925 0.6141838291063382 0.5394603415289573) (0.5552006767616925 0.6141838291063382 0.5394603415289573)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0 0.8840439670365456) (0 0.8840439670365456) (0.05086042621718323 0.8840439670365456) (0.05086042621718323 0.8840439670365456)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0.4542352749543329 0.1461362986614135) (0.4542352749543329 0.1461362986614135) (0.48558692894874966 0.1361635844882853) (0.48558692894874966 0.1361635844882853)) ((1 1) (1 1) (1 1) (1 1)) ((1 1) (1 1) (1 1) (1 1))) (((0 -0.8843720758308804 -1.1705020225882112) (0 -0.8843720758308804 -1.1705020225882112) (-0.13669757338495558 -0.8843720758308804 -1.2183535127273295) (-0.13669757338495558 -0.8843720758308804 -1.2183535127273295)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) (((0.5 0.292272597322827 0.23676425331836368) (0.5 0.292272597322827 0.23676425331836368) (0.4658787232724629 0.292272597322827 0.2282263315362761) (0.4658787232724629 0.292272597322827 0.2282263315362761)) ((1) (1) (1) (1)) ((1) (1) (1) (1))) ((0.9296887704055541) (-0.907598853853145) (-0.5887376135740943) (-0.14296912574229603)) ((0.22170640670442276) (0.8808847128077213) (-0.5218785939905477) (0.7888931195507576)) ((0.05086042621718323) (0.8207282141794867) (0.3025682795565583) (-0.3073405543052721)) ((-0.13669757338495558) (0.37377031065177246) (-0.8512793940251324) (0.11968547003061492)) ((-0.21459208668747043 -0.8965748511740419 0.7943636445297415 0.990007865873525) (-0.8821479285871052 0.6636688173825449 -0.5551353584527452 -0.4772087798196254) (-0.8565885144338832 -0.07519972572688127 -0.620256931230702 -0.5251397473386694) (-0.9466091394345872 -0.38248599870842526 -0.13871723652589152 -0.9713259293617815)) ((0.4649332644579447 0.5484737052362565 0.17050547749139477 0.23332774618763685) (-0.7568700774737114 -0.29752329737038985 0.5840257626209417 0.5347835911274317) (0.7112505023700333 0.6550148415631949 0.20767213247047955 0.1445678343988681) (-0.5740573345366762 -0.6638969855519823 -0.8433113832873311 0.2182524505864405)) ((0.8840439670365456 -0.3746827648497655 0.6603139423564706 -0.5148916590479058) (0.3716900690066678 -0.8407501965919535 -0.14163038469632672 -0.8055321386210785) (0.4021380757281259 0.1337917633036274 0.46966301427393753 -0.8968644666633103) (0.3155641311510331 -0.10028441117909015 -0.2124201236659029 0.8104811632789779)) ((-0.8843720758308804 -0.9285600296718104 0.008457940869378122 0.5440601625171215) (0.44164395463021666 0.669302536428078 -0.08041992878954884 0.7167518873592327) (0.9418167983002512 -0.6717900957815228 0.1264458774719086 -0.5810933706372128) (-0.29261897835037853 -0.07494641585636197 0.5326173960197067 -0.530869930990612)) ((0.3506126689552902) (-0.15176383043235653) (0.3397192209959339) (-0.38272852295542314)) ((0.15817029987316378) (0.9858871109913221) (0.1647888812850895) (0.6652248495020041)) ((-0.774632703447341) (-0.03830628197968711) (-0.9894060966772786) (0.5488986409886363)) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) (((1) (1) (1) (1)) ((1) (1) (1) (1))) ((0) (0) (0) (0)) ((0) (0) (0) (0)) ((0) (0) (0) (0)) ((0) (0) (0) (0)) ((0 0 0 0) (0 0 0 0) (0 0 0 0) (0 0 0 0)) ((0 0 0 0) (0 0 0 0) (0 0 0 0) (0 0 0 0)) ((0 0 0 0) (0 0 0 0) (0 0 0 0) (0 0 0 0)) ((0 0 0 0) (0 0 0 0) (0 0 0 0) (0 0 0 0)) ((0) (0) (0) (0)) ((0) (0) (0) (0)) ((0) (0) (0) (0)))
|#
