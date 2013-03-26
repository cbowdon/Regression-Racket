#lang typed/racket

(require math)

(: sigmoid (Real -> Real))
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(: linear (Real -> Real))
(define (linear x) x)

; x = features
; y = targets
; theta = params
;(: batch-update (Matrix Array Array (Real -> Real) Real -> Array))
;(define (batch-update x y theta hypothesis alpha)
;  (let* ([prod (matrix* x (matrix-transpose theta))]
;		 [diff (- y (hypothesis prod))]
;		 [a/m (/ alpha (array-size y))])
;	(matrix+ theta (* a/m (matrix* x diff)))))

(: features (Matrix Real))
(define features (build-matrix 10 (cast 1 Integer) (lambda (m n) (cast m Real))))
(: targets (Matrix Real))
(define targets (build-matrix 10 (cast 1 Integer) (lambda (m n) (* 2 (cast m Real)))))

(: params (Matrix Real))
(define params (make-matrix (cast 2 Integer) (cast 1 Integer) (cast 0 Real)))

;(batch-update features targets params linear 0.02)
;features
;targets
;params

(matrix* features (matrix-transpose params))
