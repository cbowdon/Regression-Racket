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
;		 [diff (matrix- y (hypothesis prod))]
;		 [a/m (/ alpha (array-size y))])
;	(matrix+ theta (* a/m (matrix* x diff)))))

(: features (Matrix Real))
(define features (build-matrix 10 1 (lambda (m n) (cast m Real))))
(: targets (Matrix Real))
(define targets (build-matrix 10 1 (lambda (m n) (* 2 (cast m Real)))))

(: params (Matrix Real))
(define params (make-matrix 1 2 0.0))

(: insert-bias ((Matrix Real) -> (Matrix Real)))
(define (insert-bias mat)
  (let ([bias-col (make-matrix (matrix-num-rows mat) 1 1)])
	(matrix-augment (list bias-col mat))))

targets
features
params

; TODO make into function
(matrix-scale (matrix* (matrix-transpose (insert-bias features))
					   (matrix- targets 
								(matrix* (insert-bias features) 
										 (matrix-transpose params))))
			  (/ 0.02 (matrix-num-rows targets)))
