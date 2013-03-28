#lang typed/racket/base

(require math/array
		 math/matrix)

; Constants
(: alpha Float)
(define alpha 0.02)

(: step-limit Integer)
(define step-limit 100)

; Hypotheses
(: sigmoid (Float -> Float))
(define (sigmoid x)
  (/ 1.0 (+ 1.0 (exp (- x)))))

(: linear (Float -> Float))
(define (linear x) x)

; Common utilities
(: insert-bias-col ((Matrix Float) -> (Matrix Float)))
(define (insert-bias-col mat)
  (let ([bias-col (make-matrix (matrix-num-rows mat) 1 1.0)])
	(matrix-augment (list bias-col mat))))

; Stochastic Gradient Descent
(: stochastic-descent ((Float -> Float)
					   (Matrix Float) 
					   (Matrix Float) 
					   (Matrix Float) -> (Matrix Float)))
(define (stochastic-descent hypothesis features targets params)
  params)

(: stochastic-update ((Matrix Float) Float (Matrix Float) Float Integer -> (Matrix Float)))
(define (stochastic-update features target params f_ij m)
  (let* ([prod (matrix-dot features params)]
		 [diff (- target (linear prod))])
	(displayln (/ (* f_ij alpha diff) m))
	params))


; Batch Gradient Descent
(: batch-descent ((Float -> Float)
				  (Matrix Float) 
				  (Matrix Float) 
				  (Matrix Float) -> (Matrix Float)))
(define (batch-descent hypothesis features targets params)

  (: batch-update ((Matrix Float) -> (Matrix Float)))
  (define (batch-update params)
	(let* ([prod (matrix* features (matrix-transpose params))]
		   [hyp (matrix-map hypothesis prod)])
	  (matrix+ params
			   (matrix-scale 
				 (matrix-transpose (matrix* (matrix-transpose features) (matrix- targets hyp)))
				 (/ alpha (matrix-num-rows targets))))))

  (: iter ((Matrix Float) Integer -> (Matrix Float)))
  (define (iter theta steps)
	(if (eq? 0 steps) 
	  theta
	  (iter (batch-update theta) (sub1 steps))))

  (iter params step-limit))

; Testing
(: features (Matrix Float))
(define features (insert-bias-col (matrix-transpose (matrix [[1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]]))))

(: targets (Matrix Float))
(define targets (matrix-transpose (matrix [[1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]])))

(: params (Matrix Float))
(define params (make-matrix 1 2 0.0))

(batch-descent linear features targets params)
(stochastic-descent linear features targets params)

(stochastic-update (matrix-row features 0) (matrix-ref targets 0 0) params (matrix-ref features 0 0) (matrix-num-rows targets))
