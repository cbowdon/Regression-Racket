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
				 (matrix-transpose 
				   (matrix* 
					 (matrix-transpose features) 
					 (matrix- targets hyp)))
				 (/ alpha (matrix-num-rows targets))))))

  (: iter (Integer (Matrix Float) -> (Matrix Float)))
  (define (iter steps theta)
	(if (eq? 0 steps) 
	  theta
	  (iter (sub1 steps) (batch-update theta))))

  (iter step-limit params))

; Stochastic Gradient Descent
; TODO does not work
(: stochastic-descent ((Float -> Float)
					   (Matrix Float) 
					   (Matrix Float) 
					   (Matrix Float) -> (Matrix Float)))
(define (stochastic-descent hypothesis features targets params)

  (: stochastic-update-param ((Matrix Float) Float (Matrix Float) Float Integer -> Float))
  (define (stochastic-update-param x_i y_i params f_ij m)
	(let* ([prod (matrix-dot x_i params)]
		   [diff (- y_i (linear prod))])
	  (/ (* f_ij alpha diff) m)))

  (: stochastic-update ((Matrix Float) Float (Matrix Float) Integer -> (Matrix Float)))
  (define (stochastic-update x_i y_i params m)
	(matrix-map 
	  (lambda (x_ij) (stochastic-update-param x_i y_i params x_ij m))
	  x_i))

  (: iter-row (Integer (Matrix Float) -> (Matrix Float)))
  (define (iter-row i params)
	(if (eq? (matrix-num-rows targets) i)
	  params
	  (iter-row
		(add1 i)
		(stochastic-update 
		  (matrix-row features i) 
		  (matrix-ref targets i 0)
		  params 
		  (matrix-num-rows targets)))))

  (: iter (Integer (Matrix Float) -> (Matrix Float)))
  (define (iter steps theta)
	(if (eq? 0 steps) 
	  theta
	  (iter (sub1 steps) (iter-row 0 theta))))

  (iter step-limit params))


; Test data
(: features (Matrix Float))
(define features (insert-bias-col (matrix-transpose (matrix [[1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]]))))

(: targets (Matrix Float))
(define targets (matrix-transpose (matrix [[1. 2. 3. 4. 5. 6. 7. 8. 9. 10.]])))

(: params (Matrix Float))
(define params (make-matrix 1 2 0.0))

; Tests
(batch-descent linear features targets params)
(stochastic-descent linear features targets params)
