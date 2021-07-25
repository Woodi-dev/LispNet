(in-package #:lispnet)

(defun sigmoid(input)
	(lazy #'/ 1.0 (lazy #'+ 1.0 (lazy #'exp (lazy #'* -1.0 input)))))
  
(defun softmax (input)
  (let* ((c (lazy-allreduce #'max input))
		(totals (lazy #'exp (lazy #'- input c))))		
    (lazy #'/ totals (lazy-allreduce #'+ totals))))

(defun relu (input)
  (lazy #'max 0.0 input))
  