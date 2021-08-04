(in-package #:lispnet)

(defun sigmoid(input)
	(lazy #'/ 1.0 (lazy #'+ 1.0 (lazy #'exp (lazy #'* -1.0 input)))))
  ;;TODO
(defun softmax (input)
  (let* ((c (lazy-allreduce #'max input))
		(array (lazy #'exp (lazy #'- input c)))
		(sums (lazy-allreduce-batchwise array #'+)))			
        (lazy #'/ array (lazy #'* (lazy-reshape sums (transform a to a 0))
								  (lazy-reshape 1.0 (lazy-array-shape array))))))

(defun relu (input)
  (lazy #'max 0f0 input))
  