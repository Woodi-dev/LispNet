(in-package :common-lisp-user)
(in-package #:lispnet)

(define-modify-macro minf (&rest numbers) min)
(define-modify-macro maxf (&rest numbers) max)

(defun lazy-multi-stack (axis arrays)
  (apply #'lazy-stack axis arrays))

(defun range-to-list (ra) (list (range-start ra) (range-end ra) (range-step ra)))

(defun stride-range (ra stride)
  (range (range-start ra)(range-end ra) (* (range-step ra) stride)))

(defun stride-shape (s strides)
  (assert (= (shape-rank s) (length strides)))
    (~l (loop for ra in (shape-ranges s)
				for stride in strides collect
          (stride-range ra stride) )))

(defun pad (array &key paddings (value 0))
  (let ((ranges (shape-ranges (lazy-array-shape array)))
        (new-ranges '()))
    (assert (= (length paddings) (length ranges)))
    (loop for ra in ranges
          for padding in paddings do
            (assert (= (length padding) 2))
            (assert (and  (= (range-step ra) 1) (= (range-start ra) 0)))
            (setq new-ranges (nconc new-ranges (list (range 0 (+ (+ (first padding) (second padding)) (range-end ra)))))))
    (let ((result (lazy-reshape value (~l new-ranges)))
          (offsets (loop for padding in paddings collect (first padding))))
      (lazy-collapse (lazy-overwrite result (lazy-reshape array (make-transformation :offsets offsets)))))))

(defun make-2d-kernel (kernel-size)
  (let ((offsets '())
		(size-y (first kernel-size))
		(size-x (second kernel-size)))
    (loop for i from (- 1  (ceiling size-y 2) ) to (floor (/ size-y 2)) do
      (loop for j from (- 1  (ceiling size-x 2)) to (floor (/ size-x 2)) do
        (push (list i j) offsets)))
    (reverse offsets)))

(defun argmax (li)
  (let ((max-index 0)
        (max-value (aref li 0)))
    (loop for val across li
          for index from 0 do
            (when (> val max-value)  (setf max-value val) (setf max-index index)))
    (values max-index max-value)))

(defun lazy-batch-argmax (array)
  (let* ((dim-indices (alexandria:iota (lazy-array-rank array)))
		(max-array (lazy-reshape array (make-transformation :output-mask (nconc (cdr dim-indices) (list(first dim-indices)))))))
		(loop for i from 1 below (lazy-array-rank array) do
			(setf max-array (lazy-reduce #'max max-array)))
		(lazy
			(lambda (x y)
				(if (>= x y) 1f0 0f0))
			 array
			 (lazy #'* (lazy-reshape max-array (transform a to a 0))(lazy-reshape 1.0 (lazy-array-shape array))))))
		
 
(defun lazy-allreduce-batchwise (array f)
  (let* ((dim-indices (alexandria:iota (lazy-array-rank array)))
		(result-array (lazy-reshape array (make-transformation :output-mask (nconc (cdr dim-indices) (list(first dim-indices)))))))
		(loop for i from 1 below (lazy-array-rank array) do
			(setf result-array (lazy-reduce f result-array)))
		result-array))

(defun binary-decision (array threshold)
  (lazy
   (lambda (x)
     (if (>= x (coerce threshold 'single-float))
         1f0
         0f0))
   array))

(defun print-list-horizontal (l)
   (format t "(")
   (loop for x in l 
		 for index from 0 do
		 (if (= index 0)
		 (format t "~S" x)
		 (format t " ~S" x)))
    (format t ")"))
	

		 
   


