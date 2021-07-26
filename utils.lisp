(in-package :common-lisp-user)

(defpackage #:lispnet.utils
  (:use
   #:common-lisp
   #:petalisp)
  (:export
   #:range-to-list
   #:stride-range
   #:stride-shape
   #:pad
   #:make-2d-kernel
   #:argmax
   #:lazy-multi-stack
   #:minf
   #:maxf
   #:binary-decision
   #:lazy-argmax))

(in-package #:lispnet.utils)

(define-modify-macro minf (&rest numbers) min)
(define-modify-macro maxf (&rest numbers) max)

(defun lazy-multi-stack (axis arrays)
  (apply #'lazy-stack axis arrays))

(defun range-to-list (ra) (list (range-start ra) (range-end ra) (range-step ra)))

(defun stride-range (ra stride)
  (range (range-start ra) (range-end ra) (* (range-step ra) stride)))

(defun stride-shape (s strides)
  (assert (= (shape-rank s) (length strides)))
  (~l (loop for ra in (shape-ranges s)
        for stride in strides collect
            (stride-range ra stride))))

(defun pad (array &key paddings (value 0.0))
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
  (let ((offsets '()))
    (loop for i from (- (floor (/ kernel-size 2))) to (floor (/ kernel-size 2)) do
      (loop for j from (- (floor (/ kernel-size 2))) to (floor (/ kernel-size 2)) do
        (push (list i j) offsets)))
    offsets))

(defun argmax (li)
  (let ((max-index 0)
        (max-value (aref li 0)))
    (loop for val across li
          for index from 0 do
            (when (> val max-value)  (setf max-value val) (setf max-index index)))
    (values max-index max-value)))

(defun lazy-argmax (array)
  (let* ((max-val (lazy-allreduce #'max array)))
    (lazy
     (lambda (x y)
       (if (>= x y) 1f0 0f0))
     array
     max-val)))

(defun binary-decision (array threshold)
  (lazy
   (lambda (x)
     (if (>= x threshold)
         (coerce 1.0 'single-float)
         (coerce 0.0 'single-float)))
   array))

