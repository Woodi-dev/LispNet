(in-package #:lispnet)



(defun init-weights(&key shape (mode #'glorot-uniform) (fan-in 0) (fan-out 0) (element-type 'single-float))
  (funcall mode :shape shape :fan-in fan-in :fan-out fan-out :element-type element-type))


(defun uniform (&key shape (fan-in 0) (fan-out 0) (element-type 'single-float) )
  (let ((array (make-array (shape-dimensions shape) :element-type element-type)))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)
                (random (coerce 1 element-type))))
    array))
	
	
(defun glorot-uniform (&key shape fan-in fan-out (element-type 'single-float) )
  (let ((array (make-array (shape-dimensions shape) :element-type element-type))
        (limit (sqrt (/ 6.0 (+ fan-in fan-out)))))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)
                (coerce (-  limit (random (* 2 limit))) element-type)))
    array))

(defun zeros (&key shape (fan-in 0) (fan-out 0) (element-type 'single-float))
  (let ((array (make-array (shape-dimensions shape) :element-type element-type)))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)
                (coerce 0.0 element-type)))
    array))
	
(defun ones (&key shape (fan-in 0) (fan-out 0) (element-type 'single-float))
  (let ((array (make-array (shape-dimensions shape) :element-type element-type)))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)
                (coerce 1.0 element-type)))
    array))
