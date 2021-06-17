
(in-package :common-lisp-user)

(defpackage #:lispnet.initializers
  (:use
   #:common-lisp
   #:petalisp)
  (:export
   #:make-random-array
   
   ))

(in-package #:lispnet.initializers)

(defun make-random-array (dimensions &key (element-type 't))
  (let ((array (make-array dimensions :element-type element-type)))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)
                (1- (random (coerce 2 (array-element-type array))))))
    array))
