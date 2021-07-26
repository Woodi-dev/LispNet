(in-package #:lispnet)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Trainable Parameters
(defclass trainable-parameter ()
  ((weights
    :accessor weights)
   (weights-value
    :accessor weights-value)
   (shape
    :accessor weights-shape
    :initarg :shape)))


(defmethod initialize-instance :after ((parameter trainable-parameter) &rest initargs)
  (setf (weights parameter)(make-unknown :shape (weights-shape parameter) :element-type 'single-float)))

(defun make-trainable-parameter (&key shape)
    (make-instance 'trainable-parameter
                   :shape shape))

(declaim (inline trainable-parameter-p))
(defun trainable-parameter-p (object)
  (typep object 'trainable-parameter))
