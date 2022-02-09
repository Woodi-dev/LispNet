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
    :initarg :shape)
   (trainable 
    :accessor trainable
	:initarg :trainable
	:initform t)))


(defmethod initialize-instance :after ((parameter trainable-parameter) &rest initargs)
  (setf (weights parameter)(make-unknown :shape (weights-shape parameter) :element-type *network-precision*)))

(defun make-trainable-parameter (&key shape (trainable t))
    (make-instance 'trainable-parameter
                   :shape shape
				   :trainable trainable))

(declaim (inline trainable-parameter-p))
(defun trainable-parameter-p (object)
  (typep object 'trainable-parameter))
