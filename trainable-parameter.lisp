(in-package #:lispnet)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Trainable Parameters
(defclass trainable-parameter ()
  ((weights
    :accessor weights)
   (weights-value
    :accessor weights-value
    :initform nil)
   (shape
    :accessor weights-shape
    :initarg :shape)
   (trainable 
    :accessor trainable
    :initarg :trainable
    :initform t)))


(defmethod initialize-instance :after ((parameter trainable-parameter) &rest initargs)
  (setf (weights parameter)(make-unknown :shape (weights-shape parameter) :element-type *network-precision*)))

(defun make-trainable-parameter (model &key shape (trainable t))
  (if (not (compiled (model-backend model)))
      (let ((parameter (make-instance 'trainable-parameter
                                      :shape shape
				      :trainable trainable)))
	(push parameter (parameters (model-backend model)))
	parameter)
      (let ((parameter (nth (parameter-pointer (model-backend model)) (model-weights model))))
	(decf (parameter-pointer (model-backend model)))
	parameter)))
			

(declaim (inline trainable-parameter-p))
(defun trainable-parameter-p (object)
  (typep object 'trainable-parameter))
