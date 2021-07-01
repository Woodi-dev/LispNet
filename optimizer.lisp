
(in-package :common-lisp-user)

(defpackage #:lispnet.optimizer
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.trainable-parameter)
  (:export
   #:optimizer
   #:sgd
   #:update-weights
   #:make-sgd
   ))
   
(in-package #:lispnet.optimizer)



(defclass optimizer ()
  ((learning-rate
    :initarg :learning-rate
    :accessor learning-rate
    :initform 0.001)
   (network
    :initarg :network
    :accessor network
    :initform (error "Missing network argument"))
   (last-gradients
    :accessor last-gradients
    :initform '())
   ))


(defclass sgd (optimizer)
  ((momentum
   :initarg :momentum
   :accessor momentum
   :initform 0.0)
  ))
 
(defun make-sgd (&key (learning-rate 0.001) network (momentum 0.0))
(make-instance 'sgd :learning-rate learning-rate :network network :momentum momentum)
)

(defmethod initialize-instance :after ((opt optimizer) &rest initargs))

(defgeneric update-weights (optimizer &key weights gradient &allow-other-keys))

(defmethod update-weights ((opt sgd) &key weights gradient)
  (loop for i below (length weights) do
        (setf (trainable-parameter-value (nth i weights)) 
        	(lazy #'- (trainable-parameter-value (nth i weights))
              (lazy #'* (learning-rate opt)
                    (nth i gradient)))
        )
  )
 )

