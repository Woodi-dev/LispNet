
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
(defmethod initialize-instance :after ((opt optimizer) &rest initargs) 
(let ((trainable-parameters
           (remove-if-not #'trainable-parameter-p (network-parameters (network opt)))))
(loop for i below (list-length trainable-parameters) do (push (lazy-reshape 0.0 (~)) (last-gradients opt)))
))

(defgeneric update-weights (optimizer &key weights gradient &allow-other-keys))


(defclass sgd (optimizer)
  ((momentum
   :initarg :momentum
   :accessor momentum
   :initform 0.0)
  ))
 
(defun make-sgd (&key (learning-rate 0.001) network (momentum 0.0))
(make-instance 'sgd :learning-rate learning-rate :network network :momentum momentum)
)

(defmethod update-weights ((opt sgd) &key weights gradient)
  (loop for i below (length weights) do
        (setf (trainable-parameter-value (nth i weights))
        	(lazy #'- (lazy #'- (trainable-parameter-value (nth i weights))
					         (lazy #'* (learning-rate opt)(nth i gradient)))
				      (lazy #'* (momentum opt) (nth i (last-gradients opt)))))    				  
		(setf (nth i (last-gradients opt)) (lazy #'* (learning-rate opt)(nth i gradient)))
  )

 )

