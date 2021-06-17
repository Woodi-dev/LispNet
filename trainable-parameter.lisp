
(in-package :common-lisp-user)

(defpackage #:lispnet.trainable-parameter
  (:use
   #:common-lisp
   #:petalisp)
  (:export
   #:trainable-parameter
   #:make-trainable-parameter
   #:trainable-parameter-p
   #:trainable-parameter-value
   ))
   
(in-package #:lispnet.trainable-parameter)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Trainable Parameters
(defclass trainable-parameter (parameter)
  ((%value :initarg :value :accessor trainable-parameter-value)))

(defun make-trainable-parameter (initial-value)
  (let ((value (lazy-array initial-value)))
    (make-instance 'trainable-parameter
                   :shape (array-shape initial-value)
                   :element-type (upgraded-array-element-type (element-type value))
                   :value value)))

(declaim (inline trainable-parameter-p))
(defun trainable-parameter-p (object)
  (typep object 'trainable-parameter))
