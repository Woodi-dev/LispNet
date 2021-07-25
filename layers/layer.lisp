(in-package #:lispnet)


(defclass layer ()
  ((weights
    :accessor layer-weights
    :initform '())
   (activation 
	:initarg :activation
	:accessor layer-activation
	:initform nil)))
  
(defgeneric layer-compile(layer))

(defgeneric call (layer input))





   
		   
