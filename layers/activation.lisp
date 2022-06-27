(in-package #:lispnet)

(defclass activation-layer (layer)
	(activation :initarg :activation :initform (error "Missing layer argument")))
	  
(defmethod layer-compile ((layer activation-layer)))

(defmethod call ((layer activation-layer) input &rest args)
	(funcall (activation layer) input))

 
