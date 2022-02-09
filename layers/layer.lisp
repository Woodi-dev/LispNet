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

(defgeneric call (layer input &rest args))


(defun create-layer (layer model &rest args &key &allow-other-keys)
  (if (not (model-state-weights-initialized (model-state model)))
	(let ((layer (apply #'make-instance (cons layer args))))
		(push layer (model-layers model))
			layer)
	(let ((layer (nth (model-state-layer-pointer (model-state model)) (model-layers model))))
		  (decf (model-state-layer-pointer (model-state model)))
			layer)))
