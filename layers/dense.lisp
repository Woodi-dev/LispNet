(in-package #:lispnet)

(defclass dense-layer (layer)
	((in-features 
	  :initarg :in-features
	  :accessor in-features
	  :initform (error "Missing layer argument"))
	  (out-features 
	  :initarg :out-features
	  :accessor out-features
	  :initform (error "Missing layer argument"))))

(defmethod initialize-instance :after ((layer dense-layer) &rest initargs)
	(let ((weights  (make-trainable-parameter :shape (~ (in-features layer) ~ (out-features layer))))
		  (bias (make-trainable-parameter :shape (~ (out-features layer)))))
		  (setf (layer-weights layer) (list weights bias))
	)) 
	  
(defun make-dense-layer (model &key in-features out-features (activation nil))
  (let ((layer (make-instance 'dense-layer :in-features in-features :out-features out-features :activation activation)))
    (push layer (model-layers model))
	    layer))
		
(defmethod layer-compile ((layer dense-layer))
   (let* ((weights (first (layer-weights layer)))
          (bias (second (layer-weights layer)))		
		  (s (lazy-array-shape (weights weights)))
		  (fan-in (first(shape-dimensions s)))
		  (fan-out (second(shape-dimensions s))))
		  (setf (weights-value weights)
					(init-weights :shape s :mode #'ones :fan-in fan-in :fan-out fan-out))
		  (setf (weights-value bias)
					(init-weights :shape (lazy-array-shape (weights bias)) :mode #'zeros))))
									
(defmethod call ((layer dense-layer) input)
  (let* ((weights (weights (first (layer-weights layer))))
         (bias  (weights (second (layer-weights layer))))
		 (result (lazy #'+ bias (matmul input weights))))
		 (if (layer-activation layer) (funcall (layer-activation layer) result) result)))