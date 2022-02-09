(in-package #:lispnet)

(defclass dense-layer (layer)
	((out-features 
	  :initarg :out-features
	  :accessor out-features
	  :initform (error "Missing layer argument"))
	(trainable
	  :initarg :trainable
	  :accessor trainable
	  :initform t)))


		
(defmethod layer-compile ((layer dense-layer))
   (let* ((weights (first (layer-weights layer)))
          (bias (second (layer-weights layer)))		
		  (s (lazy-array-shape (weights weights)))
		  (fan-in (first(shape-dimensions s)))
		  (fan-out (second(shape-dimensions s))))
		  (setf (weights-value weights)
					(init-weights :shape s :mode #'glorot-uniform :fan-in fan-in :fan-out fan-out))
		  (setf (weights-value bias)
					(init-weights :shape (lazy-array-shape (weights bias)) :mode #'zeros))))
									
(defmethod call ((layer dense-layer) input &rest args)
  (when (null (layer-weights layer))
		(let ((weights (make-trainable-parameter :shape (~ (second (shape-dimensions (lazy-array-shape input))) ~ (out-features layer))
												 :trainable (trainable layer)))
             (bias  (make-trainable-parameter :shape (~ (out-features layer)) :trainable (trainable layer))))
		(setf (layer-weights layer) (list weights bias))))

  (let* ((weights (weights (first (layer-weights layer))))
         (bias  (weights (second (layer-weights layer))))
		 (result (lazy #'+ bias (matmul input weights))))
		 (if (layer-activation layer) (funcall (layer-activation layer) result) result)))