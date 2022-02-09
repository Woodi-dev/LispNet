(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass prolongation-layer (layer)
  ((level 
      :initarg :level
	  :accessor level
	  :initform (error "argument level required"))
  (restrict-layer
      :initarg :restrict-layer
	  :accessor restrict-layer
	  :initform (error "argument restrict-layer required"))
   (target-size :accessor target-size)
   (source-size :accessor source-size)))

		

(defmethod initialize-instance :after ((layer prolongation-layer) &rest initargs)
	(let* ((input-size (1+ (expt 2 (level layer))))
	       (target-size (1+ (expt 2 (1+ (level layer))))))
		   (setf (target-size layer) target-size)
		   (setf (source-size layer) input-size)
		   )) 
									   
		
(defmethod layer-compile ((layer prolongation-layer)))

		
		

(defmethod call ((layer prolongation-layer) input &rest args)
	(let* ((source-size (source-size layer))
		   (target-size (target-size layer))
		   ;; copy weights from restriction layer
		   (weights (lazy #'* 4.0 (lazy-reshape (weights (first(layer-weights (restrict-layer layer)))) 
							      (transform n m to m n))))
	       (batch-size (first (shape-dimensions (lazy-array-shape input))))
		   (cut-b (lazy-overwrite (lazy-reshape 0.0 (~ batch-size ~ target-size ~ target-size)) 
												(lazy-reshape 1.0 (~ batch-size ~ 1 (1- target-size) ~ 1 (1- target-size)))))
		   (input (lazy-reshape input (~ batch-size ~ (* source-size source-size))))
		   (result (lazy-reshape (matmul input weights)(~ batch-size ~ target-size ~ target-size))))
		   (lazy #'* cut-b result)))
		   

		
		 