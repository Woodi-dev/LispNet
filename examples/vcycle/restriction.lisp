(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass restriction-layer (layer)
  ((level 
      :initarg :level
	  :accessor level
	  :initform (error "argument level required"))
   (target-size :accessor target-size)
   (source-size :accessor source-size)
   (trainable :accessor trainable :initarg :trainable :initform t)))



(defun create-restriction-matrix (level)
	(let* ((input-size (1+ (expt 2 level)))
	   (target-size  (1+ (expt 2 (1- level))))
	   (stencil (mapcar #'(lambda (x) (/ x 16.0)) '(1 2 1 2 4 2 1 2 1)))
       (array (make-array (list (* target-size target-size) (* input-size input-size)) :element-type *network-precision*)))
			(loop for y from 1 below (1- target-size) do
				(loop for x from 1 below (1- target-size) do
						(let ((middle (+ (* (* 2 y) input-size) (* 2 x))))
							(loop for i from 0 below 9 do
								(let ((target-index (+ (* y target-size) x))
									  (source-index (+ (+ middle (* (1- (mod i 3)) input-size)) (1- (floor i 3)))))
									  (setf (aref array target-index source-index) (coerce (nth i stencil) *network-precision*)))))))
		array))
		

(defmethod initialize-instance :after ((layer restriction-layer) &rest initargs)
	(let* ((input-size (1+ (expt 2 (level layer))))
	       (target-size (1+ (expt 2 (1- (level layer)))))
		   (weights (make-trainable-parameter :shape (~ (* input-size input-size) ~ (* target-size target-size)) :trainable (trainable layer))))
		   (setf (layer-weights layer) (list weights))
		   (setf (target-size layer) target-size)
		   (setf (source-size layer) input-size))) 
									   
		
(defmethod layer-compile ((layer restriction-layer))
   (let* ((weights (first (layer-weights layer)))
		  (array (compute (transpose (create-restriction-matrix (level layer))))))
		  (setf (weights-value weights) array)))
		
		
(defmethod call ((layer restriction-layer) input &rest args)
	(let* ((source-size (source-size layer))
		   (target-size (target-size layer))
	       (batch-size (first (shape-dimensions (lazy-array-shape input))))
		   (input (lazy-reshape input (~ batch-size ~ (* source-size source-size))))
		   (result (matmul input (weights (first (layer-weights layer))))))
		   (lazy-reshape result (~ batch-size ~ target-size ~ target-size))))

		
		 
