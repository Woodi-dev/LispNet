(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass vcycle-model (model)
  ((pre-smoothing :initarg :pre-smoothing :accessor pre-smoothing :initform 2)
   (post-smoothing :initarg :post-smoothing :accessor post-smoothing :initform 2)
   (coarse-smoothing :initarg :coarse-smoothing :accessor coarse-smoothing :initform 1)
   (maximum-depth :initarg :maximum-depth :accessor maximum-depth :initform nil)))
  
  
(defun coarse-func(array)
	(let* ((shape (lazy-array-shape array))
		   (batch-size (first (shape-dimensions shape)))
		   (input-size (second (shape-dimensions shape))))
			  (lazy-collapse (lazy-reshape array (~ batch-size ~ 0 input-size 2 ~ 0 input-size 2)))))
			  
(defun batch-l2norm (array)
	(let* ((size (second (shape-dimensions (lazy-array-shape array))))
          (batch-size (first (shape-dimensions (lazy-array-shape array))))
   		  (level  (floor (log (1- size) 2))))
					(lazy #'sqrt 
							(lazy-allreduce-batchwise (lazy #'* array array) #'+))))
		  

(defmethod jacobi((model vcycle-model) v f c)
     (let* ((size (second (shape-dimensions (lazy-array-shape v))))
		   (level  (floor (log (1- size) 2)))
		   (smoother (create-layer 'jacobi-layer model :level level :w 0.6 :model model)))
			(call smoother v f c)))


(defmethod vcycle((model vcycle-model) v f c &key (depth 0))
	(let* ((size (second (shape-dimensions (lazy-array-shape v))))
		   (level  (floor (log (1- size) 2)))
		   (jacobi (create-layer 'jacobi-layer model :level level :w 0.6 :model model)))
		   (if (or (= level 1) (and (not (null (maximum-depth model)))(= depth (maximum-depth model))))
				(progn (loop for i below (coarse-smoothing model) do
						(setq v (call jacobi v f c)))
						v)
		    (let* ((residual (create-layer 'residual-layer model :level level))
			       (restrict (create-layer 'restriction-layer model :level level))
				   (prolongate (create-layer 'prolongation-layer model :level (1- level) :restrict-layer restrict))
				   (c2h (coarse-func c)))
				   (loop for i below (pre-smoothing model) do
						(setq v (call jacobi v f c)))
				   (setq rh (call residual v f c))
				   (setq r2h (call restrict rh ))
				   (setq e2h (vcycle model (lazy-reshape 0.0  (lazy-array-shape r2h)) r2h c2h :depth (1+ depth)))
				   (setq eh (call prolongate e2h))
				   (setq v (lazy #'+ v eh))
				   (loop for i below (post-smoothing model) do
						(setq v (call jacobi v f c)))
					v))))
		  
;;this forward pass returns residual of vycle, r = f - A @ vcycle(v0) => 0
(defmethod forward-train ((model vcycle-model) input)
   (let*  ((v (lazy-drop-axes (lazy-slices input (range 0 1) 3) 3))
		  (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
		  (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
		  (prediction (vcycle model v f c))
          (size (second (shape-dimensions (lazy-array-shape input))))
   		  (level  (floor (log (1- size) 2)))
		  (residual (create-layer 'residual-layer model :level level)))
		  (call residual prediction f c)))
		  
;;this forward pass returns result of vcycle
(defmethod forward-calc ((model vcycle-model) input)
   (let*  ((v (lazy-drop-axes (lazy-slices input (range 0 1) 3) 3))
		  (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
		  (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
		  (prediction (vcycle model v f c))
          (size (second (shape-dimensions (lazy-array-shape input))))
   		  (level  (floor (log (1- size) 2)))
		  (residuum (call (create-layer 'residual-layer model :level level) prediction f c)))
		  prediction))
		  
;;this forward pass returns L2-norm of residuum  
(defmethod forward-residuum ((model vcycle-model) input)
   (let*  ((v (lazy-drop-axes (lazy-slices input (range 0 1) 3) 3))
		  (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
		  (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
          (size (second (shape-dimensions (lazy-array-shape input))))
	      (batch-size (first (shape-dimensions (lazy-array-shape input))))
   		  (level  (floor (log (1- size) 2)))
		  (residual (create-layer 'residual-layer model :level level))
		  (r (call residual v f c)))
		  (lazy #'/ (lazy-reduce #'+
					(lazy #'sqrt 
							(lazy-allreduce-batchwise (lazy #'* r r) #'+)		
							))
					 batch-size)
		  ))