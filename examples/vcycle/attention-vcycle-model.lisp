(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass attention-vcycle-model (model)
  ((pre-smoothing :initarg :pre-smoothing :accessor pre-smoothing :initform 2)
   (post-smoothing :initarg :post-smoothing :accessor post-smoothing :initform 2)
   (coarse-smoothing :initarg :coarse-smoothing :accessor coarse-smoothing :initform 1)
   (maximum-depth :initarg :maximum-depth :accessor maximum-depth :initform nil)))
  



(defun l2norm-error (residual-layer v f c)
	(let* ((batch-size (first (shape-dimensions (lazy-array-shape v))))
		  (r (call residual-layer v f c)))
		  (lazy #'/ (lazy-reduce #'+
					(lazy #'sqrt 
							(lazy-allreduce-batchwise (lazy #'* r r) #'+)		
							))
					 batch-size)))


(defmethod vcycle((model attention-vcycle-model) v f c &key (depth 0))

	
	(let* ((size (second (shape-dimensions (lazy-array-shape v))))
		   (level  (floor (log (1- size) 2)))
		   (smoother-rgbs (create-layer 'rgbs-layer model :level level :w 0.6 :model model))
		   (smoother-jacobi (create-layer 'jacobi-layer model :level level :w 0.6 :model model)))
		   ;;(when (= depth 0) (progn (setq restrict-c (create-layer 'att-restriction-layer model :filters 6 :model model ))));;:kernel-initializer #'glorot-uniform
							 ;;(setq prolongate (create-layer 'att-prolongation-layer model  :filters 6 :model model))))
							 
		   (if (or (= level 1) (and (not (null (maximum-depth model)))(= depth (maximum-depth model))))
				(progn (loop for i below (coarse-smoothing model) do
						(setq v (call smoother-jacobi v f c)))
						v)
		    (let* ((residual (create-layer 'residual-layer model :level level))
					(restrict-c (create-layer 'att-restriction-layer model :filters 6 :model model ))			;;:kernel-initializer #'glorot-uniform
			       (restrict (create-layer 'att-restriction-layer model :filters 1 :model model :trainable nil ));;:kernel-initializer #'glorot-uniform
				;;  (restrict-normal (create-layer 'restriction-layer model :level level :trainable nil))
			  	;;(prolongate-normal (create-layer 'prolongation-layer model :level (1- level) :restrict-layer restrict-normal))
				  (prolongate (create-layer 'att-prolongation-layer model :filters 1  :model model :restrict-layer restrict))
					(c2h (lazy #'+ 1 (lazy #'abs (call restrict-c (lazy #'- c 1) c)))))
					
					;;(c2h (coarse-func c)))
				;; (c2h (lazy #'+ 1 (lazy #'abs (call restrict (lazy #'- c 1) c))))) ;; (coarse-func c))) ;; (call restrict c c) ;;(lazy #'+ 1d0 (lazy #'abs (call restrict c c)))
				   (loop for i below (pre-smoothing model) do
						(setq v (call smoother-rgbs v f c)))
				   (setq rh (call residual v f c))
				   (setq r2h (call restrict rh c))
				   (setq e2h (vcycle model (lazy-reshape 0.0 (lazy-array-shape r2h)) r2h c2h :depth (1+ depth) ));;:restrict-c restrict-c
				   (setq eh (call prolongate e2h c2h))
				   (setq v (lazy #'+ v eh))
				   (loop for i below (post-smoothing model) do
						(setq v (call smoother-rgbs v f c)))
					v))))
					
					
				
 				
		  
;;this forward pass returns residual of vycle, r = f - A @ vcycle(v0) => 0
(defmethod forward-train ((model attention-vcycle-model) input)
   (let*  ((v  (lazy-slice input 0 3))
		  (f  (lazy-slice input 1 3))
		  (c (lazy-slice input 2 3))
		  (size (second (shape-dimensions (lazy-array-shape input))))
		  (level  (floor (log (1- size) 2)))
		  (residual (make-instance 'residual-layer :level level))
		  (v1 (vcycle model v f c))
		  (r-last (l2norm-error residual v f c)))
		 (loop for i from 0 to -1 do
			(setf (model-state-layer-pointer (model-state model)) (1- (length (model-layers model))))
			(setf (model-state-weights-initialized (model-state model)) t)
			(setq v1 (vcycle model v1 f c))
			(when (= i 2) (setf r-last   (l2norm-error residual v1 f c))))
		  (let* ((residuum (l2norm-error residual v1 f c)))
				;;residuum)))
		(lazy #'/ residuum (lazy #'+ 1d-20 r-last))
		   
		   
		   )))
		 
		  
;;this forward pass returns result of vcycle
(defmethod forward-calc ((model attention-vcycle-model) input)
   (let*  ((v (lazy-drop-axes (lazy-slices input (range 0 1) 3) 3))
		  (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
		  (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
		 (prediction (vcycle model v f c))
          (size (second (shape-dimensions (lazy-array-shape input))))
   		  (level  (floor (log (1- size) 2)))
		  (residuum (call (create-layer 'residual-layer model :level level ) prediction f c)))
		;;  prediction))
		  prediction))
		  
;;this forward pass returns L2-norm of residuum  
(defmethod forward-residuum ((model attention-vcycle-model) input)
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