(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass att-prolongation-layer (layer)
  ((model :accessor model
		  :initarg :model
		  :initform (error "argument model required"))
  (restrict-layer
      :initarg :restrict-layer
	  :accessor restrict-layer
	  :initform (error "argument restrict-layer required"))
   (target-size :accessor target-size)
   (source-size :accessor source-size)
   (filters :initarg :filters :accessor filters :initform 4)
   (prolongate-conv :accessor prolongate-conv)
   (att-conv :accessor att-conv)))

		

(defmethod initialize-instance :after ((layer att-prolongation-layer) &rest initargs)
		 ;;  (setf (att-conv layer) (make-instance 'conv2d-layer  :in-channels 1 :out-channels (filters layer) :kernel-size 3 :padding "same"  :activation #'channel-softmax));;:activation #'channel-softmax
		   (setf (prolongate-conv layer) (make-instance 'transposed-conv2d-layer  :in-channels (filters layer) :out-channels 1 :kernel-size 3 :strides '(2 2) :padding "valid" :kernel-initializer #'conv2d-restrict-weights)))	;; 
		  ;; (setf (layer-weights layer) (list (layer-weights (att-conv layer)) (layer-weights (prolongate-conv layer)))))
									   
		
(defmethod layer-compile ((layer att-prolongation-layer)))
	;;(layer-compile (att-conv layer))
	;;(layer-compile (prolongate-conv layer)))
	
	
(defmethod call ((layer att-prolongation-layer) input &rest args)


	(let* ((level (floor (log (1- (second (shape-dimensions (lazy-array-shape input)))) 2)))
		   (source-size (1+ (expt 2 level )))
		   (target-size (1+ (expt 2 (1+ level))))
	       (filters (filters layer))
		   (batch-size (first (shape-dimensions (lazy-array-shape input))))
		   (c (first args))
		   (c-max (lazy #'+ 1d-10  (lazy-allreduce-batchwise c #'max)))
		   (c (lazy #'/ c (lazy-reshape c-max (transform b to b 0 0) (~ batch-size ~ source-size ~ source-size))))		   
		   (r2h (lazy-reshape input  (transform b y x  to b y x 0) (~ batch-size ~ source-size ~ source-size ~ filters ))))


		
		   ;; reuse weights from restriction layer
		   (setf (layer-weights (prolongate-conv layer))  (layer-weights (restrict-conv (restrict-layer layer))))
		;;   (setf (layer-weights (att-conv layer))  (layer-weights (att-conv (restrict-layer layer))))
	;;	 
		   (let* (;;(attention-maps  (call (att-conv layer)  (lazy-reshape c (transform b y x to b y x 0))))
				  (att-input r2h);;(lazy #'*  attention-maps r2h))
				  (result (lazy #'* 4.0
						(lazy-reshape (lazy-collapse
										(lazy-reshape
											(call (prolongate-conv layer)  att-input) 	
											(~ batch-size ~ 1 (1+ target-size) ~ 1 (1+ target-size) ~ 1)))

							(transform b y x 0 to b y x)))))
				    (lazy-overwrite (lazy-reshape 0.0 (~ batch-size ~ target-size ~ target-size))
					 (lazy-reshape result (~ batch-size ~ 1 (1- target-size) ~ 1 (1- target-size))) )
					
											
											)))
		   
		   
	
		   

		
		 