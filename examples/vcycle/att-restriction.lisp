(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)



(defclass att-restriction-layer (layer)
  ((model :accessor model
		  :initarg :model
		  :initform (error "argument model required"))
   (target-size :accessor target-size)
   (source-size :accessor source-size)
   (att-conv :accessor att-conv)
   (restrict-conv :accessor restrict-conv)
   (kernel-initializer :accessor kernel-initializer :initarg :kernel-initializer :initform #'conv2d-restrict-weights)
   (filters :initarg :filters :accessor filters :initform 4)
   (trainable :initarg :trainable :accessor trainable :initform t)))



(defun conv2d-restrict-weights (&key shape fan-in fan-out element-type)
  (let* ((in-channels (nth 1 (shape-dimensions shape)))
		 (out-channels (nth 2 (shape-dimensions shape)))
		 (array (lazy #'/ (lazy-array (make-array '(9) :element-type *network-precision* :initial-contents '(1d0 2d0 1d0 2d0 4d0 2d0 1d0 2d0 1d0))) 16d0)))
		 (compute(lazy-reshape array (transform n to n 0) (~ 9 ~ in-channels) (transform n i to n i 0)(~ 9 ~ in-channels ~ out-channels)))))


(defun channel-softmax (input)
  (let* ((input (lazy #'abs input))
		 (shape (lazy-array-shape input))
		 (channels (nth 3 (shape-dimensions shape)))
		 (y (nth 1 (shape-dimensions shape)))
		 (x (nth 2 (shape-dimensions shape)))
		 (batch-size (nth 0 (shape-dimensions shape)))
		 (cmax (lazy-reduce #'max (lazy-reshape input (transform b y x c to c b y x))))
		 (c (lazy-collapse (lazy-reshape cmax (transform b y x to b y x 0) shape)))
		(array (lazy #'exp (lazy #'- input c)))
		(sums (lazy-reduce #'+  (lazy-reshape array (transform b y x c to c b y x)))))			
        (lazy #'/ array  (lazy-reshape sums (transform b y x to b y x 0) (~ batch-size ~ y ~ x ~ channels)))))
								  

(defmethod initialize-instance :after ((layer att-restriction-layer) &rest initargs)	  
		(if (> (filters layer) 1) 
		   (progn
				(setf (att-conv layer) (make-instance 'conv2d-layer :in-channels 1 :out-channels (filters layer) :kernel-size 3 :padding "same" :activation #'channel-softmax :trainable (trainable layer)));;
				(setf (restrict-conv layer) (make-instance 'conv2d-layer  :in-channels (filters layer) :out-channels 1 :kernel-size 3 :padding "same" :strides '(2 2) :kernel-initializer (kernel-initializer layer)  :trainable (trainable layer))) ;; 
				(setf (layer-weights layer)  (append (layer-weights (att-conv layer)) (layer-weights (restrict-conv layer))))
				)
		   (progn 
				(setf (restrict-conv layer) (make-instance 'conv2d-layer  :in-channels (filters layer) :out-channels 1 :kernel-size 3 :padding "same" :strides '(2 2) :kernel-initializer (kernel-initializer layer)  :trainable (trainable layer))) ;; 
				(setf (layer-weights layer) (layer-weights (restrict-conv layer))))))					   
		
(defmethod layer-compile ((layer att-restriction-layer))
	(when (> (filters layer) 1) 
		(layer-compile (att-conv layer)))
	(layer-compile (restrict-conv layer)))
		
		
(defmethod call ((layer att-restriction-layer) input &rest args)
	(let* ((level (floor (log (1- (second (shape-dimensions (lazy-array-shape input)))) 2)))
		   (source-size (1+ (expt 2 level )))
		   (target-size (1+ (expt 2 (1- level ))))
	       (batch-size (first (shape-dimensions (lazy-array-shape input))))
		   (c (first args))
		   (c-max (lazy #'+ 1d-10 (lazy-allreduce-batchwise c #'max)))
		   (c  (lazy #'/ c  (lazy-reshape c-max (transform b to b 0 0) (~ batch-size ~ source-size ~ source-size))))
		   (rh  (lazy-reshape input (transform b y x  to b y x 0) (~ batch-size ~ source-size ~ source-size ~ (filters layer))))
		   (cut-b (lazy-collapse (lazy-overwrite (lazy-reshape (coerce 0 *network-precision*) (~ batch-size ~ target-size ~ target-size))
												(lazy-reshape (coerce 1 *network-precision*) (~ batch-size ~ 1 (1- target-size) ~ 1 (1- target-size))))))
		  
		   (attention-maps (if (> (filters layer) 1) 
								(call (att-conv layer)  (lazy-reshape c (transform b y x to b y x 0))) 
								1.0))
				  (att-input (lazy #'*  attention-maps rh))
				  (restrict-input (call (restrict-conv layer) att-input))
				  (result (lazy-reshape restrict-input (transform b y x 0 to b y x))))


				     (lazy-collapse (lazy-overwrite (lazy-reshape 0.0 (~ batch-size ~ target-size ~  target-size))  ;; (lazy-collapse(lazy-reshape input (~ batch-size ~ 0 source-size 2 ~ 0 source-size 2)))
							(lazy-reshape result (~ batch-size ~ 1 (1- target-size) ~ 1 (1- target-size)))))))

		
		 
