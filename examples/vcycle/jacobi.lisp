(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)


(defclass jacobi-layer (layer)
  ((model :accessor model
		  :initarg :model
		  :initform (error "argument model required"))
   (level 
      :initarg :level
	  :accessor level
	  :initform (error "argument level required"))
   (w :initarg :w :accessor w :initform 0.6)))
	  

(defmethod call ((layer jacobi-layer) v &rest args)
	(let* ((input-size (1+ (expt 2 (level layer))))
		  (batch-size (first (shape-dimensions (lazy-array-shape v))))
		  (f (first args))
		  (c (second args))
		  (h (/ 1d0 (1- input-size)))
		  (interior (~ batch-size ~ 1 (1- input-size) ~ 1 (1- input-size)))
		  (cv (lazy #'* c v)))
		  
		  (lazy #'+ (lazy #'* (w layer) 
		            (lazy-overwrite v 					
						(lazy #'*  (lazy #'/ 0.25 (lazy-reshape c interior))
							(lazy #'+
                               (lazy-reshape  cv (transform b i j to b (1+ i) j) interior)
                              (lazy-reshape  cv (transform b i j to b (1- i) j) interior)
                              (lazy-reshape  cv (transform b i j to b i (1+ j)) interior)
                              (lazy-reshape  cv (transform b i j to b i (1- j)) interior)
                              (lazy-reshape (lazy #'* (* h h) f) interior)))))
			 (lazy #'* (- 1d0 (w layer)) v))))
		  
	
		  
