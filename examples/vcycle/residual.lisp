(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)


(defclass residual-layer (layer)
  ((level 
      :initarg :level
	  :accessor level
	  :initform (error "argument level required"))))

(defmethod call ((layer residual-layer) v &rest args)
	(let* ((input-size (1+ (expt 2 (level layer))))
		  (batch-size (first (shape-dimensions (lazy-array-shape v))))
		  (f (first args))
		  (c (second args))
		  (h (/ 1 (1- input-size)))
		  (interior (~ batch-size ~ 1 (1- input-size) ~ 1 (1- input-size)))
		  (zeros (lazy-reshape 0.0  (lazy-array-shape v)))
		  (cv (lazy #'* c v)))
		  
		  (lazy-overwrite zeros 
						 (lazy #'- (lazy-reshape f interior) 						
							(lazy #'* (/ 1.0 (* h h))
								(lazy #'-
								(lazy #'* 4.0 (lazy-reshape  cv interior))
								(lazy-reshape  cv (transform b i j to b (1+ i) j) interior)
								(lazy-reshape  cv (transform b i j to b (1- i) j) interior)
								(lazy-reshape  cv (transform b i j to b i (1+ j)) interior)
								(lazy-reshape  cv (transform b i j to b i (1- j)) interior)))))))
		
