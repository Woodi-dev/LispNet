(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)


(defclass rgbs-layer (layer)
  ((model :accessor model
	  :initarg :model
	  :initform (error "argument model required"))
   (level 
    :initarg :level
    :accessor level
    :initform (error "argument level required"))
   (w :initarg :w :accessor w :initform 0.6)))
	  


(defun rbgs-smooth (v f h c w space)
  (let* ((input-size (second (shape-dimensions (lazy-array-shape v))))
	 (batch-size (first (shape-dimensions (lazy-array-shape v))))
	 (cv (lazy #'* c v))
	 (result (lazy #'* (lazy #'/ 0.25 (lazy-reshape c space))
                       (lazy #'+
                             (lazy-reshape  cv  (transform b i j to b (+ i 1) j) space)
                             (lazy-reshape cv (transform b i j to b (- i 1) j)  space )
                             (lazy-reshape cv  (transform b i j to b i (+ j 1))  space)
                             (lazy-reshape cv   (transform b i j to b i (- j 1)) space)
                             (lazy-reshape (lazy #'* (* h h) f) space)))))
    (lazy #'+ (lazy #'* w (lazy-overwrite v result))  (lazy #'* (- 1d0 w) v))))
							  
	  

(defun rbgs (v f h c w)
  (let* ((input-size (second (shape-dimensions (lazy-array-shape v))))
	 (batch-size (first (shape-dimensions (lazy-array-shape v))))
	 (red-spaces (~ batch-size ~ 1 (1- input-size) 2 ~ 1 (1- input-size) 2 ))
	 (red2-spaces (~ batch-size ~ 2 (1- input-size) 2 ~ 2 (1- input-size) 2 ))
	 (black-spaces  (~ batch-size ~ 1 (1- input-size) 2 ~ 2 (1- input-size) 2))
	 (black2-spaces  (~ batch-size ~ 2 (1- input-size) 2 ~ 1 (1- input-size) 2)))


    (setf v (rbgs-smooth v f h c w red-spaces))
    (setf v (rbgs-smooth v f h c w red2-spaces))
    (setf v (rbgs-smooth v f h c w black-spaces))
    (setf v (rbgs-smooth v f h c w black2-spaces))
    v))
		
(defmethod call ((layer rgbs-layer) v &rest args)
  (let* ((input-size (1+ (expt 2 (level layer))))
	 (batch-size (first (shape-dimensions (lazy-array-shape v))))
	 (f (first args))
	 (c (second args))
	 (h (/ 1d0 (1- input-size))))
    (rbgs v f h c (w layer))))
		  
	
		  
