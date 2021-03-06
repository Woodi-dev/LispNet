(in-package #:lispnet)

(defclass flatten-layer (layer)
	())
	  

(defmethod layer-compile ((flatten-layer layer)))

(defmethod call ((layer flatten-layer) input &rest args)
  (assert (> (lazy-array-rank input) 1))
  (let* ((dimensions (shape-dimensions (lazy-array-shape input)))
		 (samplesize (reduce #'*(cdr dimensions))))
		 (lazy-reshape input (~ (first dimensions) ~ samplesize))))
