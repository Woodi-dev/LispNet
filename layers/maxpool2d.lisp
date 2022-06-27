(in-package #:lispnet)

(defclass maxpool2d-layer (layer)
   ((strides
    :initarg :strides
    :accessor strides
    :initform nil)
   (padding
    :initarg :padding
    :accessor padding
    :initform "valid")
   (pool-size
    :initarg :pool-size
    :accessor pool-size
    :initform '(2 2))
   (stencil
    :initarg :stencil
    :accessor stencil
    :initform '())))
	  
   
(defmethod layer-compile ((flatten-layer layer)))

(defmethod call ((layer maxpool2d-layer) input &key)
  (let* ((lower-bounds (make-array 2 :initial-element 0))
         (upper-bounds (make-array 2 :initial-element 0))
         (batch-size (first (shape-dimensions (lazy-array-shape input))))
		 (channels (nth 3 (shape-dimensions (lazy-array-shape input))))
         (stencil (make-2d-kernel (pool-size layer)))		 
         (input-pad input)
		 (strides (if (strides layer) (strides layer) (pool-size layer))))
	
    ;; Determine the bounding box of the stencil.    
    (loop for offsets in stencil do
          (loop for offset in offsets
                for index from 0 do
                (minf (aref lower-bounds index) offset)
                (maxf (aref upper-bounds index) offset)))
    ;; Padding
    (when (string-equal (padding layer) "same")
      (setf input-pad (pad input :paddings (append '((0 0))
                                                   (loop for lb across lower-bounds
                                                         for ub across upper-bounds collect
                                                         (list (abs lb) ub)) '((0 0))))))							        						
    (let* ((input-pad-ranges (shape-ranges (lazy-array-shape input-pad)))
		   (spatial-pad-ranges (list (nth 1 input-pad-ranges) (nth 2 input-pad-ranges)))
		   (interior-shape (~l
                            (loop for lb across lower-bounds
                                  for ub across upper-bounds
                                  for range in spatial-pad-ranges
                                  collect
                                  (if (and (integerp lb)
                                           (integerp ub))
                                      (let ((lo (- (range-start range) lb))
                                            (hi (- (range-end range) ub)))
                                        (assert (< lo hi))
                                        (range lo hi))
                                      range))))
		(result 
		  (lazy-collapse
		    (lazy-reshape 
		       (apply #'lazy #'max
                  (loop for offsets in stencil
                        for offset-index from 0 collect                
                             (lazy-reshape input-pad
                                       (make-transformation
                                        :offsets (append '(0) (mapcar #'- offsets) '(0)))
                                       (~ batch-size ~s interior-shape ~ channels)
									   )))
				(~ batch-size ~s (stride-shape interior-shape strides) ~ channels))))) 
	   ;;(setf result (lazy #'max result result))

		result
		)))
