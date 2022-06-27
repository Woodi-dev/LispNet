(in-package #:lispnet)

(defclass transposed-conv2d-layer (layer)
  ((in-channels
    :initarg :in-channels
    :accessor in-channels
    :initform (error "Missing layer argument"))
   (out-channels
    :initarg :out-channels
    :accessor out-channels
    :initform 1)
   (kernel-size
    :initarg :kernel-size
    :accessor kernel-size
    :initform 3)
   (strides
    :initarg :strides
    :accessor strides
    :initform '(1 1))
   (padding
    :initarg :padding
    :accessor padding
    :initform "valid")
   (stencil
    :initarg :stencil
    :accessor stencil
    :initform '())
   (trainable
    :initarg :trainable
	:accessor trainable
	:initform t)
   (kernel-initializer
	:initarg :kernel-initializer
	:accessor kernel-initializer
	:initform #'glorot-uniform)))



(defmethod initialize-instance :after ((layer transposed-conv2d-layer) &rest initargs)
  (let* ((n-weights (length (stencil layer))))
    ;; Generate stencil if not set
    (when (= (length (stencil layer)) 0)
      (setf (stencil layer) (reverse(make-2d-kernel (list (kernel-size layer) (kernel-size layer)))))
      (setf n-weights (length (stencil layer))))
    (setf (layer-weights layer) (list (make-trainable-parameter (model layer)
                                       :shape (~ n-weights ~ (in-channels layer) ~ (out-channels layer))
									   :trainable (trainable layer))))))

(defmethod layer-compile ((layer transposed-conv2d-layer))
  (let* ((trainable-parameter (first (layer-weights layer)))
         (s (lazy-array-shape (weights trainable-parameter)))
         (fan-in (* (nth 1 (shape-dimensions s)) (nth 2 (shape-dimensions s))))
         (fan-out (/ (* (nth 0 (shape-dimensions s)) (nth 1 (shape-dimensions s))) (reduce #'* (strides layer)))))
    (setf (weights-value trainable-parameter)
          (init-weights :shape s :mode (kernel-initializer layer) :fan-in fan-in :fan-out fan-out))))


(defmethod call ((layer transposed-conv2d-layer) input &key)
  (let* ((lower-bounds (make-array 2 :initial-element 0))
         (upper-bounds (make-array 2 :initial-element 0))
         (stencil (stencil layer))
         (filters (weights (first (layer-weights layer))))
		 (strides (strides layer))
         (batch-size (first (shape-dimensions (lazy-array-shape input)))))
    (loop for offsets in stencil do
      (assert (= (length offsets) 2)))
    ;; Determine the bounding box of the stencil.
    (loop for offsets in stencil do
      (loop for offset in offsets
            for index from 0 do
              (minf (aref lower-bounds index) offset)
              (maxf (aref upper-bounds index) offset)))

	
	
	(let* ((input-dim (shape-dimensions (lazy-array-shape input)))
		   (input-strided (lazy-overwrite 
									(lazy-reshape (coerce 0 *network-precision*) (~ batch-size ~ (- (* (nth 1 input-dim) (nth 0 strides)) 1)
													 ~ (- (* (nth 2 input-dim) (nth 1 strides)) 1) ~ (nth 3 input-dim)))
									(lazy-reshape input (transform b y x c to b (* y (nth 0 strides)) (* x (nth 1 strides)) c))))		   
		   (input-pad  (pad input-strided :paddings (list '(0 0)(list (* 2 (abs(aref lower-bounds 0))) (* 2(aref upper-bounds 0))) (list (* 2(abs(aref lower-bounds 1))) (* 2(aref upper-bounds 1))) '(0 0))))
		   (input-pad-ranges (shape-ranges (lazy-array-shape input-pad)))
		   (interior-shape  (~l
                            (loop for lb across lower-bounds
                                  for ub across upper-bounds
                                  for range in (list (nth 1 input-pad-ranges) (nth 2 input-pad-ranges))
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
                    (lazy-reduce #'+
                                     (lazy-reshape
                                           (apply #'lazy #'+
                                                  (loop for offsets in stencil
                                                        for offset-index from 0 collect
                                                                        (lazy #'* (lazy-reshape (lazy-slice filters offset-index) (transform c f to 0 0 0 c f))
                                                                                      (lazy-reshape input-pad
                                                                                                    (make-transformation
                                                                                                     :offsets (append '(0) (mapcar #'- offsets) '(0)))
                                                                                                    (~ batch-size ~s interior-shape ~ (in-channels layer))
																									(transform b y x c to b y x c 0)
																									))))
                                           (transform b y x c f  to c b y x f)))))))
	  (when (string-equal (padding layer) "same")
	  (setq result (lazy-reshape result
						(~ batch-size ~ (* (nth 1 input-dim) (nth 0 strides))
						 ~ (* (nth 2 input-dim) (nth 1 strides)) ~ (out-channels layer)))))
	  (setf result (lazy #'max result result))
	  						 
      (if (layer-activation layer)(funcall (layer-activation layer) result)
          result))))
