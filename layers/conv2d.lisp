(in-package #:lispnet)



(defclass conv2d-layer (layer)
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
    :initform '())))



(defmethod initialize-instance :after ((layer conv2d-layer) &rest initargs)
  (let* ((n-weights (length (stencil layer))))
    ;; Generate stencil if not set
    (when (= (length (stencil layer)) 0)
      (setf (stencil layer) (make-2d-kernel (kernel-size layer)))
      (setf n-weights (length (stencil layer))))
    (setf (layer-weights layer) (list (make-trainable-parameter
                                       :shape (~ (out-channels layer) ~ n-weights ~ (in-channels layer)))))))

(defmethod layer-compile ((layer conv2d-layer))
  (let* ((trainable-parameter (first (layer-weights layer)))
         (s (lazy-array-shape (weights trainable-parameter)))
         (fan-in (* (nth 1 (shape-dimensions s)) (nth 2 (shape-dimensions s))))
         (fan-out (/ (* (nth 0 (shape-dimensions s)) (nth 1 (shape-dimensions s))) (reduce #'* (strides layer)))))
    (setf (weights-value trainable-parameter)
          (init-weights :shape s :mode #'glorot-uniform :fan-in fan-in :fan-out fan-out))))

(defun make-conv2d-layer (model &key in-channels (out-channels 1) (kernel-size 3) (strides '(1 1)) (padding "valid") (stencil '()) (activation nil))
  (let ((layer (make-instance 'conv2d-layer :in-channels in-channels
                              :out-channels out-channels :kernel-size kernel-size
                              :strides strides :padding padding :stencil stencil :activation activation)))
    (push layer (model-layers model))
    layer))

(defmethod call ((layer conv2d-layer) input)
  (let* ((lower-bounds (make-array 2 :initial-element 0))
         (upper-bounds (make-array 2 :initial-element 0))
         (stencil (stencil layer))
         (filters (weights (first (layer-weights layer))))
         (batch-size (first (shape-dimensions (lazy-array-shape input))))
         (input-pad input))
    (loop for offsets in stencil do
      (assert (= (length offsets) 2)))
    ;; Determine the bounding box of the stencil.
    (loop for offsets in stencil do
      (loop for offset in offsets
            for index from 0 do
              (minf (aref lower-bounds index) offset)
              (maxf (aref upper-bounds index) offset)))
    ;; Padding
    (when (string-equal (padding layer) "same")
      (setf input-pad (pad input :paddings (append '((0 0)(0 0))
                                                   (loop for lb across lower-bounds
                                                         for ub across upper-bounds collect
                                                                                    (list (abs lb) ub))))))
    (let* ((interior-shape (~l
                            (loop for lb across lower-bounds
                                  for ub across upper-bounds
                                  for range in (cdr (cdr(shape-ranges (lazy-array-shape input-pad))))
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
               (apply #'lazy-fuse
                      (loop for filter-index below (out-channels layer)
                            collect
                            (lazy-reshape
                             (lazy-reduce #'+
                                          (lazy-reshape
                                           (apply #'lazy #'+
                                                  (loop for offsets in stencil
                                                        for offset-index from 0 collect
                                                                                (lazy #'* (lazy-reshape (lazy-slice (lazy-slice filters filter-index) offset-index) (transform A to 0 A 0 0))
                                                                                      (lazy-reshape input-pad
                                                                                                    (make-transformation
                                                                                                     :offsets (list* 0 0 (mapcar #'- offsets)))
                                                                                                    (~ batch-size ~ (in-channels layer) ~s interior-shape)))))
                                           (transform A B C D  to B A C D)))
                             (~ filter-index (+ 1 filter-index) ~ batch-size ~s interior-shape))))
               (~ (out-channels layer) ~ batch-size ~s (stride-shape interior-shape (strides layer)))
               (transform c b w h to b c w h)))))
      (if (layer-activation layer)(funcall (layer-activation layer) result)
          result))))
