(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; performance tests



(defclass perf-conv2d-model (model)
  ((num-layers :initarg :num-layers :accessor num-layers)
   (kernel-size :initarg :kernel-size :accessor kernel-size)))

(defmethod initialize-instance :after ((model perf-conv2d-model) &rest initargs))

(defmethod forward ((model perf-conv2d-model) input)
  (let ((result input))
    (loop for i below (num-layers model) do
	  (let* ((conv-layer (create-layer 'conv2d-layer model :kernel-size (kernel-size model) :in-channels 3 :out-channels 3 :padding "valid")))
	    (setf result (call conv-layer result))))		
    result))
	  
(defclass perf-dense-model (model)
  ((num-layers :initarg :num-layers :accessor num-layers)
   (neurons :initarg :neurons :accessor neurons)))

(defmethod initialize-instance :after ((model perf-dense-model) &rest initargs))

(defmethod forward ((model perf-dense-model) input)
  (let ((result input))
    (loop for i below (num-layers model) do
	  (let* ((dense-layer (create-layer 'dense-layer model :out-features (neurons model))))
	    (setf result (call dense-layer result))))		
    result))
	  	  
	  
	
(defun randomData(shape element-type) 
  (let ((array (make-array (shape-dimensions shape) :element-type element-type)))
    (loop for index below (array-total-size array) do
          (setf (row-major-aref array index)(coerce (random 1.0 ) element-type)))
    array))				
						
					
 
						  
(defun test-conv-perf()

  (setf petalisp.core:*backend* (petalisp.xmas-backend:make-xmas-backend ))

  ;;(setf petalisp.core:*backend* (petalisp.multicore-backend:make-multicore-backend ))
  (setf *network-precision* 'single-float)
  (let*  ((optimizer (make-adam :learning-rate 0.001))
	  (layers 4)	  
          (model (make-instance 'perf-conv2d-model :num-layers layers :kernel-size 3))
	  (in-dim 32)
	  (out-dim (- in-dim (* 2 layers)))
	  (channels 3)
	  (train-input-data (randomData (~ 4096 ~ in-dim ~ in-dim ~ channels) 'single-float))
	  (train-label-data (randomData (~ 4096 ~ out-dim ~ out-dim ~ channels) 'single-float))
	  (val-input-data (randomData (~ 1024 ~ in-dim ~ in-dim ~ channels) 'single-float)) 
	  (val-label-data (randomData (~ 1024 ~ out-dim ~ out-dim ~ channels) 'single-float)))
          
    (format t "--------------------------~%")
	 
    (model-compile model :loss #'mse :optimizer optimizer)
    (model-summary model :sample-shape (~ in-dim ~ in-dim ~ channels))
    (time(fit model train-input-data train-label-data val-input-data val-label-data :epochs 20 :batch-size 64))))
		
(defun test-dense-perf()

  (setf petalisp.core:*backend* (petalisp.xmas-backend:make-xmas-backend))

  ;;(setf petalisp.core:*backend* (petalisp.multicore-backend:make-multicore-backend))
  (setf *network-precision* 'single-float)
  (let*  ((optimizer (make-adam :learning-rate 0.001))
	  (layers 1)
	  (neurons 256)		  
          (model (make-instance 'perf-dense-model :num-layers layers :neurons neurons))
	  (train-input-data (randomData (~ 4096 ~ neurons) 'single-float))
	  (train-label-data (randomData (~ 4096 ~ neurons) 'single-float))
	  (val-input-data (randomData (~ 1024 ~ neurons) 'single-float)) 
	  (val-label-data (randomData (~ 1024 ~ neurons) 'single-float)))
          
    (format t "--------------------------~%")
	 
    (model-compile model :loss #'mse :optimizer optimizer)
    (model-summary model :sample-shape (~ neurons))
    (time(fit model train-input-data train-label-data val-input-data val-label-data :epochs 20 :batch-size 64))))	
		
;;(test-conv-perf)
;;(test-dense-perf)
