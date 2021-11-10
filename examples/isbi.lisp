(in-package #:lispnet)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; classification of IEEE International Symposium on Biomedical Imaging 2021 data


(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet") path)))

(defparameter *isbi-train-images* (load-array "examples/isbi-data/train-images.npy"))
(defparameter *isbi-train-labels* (load-array "examples/isbi-data/train-labels.npy"))
(defparameter *isbi-test-images* (load-array "examples/isbi-data/test-images.npy"))




(defclass unet-model (model)
  ((conv1 :accessor conv1)
   (conv2 :accessor conv2)
   (conv3 :accessor conv3)
	(conv4 :accessor conv4)
   (maxpool1 :accessor maxpool1)
   (maxpool2 :accessor maxpool2)
   (tconv1 :accessor tconv1)
   (tconv2 :accessor tconv2)
   ))

(defmethod initialize-instance :after ((model unet-model) &rest initargs)
   ;;(setf (conv1 model) (make-conv2d-layer model :in-channels 1 :out-channels 64 :padding "same" :kernel-size 3  :strides '(1 1) ))
   ;;(setf (conv2 model) (make-conv2d-layer model :in-channels 64 :out-channels 1 :padding "same" :kernel-size 3  :strides '(1 1) ))
   
    (setf (conv1 model) (make-conv2d-layer model :in-channels 1 :out-channels 4 :padding "same" :kernel-size 3  :strides '(1 1) ))
   (setf (conv2 model) (make-conv2d-layer model :in-channels 4 :out-channels 4 :padding "same" :kernel-size 3  :strides '(1 1) ))
   (setf (conv3 model) (make-conv2d-layer model :in-channels 4 :out-channels 4 :padding "same" :kernel-size 3  :strides '(1 1) ))
   (setf (conv4 model) (make-conv2d-layer model :in-channels 4 :out-channels 1 :padding "same" :kernel-size 3  :strides '(1 1) ))

  ;; (setf (conv3 model) (make-conv2d-layer model :in-channels 32 :out-channels 16 :padding "same" :kernel-size 3  :strides '(1 1) ))
  ;; (setf (conv4 model) (make-conv2d-layer model :in-channels 16 :out-channels 1 :padding "same" :kernel-size 3  :strides '(1 1) ))

  ;; (setf (maxpool1 model) (make-maxpool2d-layer model :pool-size '(2 2)))
  ;; (setf (maxpool2 model) (make-maxpool2d-layer model :pool-size '(2 2)))
  ;; (setf (tconv1 model) (make-transposed-conv2d-layer model :in-channels 16 :out-channels 8 :padding "same" :kernel-size 2  :strides '(2 2) :activation #'sigmoid))
  ;; (setf (tconv2 model) (make-transposed-conv2d-layer model :in-channels 8 :out-channels 1 :padding "same" :kernel-size 2  :strides '(2 2) :activation #'sigmoid))

 )

(defmethod forward ((model unet-model) input)
    (lazy-drop-axes
	  ;;(call (tconv2 model)  
		;;(call (tconv1 model)  
		  ;;(call (maxpool2 model)
		  			(call (conv4 model)

						(call (conv3 model)

			(call (conv2 model)
		;;	(call (maxpool1 model)
				(call (conv1 model)
                     (lazy-reshape input (transform a b c to a b c 0)))))) 3))

						  
						  

(defun train-isbi()
  (let*  ((optimizer (make-adam :learning-rate 0.001))
          (model (make-instance 'unet-model)))
    (model-compile model :loss #'mse :optimizer optimizer :metrics (list #'binary-accuracy)) ;;
    (let* ((train-input-data
             (compute (lazy-slices (lazy #'/ *isbi-train-images* 255.0) (range 0 1))))
           (train-label-data
             (compute (lazy-slices (lazy #'/ *isbi-train-labels* 255.0)(range 0 1)) ))

           (val-input-data
             (compute (lazy-slices (lazy #'/ *isbi-train-images* 255.0)(range 20 21))))
           (val-label-data
             (compute (lazy-slices (lazy #'/ *isbi-train-labels* 255.0)(range 20 21)) )))
      (format t "--------------------------~%")
      (model-summary model)
      (fit model train-input-data train-label-data val-input-data val-label-data :epochs 1 :batch-size 1)
      ;;(format t "check test data~%")
     )))

;;(train-isbi)


