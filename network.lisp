
(in-package :common-lisp-user)
(defpackage #:lispnet.network
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.layers
   #:lispnet.initializers
   #:lispnet.trainable-parameter
   #:lispnet.loss
   #:lispnet.optimizer
  )
  (:export
   #:train
   #:predict
   #:fit
   ))

(in-package #:lispnet.network)



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Training


(defun train (network output-training-data
              &rest training-data-plist
              &key loss optimizer &allow-other-keys
			  )
  (let* ((trainable-parameters
           (remove-if-not #'trainable-parameter-p (network-parameters network)))
         (output-parameters
           (loop for output in (network-outputs network)
                 collect (make-instance 'parameter
                                        :element-type (element-type output)
				        :shape (array-shape output))))

         (lossfunc  (loop for output in (network-outputs network)
                  for output-parameter in output-parameters
                  collect		 
                  (funcall loss output-parameter output )))
		 (loss-network 
		            (apply #'make-network lossfunc))
		 (loss-network-gradient (list(lazy #'/ (first (network-outputs loss-network)) (first (network-outputs loss-network)))))
		 (gradient (differentiator (network-outputs loss-network) loss-network-gradient))

		 
         (training-network
           (apply #'make-network
                 (nconc (loop for trainable-parameter in trainable-parameters
                        collect
                              (funcall gradient trainable-parameter)) lossfunc)))
         (n nil)
	 (losses '())
         (gradients (loop for i below (list-length trainable-parameters) collect (lazy-reshape 0.0 (~))))
         )    
    ;; Determine the training data size.
    (dolist (data output-training-data)
      (if (null n)
          (setf n (range-size (first (shape-ranges (array-shape data)))))
          (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    (alexandria:doplist (parameter data training-data-plist) ()
      (unless (symbolp parameter)
        (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    ;; Iterate over the training data.
    (loop for index below n do
          ;; Assemble the arguments.
          (let ((args '()))
            ;; Inputs.
            (alexandria:doplist (parameter data training-data-plist)
                                (unless (symbolp parameter)
                                  (push parameter args)
                                  ;; (push (lazy-slice data index) args)))
                                  (push (lazy-drop-axes (lazy-slices data (range index (+ index 1)) ) 0) args)))
            

            ;; Outputs.
            (loop for data in output-training-data
                  for output-parameter in output-parameters do
                  (push output-parameter args)
                  (push  (lazy-drop-axes (lazy-slices data (range index (+ index 1)) ) 0) args)) 

            ;; Trainable parameters.
            (dolist (trainable-parameter trainable-parameters)
              (push trainable-parameter args)
              (push (trainable-parameter-value trainable-parameter) args))
                    
            ;; Forward + backward pass
            
            (let* ((net-out-values
                    (first (multiple-value-list
                            (apply #'call-network training-network (reverse args)))))
                   (loss-out (compute(first(last net-out-values))))
                   (new-gradients (butlast net-out-values 1))
                   )
              (loop for i below (list-length new-gradients) do
                    (setf (nth i gradients) (lazy #'+ (nth i new-gradients)
                        (nth i gradients))))

              (push loss-out losses))
            )
          
          )

   

    ;;Average gradients
    (loop for i below (length gradients) do
          (setf (nth i gradients) (lazy #'/ (nth i gradients) n)

      ))
	;;Update weights  
    (update-weights optimizer :weights trainable-parameters :gradients gradients)


      
    ;; Return the trained network and batch loss.
  (values network (/ (reduce #'+ losses) (length losses)) )))

	
(defun fit(network input input-data label-data &key (epochs 10) (batch-size 100) (loss #'mse) optimizer)
   (let ((input-data-length (array-dimension input-data 0))
	 (label-data-length (array-dimension label-data 0))
         
	)
    (assert (= input-data-length label-data-length))
    (format t "Train on ~d samples~%" input-data-length)
    (loop for epoch from 1 to epochs do
    (format t "Epoch ~d/~d~%" epoch epochs)
    (let ((epoch-train-loss 0)
          (num-batches 0)
          )
    (loop for offset below input-data-length by batch-size do
          (let* ((batch-range (range offset (+ offset batch-size)))
                 (batch-data (lazy-slices input-data batch-range))
                 (batch-labels (lazy-slices label-data batch-range))

                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                 )
			
            (multiple-value-bind (trained-net batch-loss)
            (train network (list batch-output)
                   :loss loss :optimizer optimizer
                   input batch-input)
              (incf epoch-train-loss batch-loss)
              (incf num-batches 1)

              ) 
            ))
      (format t "Loss: ~S ~%" (/ epoch-train-loss num-batches))

     )
     )
          )
	network
     )


	

	
;;;;; Prediction
(defun predict (network input)

   (values-list (apply #'call-network
    network
		(loop for parameter in (network-parameters network)
			collect parameter
			collect (if (trainable-parameter-p parameter)
						(trainable-parameter-value parameter)
						input))))
)





