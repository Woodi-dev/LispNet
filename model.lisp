(in-package #:lispnet)


(defstruct model-state
	running
	layer-pointer
	weights-initialized)

(defvar *network-precision* 'single-float)

(defclass model()
  ((layers
    :accessor model-layers
    :initform '())
   (loss
    :initarg :loss
    :accessor model-loss
    :initform nil)
   (optimizer
    :initarg :optimizer
    :accessor model-optimizer
    :initform nil)
   (metrics
    :initarg :metrics
    :accessor metrics
    :initform '())
   (output
    :initarg :output
    :accessor output
    :initform #'forward)	
   (state 
   :accessor model-state
   :initform (make-model-state :running nil :layer-pointer -1 :weights-initialized nil))))




(defmethod model-weights(model)
  (alexandria:flatten
        (loop for layer in (model-layers model) collect
                (loop for weight in (layer-weights layer) collect
                        weight))))

(defmethod model-compile ((model model) &key optimizer loss (metrics '()) &allow-other-keys)
        (setf (model-loss model) loss)
        (setf (model-optimizer model) optimizer)
        (setf (metrics model) metrics))


(defmethod initialize-weights ((model model) sample-shape)
    (setf (model-state-weights-initialized (model-state model)) nil)
	(let* ((input-parameter (make-unknown :shape (~ 1 ~s sample-shape) :element-type *network-precision*))
			;; Generate a computation graph to trigger constructor calls in forward pass
		   (graph (funcall (output model) model input-parameter)))		 
			;; Initialize layer weights
           (loop for layer in (model-layers model) do
                 (layer-compile layer))
			;; Initialize last gradients based on layer weights	

			
    (setf (model-state-weights-initialized (model-state model)) t)
	  (setf (model-state-layer-pointer (model-state model)) (1- (length (model-layers model))))
	))
		
(defmethod clear-weights ((model model))  
	       (loop for layer in (model-layers model) do
                 (setf (weights layer) '()))
	(setf (model-state-weights-initialized (model-state model)) nil))


(defgeneric forward (model input))

(defmethod fit((model model) train-input-data train-label-data val-input-data val-label-data &key (epochs 10) (batch-size 10) (early-stop nil) (early-stop-delta 0))
   (let* ((train-input-data-length (array-dimension train-input-data 0))
           (train-label-data-length (array-dimension train-label-data 0))
           (val-input-data-length (array-dimension val-input-data 0))
           (val-label-data-length (array-dimension val-label-data 0))
           (sample-shape (~l (mapcar #'range (cdr (array-dimensions train-input-data)))))
		   (epoch-train-losses '())
		   (epoch-val-losses '())
		   (epoch-train-metrices '())
		   (epoch-val-metrices '())
		   (best-epoch 1)
		   (best-weights nil))
    (when (not (and (model-optimizer model) (model-loss model))) 
		(error "Network needs to be compiled before"))
    (assert (= train-input-data-length train-label-data-length))
    (assert (= val-input-data-length val-label-data-length))
	;; Initialize model weights
	(when (not (model-state-weights-initialized (model-state model)))
		(initialize-weights model sample-shape))
	;;init optimizers
	(optimizer-compile (model-optimizer model) :model model)	
    (format t "~%Train on ~d samples~%" train-input-data-length)
    (loop for epoch from 1 to epochs when (or (null early-stop) (<= epoch (+ best-epoch early-stop))) do
    (format t "Epoch ~d/~d~%" epoch epochs)
    (let ((batch-train-losses '())
                  (batch-val-losses '())
                  (metrics-train (loop for i below (list-length (metrics model)) collect '()))
                  (metrics-val (loop for i below (list-length (metrics model)) collect '()))
                  (time-start (get-internal-real-time)))	  
    ;;Training
    (loop for offset below train-input-data-length by batch-size
                  for batch from 0 do
                 (format  t "Batch: ~S~%" batch)
		
          (let* ((batch-range (range offset (min train-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices train-input-data batch-range))
                 (batch-labels (lazy-slices train-label-data batch-range))
                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                 (input-parameter (make-unknown :shape (~ (range-size batch-range) ~s sample-shape) :element-type *network-precision*)))
				 	
            (multiple-value-bind (batch-loss metrics)
          (train-test model (list batch-output)
                   :loss (model-loss model) :optimizer (model-optimizer model) :mode "train"
                   :input-parameter input-parameter :batch-input batch-input)
              (setq batch-train-losses (append batch-train-losses (list batch-loss)))
                          (loop for metric in metrics
                                        for i from 0 do
                                            (setf (nth i metrics-train) (list* metric (nth i metrics-train)))))))
					 
											

        ;; Validation
        (loop for offset below val-input-data-length by batch-size
                  for batch from 0 do
                    ;;    (format  t "Batch: ~S~%" batch)
          (let* ((batch-range (range offset (min val-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices val-input-data batch-range))
                 (batch-labels (lazy-slices val-label-data batch-range))
                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                                 (input-parameter (make-unknown :shape (~ (range-size batch-range) ~s sample-shape) :element-type *network-precision*)))

            (multiple-value-bind (batch-loss metrics)
           (train-test model (list batch-output)
                   :loss (model-loss model) :optimizer (model-optimizer model) :mode "test"
                   :input-parameter input-parameter :batch-input batch-input)
             (setq batch-val-losses (append  batch-val-losses (list batch-loss)))
                          (loop for metric in metrics
                                        for i from 0 do
                                            (setf (nth i metrics-val) (list* metric (nth i metrics-val)))))))
	

	;;(save-weights model (format nil "weights-test/~a/" epoch) )	
	;;average errors and print to stdout
	(let ((epoch-train-loss (/ (reduce #'+ batch-train-losses) (length batch-train-losses)))
		  (epoch-val-loss (/ (reduce #'+ batch-val-losses) (length batch-val-losses)))
		  (epoch-train-metrics  (loop for metric in metrics-train collect
											 (/ (reduce #'+ metric) (length metric))))
		  (epoch-val-metrics  (loop for metric in metrics-val collect
											 (/ (reduce #'+ metric) (length metric)))))
			(format t "~Ss train_loss: ~S - val_loss: ~S" (/ (- (get-internal-real-time) time-start) (float INTERNAL-TIME-UNITS-PER-SECOND)) epoch-train-loss epoch-val-loss)
			(when (> (length metrics-train) 0)
                (format t " - train_metrics: ")			 
				(print-list-horizontal	epoch-train-metrics)
				(format t " - val_metrics: ")
				(print-list-horizontal	epoch-val-metrics))					 																 
            (format t "~%")
			(setf epoch-train-losses (append epoch-train-losses (list epoch-train-loss)))
			(setf epoch-val-losses (append epoch-val-losses (list epoch-val-loss)))
			(setf epoch-train-metrices (append epoch-train-metrices (list epoch-train-metrics)))
			(setf epoch-val-metrices (append epoch-val-metrices (list epoch-val-metrics)))
			

			(when (or (= epoch 1) (< (+ epoch-val-loss early-stop-delta) (nth (1- best-epoch) epoch-val-losses))) 
				  (progn 
						(setf best-epoch epoch)
						(setf best-weights (loop for weight in (model-weights model) collect
													(weights-value weight))))))))
	;;set weights of best-epoch when early-stop
	(when (not (null early-stop))
			(loop for weight in (model-weights model) 
				  for best-val in best-weights do
				(setf (weights-value weight) best-val)))
	
	;;return metrices and losses
	(values epoch-train-losses epoch-val-losses epoch-train-metrices epoch-val-metrices)))

(defun train-test (model output-training-data
                   &rest training-data-plist
                   &key loss optimizer (mode "train") input-parameter batch-input &allow-other-keys)
  (setf (model-state-running (model-state model)) t ) 
  (setf (model-state-layer-pointer (model-state model)) (1- (length (model-layers model))))
  (let* ((network (make-network (funcall (output model) model input-parameter)))
		(model-weights (model-weights model))
        (trainable-parameters (loop for parameter in model-weights
									when (trainable parameter)
										collect parameter))
         (output-parameters
           (loop for output in (network-outputs network)
                 collect (make-unknown :shape (lazy-array-shape output)
                                       :element-type *network-precision*)))

         (lossfunc (list (funcall loss (first output-parameters) (first (network-outputs network)) )))
         (metrics (loop for metric in (metrics model) collect
                                                      (funcall metric (first output-parameters) (first (network-outputs network)) )))

         (loss-gradient (list (lazy-reshape 1.0 (~))))
         (gradient (differentiator lossfunc loss-gradient))
         (training-network
           (apply #'make-network
                  (append (loop for trainable-parameter in trainable-parameters
                                collect
                                (funcall gradient (weights trainable-parameter))) lossfunc metrics)))
         (validation-network
           (apply #'make-network (append lossfunc metrics)))
         (n nil)
         (batch-loss 0)
         (metrics-values (loop for i below (list-length metrics) collect 0.0))
         (gradients (loop for i below (list-length trainable-parameters) collect (lazy-reshape 0.0 (~))))) 
  ;; (petalisp.graphviz:view (petalisp.ir:ir-from-lazy-arrays ( network-outputs validation-network)) :filename "val-ir.pdf")
 ;;(petalisp.graphviz:view (petalisp.ir:ir-from-lazy-arrays (network-outputs training-network)) :filename "train-ir.pdf")
    ;;(break "W00t")
    ;; Determine the training data size.
    (dolist (data output-training-data)
      (if (null n)
          (setf n (range-size (first (shape-ranges (array-shape data)))))
          (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    (alexandria:doplist (parameter data training-data-plist) ()
      (unless (symbolp parameter)
        (assert (= n (range-size (first (shape-ranges (array-shape data))))))))
    ;; Assemble the arguments.
    (let ((args '()))
      ;; Inputs.
      (push input-parameter args)
      (push batch-input args)
      ;; Outputs.
      (loop for data in output-training-data
            for output-parameter in output-parameters do
              (push output-parameter args)
              (push data args))
      ;; Parameters.
      (loop for parameter in model-weights do
        (push (weights parameter) args)
        (push (weights-value parameter) args))
      ;; Forward + backward pass
      (if (string-equal mode "train")
          ;;train´
          (let* ((net-out-values (apply #'call-network training-network (reverse args))))
            (loop for i below (list-length trainable-parameters) do
			 (setf (nth i gradients)(nth i net-out-values)))
            (setf batch-loss (compute (nth (list-length trainable-parameters) net-out-values)))
            (loop for i from 1 to (list-length metrics) do
              (setf (nth (- i 1) metrics-values) (compute (nth (+ (list-length trainable-parameters) i) net-out-values))))
			  
			  )
          ;;test
          (let* ((net-out-values (apply #'call-network validation-network (reverse args))))
            (setf batch-loss (compute (first net-out-values)))
            (loop for i from 1 to (list-length metrics) do
              (setf (nth (- i 1) metrics-values) (compute (nth i net-out-values)))))))

  (setf (model-state-running (model-state model)) nil) 
    (when (string-equal mode "train")
      ;;Average gradients
      (loop for i below (length gradients) do
        (setf (nth i gradients) (lazy #'/ (nth i gradients) n)))
      ;;Update weights
	  
      (update-weights optimizer :weights trainable-parameters :gradients gradients))
	 ;;Force garbage collection
	 ;; (trivial-garbage:gc :full t)
    ;; Return the batch loss and metrics.
    (values batch-loss metrics-values)))


(defmethod predict ((model model) input)
   ;;Initialize weights
   (let ((sample-shape (~l (mapcar #'range (cdr (array-dimensions input))))))
        (when (not (model-state-weights-initialized (model-state model)))
			(initialize-weights model sample-shape))
		(setf (model-state-running (model-state model)) t)
		(setf (model-state-layer-pointer (model-state model)) (1- (length (model-layers model))))    
    (let* ((args '())            
              (input-parameter (make-unknown :shape (~s (array-shape input)) :element-type *network-precision*))
              (network (make-network(funcall (output model) model input-parameter))))
				  
        ;; Inputs.
         (push input-parameter args)
         (push input args)
		 ;;(print (array-shape input))
        ;; Trainable parameters.
                (loop for trainable-parameter in (model-weights model) do
                        (push (weights trainable-parameter) args)
                        (push (weights-value trainable-parameter) args))
		
			(let ((prediction (first (apply #'call-network network (reverse args)))))
					(setf (model-state-running (model-state model)) nil)
					prediction))))

(defmethod model-weights-total((model model))
   (reduce #'+ (mapcar #'shape-size(mapcar #'weights-shape (model-weights model)))))

(defmethod model-summary ((model model))
                        (format t "Model summary: ~%Model ~S~%" model)
                        (format t "Layers: ~S ~%"  (length(model-layers model)))
                    (format t "Parameters: ~S~%" (model-weights-total model)))
					
					
(defmethod save-weights ((model model) path)
	(ensure-directories-exist path)
	(let ((weights (model-weights model)))
		(loop for weight in weights 
			  for index from 0 do
			(numpy-file-format:store-array (weights-value weight) (format nil "~a~a.npy" path index)))))
		
	
(defmethod load-weights ((model model) path)

	(let ((weights (model-weights model)))
		(loop for weight in weights 
			  for index from 0 do
			(setf (weights-value weight) (numpy-file-format:load-array  (format nil "~a~a.npy" path index))))))	
     
					
