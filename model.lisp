(in-package #:lispnet)

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
    :initform '())))


(defmethod model-weights(model)
  (alexandria:flatten
        (loop for layer in (model-layers model) collect
                (loop for weight in (layer-weights layer) collect
                        weight))))

(defmethod model-compile ((model model) &key optimizer loss (metrics '()) &allow-other-keys)
        (setf (model-loss model) loss)
        (setf (model-optimizer model) optimizer)
        (setf (metrics model) metrics)
        (loop for layer in (model-layers model) do
                (layer-compile layer))
        (optimizer-compile optimizer :model model))



(defgeneric forward (model input))

(defmethod fit((model model) train-input-data train-label-data val-input-data val-label-data &key (epochs 10) (batch-size 10))
   (let* ((train-input-data-length (array-dimension train-input-data 0))
                 (train-label-data-length (array-dimension train-label-data 0))
                 (val-input-data-length (array-dimension val-input-data 0))
                 (val-label-data-length (array-dimension val-label-data 0))
                 (sample-shape (~l (mapcar #'range (cdr (array-dimensions train-input-data))))))
    (assert (= train-input-data-length train-label-data-length))
     (assert (= val-input-data-length val-label-data-length))
    (format t "Train on ~d samples~%" train-input-data-length)
    (loop for epoch from 1 to epochs do
    (format t "Epoch ~d/~d~%" epoch epochs)
    (let ((batch-train-losses '())
                  (batch-val-losses '())
                  (metrics-train (loop for i below (list-length (metrics model)) collect '()))
                  (metrics-val (loop for i below (list-length (metrics model)) collect '()))
                  (time-start (get-internal-real-time)))
    ;;Training
    (loop for offset below train-input-data-length by batch-size
                  for batch from 0 do
                 ;; (format  t "Batch: ~S~%" batch)
          (let* ((batch-range (range offset (min train-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices train-input-data batch-range))
                 (batch-labels (lazy-slices train-label-data batch-range))
                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                                 (input-parameter (make-unknown :shape (~ (range-size batch-range) ~s sample-shape) :element-type 'single-float)))
            (multiple-value-bind (batch-loss metrics)
           (train-test model (list batch-output)
                   :loss (model-loss model) :optimizer (model-optimizer model) :mode "train"
                   :input-parameter input-parameter :batch-input batch-input)
              (setq batch-train-losses (append batch-train-losses (list batch-loss)))
                          (loop for metric in metrics
                                        for i from 0 do
                                            (setf (nth i metrics-train) (list* metric (nth i metrics-train)))))))
        ;;Validation
        (loop for offset below val-input-data-length by batch-size
                  for batch from 0 do
                                ;;  (format  t "Batch: ~S~%" batch)
          (let* ((batch-range (range offset (min val-input-data-length (+ offset batch-size))))
                 (batch-data (lazy-slices val-input-data batch-range))
                 (batch-labels (lazy-slices val-label-data batch-range))
                 (batch-input (compute (lazy-collapse batch-data)))
                 (batch-output (compute (lazy-collapse batch-labels)))
                                 (input-parameter (make-unknown :shape (~ (range-size batch-range) ~s sample-shape) :element-type 'single-float)))

            (multiple-value-bind (batch-loss metrics)
            (train-test model (list batch-output)
                   :loss (model-loss model) :optimizer (model-optimizer model) :mode "test"
                   :input-parameter input-parameter :batch-input batch-input)
             (setq batch-val-losses (append  batch-val-losses (list batch-loss)))
                          (loop for metric in metrics
                                        for i from 0 do
                                            (setf (nth i metrics-val) (list* metric (nth i metrics-val)))))))

      (format t "~Ss train_loss: ~S - val_loss: ~S" (/ (- (get-internal-real-time) time-start) 1000.0)
                                                                                                        (/ (reduce #'+ batch-train-losses) (length batch-train-losses))
                                                                                                        (/ (reduce #'+ batch-val-losses) (length batch-val-losses)))
         (when (> (length metrics-train) 0)
                (format t " - train_metrics: ")			 
				(print-list-horizontal	(loop for metric in metrics-train collect
											 (/ (reduce #'+ metric) (length metric))))
				(format t " - val_metrics: ")
				(print-list-horizontal	(loop for metric in metrics-val collect
											 (/ (reduce #'+ metric) (length metric)))))					 																 
                (format t "~%")))))

(defun train-test (model output-training-data
                   &rest training-data-plist
                   &key loss optimizer (mode "train") input-parameter batch-input &allow-other-keys)
  (let* ((network (make-network(forward model input-parameter)))
         (trainable-parameters
           (model-weights model))
         (output-parameters
           (loop for output in (network-outputs network)
                 collect (make-unknown :shape (lazy-array-shape output)
                                       :element-type 'single-float)))

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
    ;;(petalisp.graphviz:view (network-outputs validation-network))
    ;;(petalisp.graphviz:view (network-outputs training-network))
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
      ;; Trainable parameters.
      (loop for trainable-parameter in trainable-parameters do
        (push (weights trainable-parameter) args)
        (push (weights-value trainable-parameter) args))
      ;; Forward + backward pass
      (if (string-equal mode "train")
          ;;train´
          (let* ((net-out-values (apply #'call-network training-network (reverse args))))
            (loop for i below (list-length trainable-parameters) do
              (setf (nth i gradients)(nth i net-out-values)))
            (setf batch-loss (compute (nth (list-length trainable-parameters) net-out-values)))
            (loop for i from 1 to (list-length metrics) do
              (setf (nth (- i 1) metrics-values) (compute (nth (+ (list-length trainable-parameters) i) net-out-values)))))
          ;;test
          (let* ((net-out-values (apply #'call-network validation-network (reverse args))))
            (setf batch-loss (compute (first net-out-values)))
            (loop for i from 1 to (list-length metrics) do
              (setf (nth (- i 1) metrics-values) (compute (nth i net-out-values)))))))

    (when (string-equal mode "train")
      ;;Average gradients
      (loop for i below (length gradients) do
        (setf (nth i gradients) (lazy #'/ (nth i gradients) n)))
      ;;Update weights
      (update-weights optimizer :weights trainable-parameters :gradients gradients))
    ;; Return the batch loss and metrics.
    (values batch-loss metrics-values)))


(defmethod predict ((model model) input)
    (let* ((args '())
              (sample-shape (~l (mapcar #'range(array-dimensions input))))
              (input-parameter (make-unknown :shape (~s sample-shape) :element-type 'single-float))
                  (network (make-network(forward model input-parameter))))
        ;; Inputs.
         (push input-parameter args)
        (push (lazy-reshape input (~s sample-shape)) args)

        ;; Trainable parameters.
                (loop for trainable-parameter in (model-weights model) do
                        (push (weights trainable-parameter) args)
                        (push (weights-value trainable-parameter) args))
                (compute 
                                 (lazy-array
                                          (values-list (apply #'call-network network (reverse args)))))))

(defmethod model-weights-total((model model))
   (reduce #'+ (mapcar #'shape-size(mapcar #'weights-shape (model-weights model)))))

(defmethod model-summary ((model model))
                        (format t "Model summary: ~%Model ~S~%" model)
                        (format t "Layers: ~S ~%"  (length(model-layers model)))
                    (format t "Trainable parameters: ~S~%" (model-weights-total model)))
