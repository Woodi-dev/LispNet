(in-package #:lispnet)
(in-package #:lispnet.examples.vcycle)


(defun load-array (path)
  (numpy-file-format:load-array
   (asdf:system-relative-pathname (asdf:find-system "lispnet.examples") path)))
   

(defparameter *train-data* (load-array "vcycle/data/train.npy"))
(defparameter *val-data* (load-array "vcycle/data/val.npy"))
(defparameter *train2-data* (load-array "vcycle/data/train2.npy"))
(defparameter *val2-data* (load-array "vcycle/data/val2.npy"))

(defparameter *test-data* (load-array "vcycle/data/test.npy"))
(defparameter *test2-data* (load-array "vcycle/data/test2.npy"))

(defparameter *jump-data* (load-array "vcycle/data/jump.npy"))
(defparameter *jumpexp-data* (load-array "vcycle/data/jumpexp.npy"))
(defparameter *jumpcount-data* (load-array "vcycle/data/jumpcount.npy"))



(defun genLabels(data)
	(let* ((dimensions (shape-dimensions (array-shape data))))
		   (compute (lazy-reshape 0.0 (~ (nth 0 dimensions) ~ (nth 1 dimensions ) ~ (nth 2 dimensions))))))
(defun genLabels2(data)
	(let* ((dimensions (shape-dimensions (array-shape data))))
		   (compute (lazy-reshape 0.0 (~ (nth 0 dimensions) )))))		   
  
(defun stackInput(v f c)
	(lazy-stack 3 (lazy-reshape v (transform b y x to  b y x 0))
				  (lazy-reshape f (transform b y x to  b y x 0))
				  (lazy-reshape c (transform b y x to  b y x 0))))
   

	
    

(defun convergence (model input err )
  (let* ((residuum-model (make-instance 'vcycle-model :output #'forward-residuum))
		 (f (lazy-drop-axes (lazy-slices input (range 1 2) 3) 3))
		 (c (lazy-drop-axes (lazy-slices input (range 2 3) 3) 3))
		 (r0 (predict residuum-model input))
		 (r-old r0)
		 (r-new r0)
		 (q 0.0)
		 (i 1))
		
		
		
			(loop while (or (= i 1) (and (< r-new 1d60) (< i 50) (>= r-new r-old)) (and (< r-new r-old)  (>= (/ r-new r0) err) (< i 250))) do ;; 
				(setf r-old r-new)
				(setq input (compute (stackInput (predict model input) f c)))
				(setf r-new  (predict residuum-model input))
				(incf q (/ r-new r-old))
				(format t "~S ~S ~S~%" i r-new (/ q i))
				
					(incf i))
	(format t "iterations: ~S convergence: ~S~%" i (/ q i))))
	;;	(print (compute (lazy-drop-axes (lazy-slices input (range 0 1) 3) 3)))))
		;;   (if (<= r-new r-old)
			;;	(format t "iterations: ~S convergence: ~S~%" i (/ q i))
				;;(format t "algorithm diverges - iterations: ~S convergence: ~S~%" i (/ q i)))))

(defun vcycle-main()
  (setf *network-precision* 'double-float)
  (setf *random-state* (make-random-state t))
  (let*  ((att-train-model (make-instance 'attention-vcycle-model :output #'forward-train))
		  (att-calc-model (make-instance 'attention-vcycle-model :output #'forward-calc))
		  (vcycle-train-model (make-instance 'vcycle-model :output #'forward-train))
		  (vcycle-calc-model (make-instance 'vcycle-model :output #'forward-calc))
		  (gmg (make-instance 'vcycle-model :output #'forward-calc))
		  (data (compute (lazy-slices *train-data* (range 0 1)))))

	(predict att-calc-model (compute (lazy-slices *train-data* (range 0 1))))
	(predict vcycle-calc-model (compute (lazy-slices *train-data* (range 0 1))))
;;	(model-summary att-calc-model)
	 (model-compile att-train-model :loss #'output-loss :optimizer (make-adam :learning-rate 0.001))
	 (model-compile vcycle-train-model :loss #'output-loss :optimizer (make-adam :learning-rate 0.001))

	;;fit attention dmg


	;;(fit att-train-model *train-data* (genLabels2 *train-data*) *val-data* (genLabels2 *val-data*) :epochs 250 :batch-size 1 );; :early-stop-delta 1000.0 :early-stop 10	
	;;(save-weights att-train-model "weights-test/")


	;;fit dmg
	;;(fit vcycle-train-model *train-data* (genLabels *train-data*) *val-data* (genLabels *val-data*)  :epochs 250 :batch-size 20 :early-stop 10 :early-stop-delta 1000.0)
	;;(save-weights vcycle-train-model "weights-vcycle/")

	 
	(format t "------ compute convergence of deep multi grid ------~%")
	;;(load-weights vcycle-calc-model  "weights-vcycle/5/")
	;;(convergence vcycle-calc-model (compute (lazy-slices *test2-data* (range 0 225 9))) 1e-10)
	
	
	(format t "------ compute convergence of deep attention multi grid ------~%")

	(loop for i from 1 to 250 do
	;;	(format t "~S " i)
	;;(load-weights att-calc-model  (format nil "weights-att_5its_1e-3_3/~a/" i))	
	(load-weights att-calc-model  (format nil "weights-test/~a/" i))	
	;;	(loop for j from 0 to 6 do
		(convergence att-calc-model  (compute (lazy-slices *test2-data* (range 8 225 9))) 1e-10)) ;;
	;;(load-weights att-calc-model "weights-att/74/" )	
	;;(loop for i from 0 to 6 do
		;;(convergence att-calc-model (compute (lazy-slices *test-data* (range i 350 7))) 1e-10))
	
	
	(format t "------ compute convergence of geometric multi grid ------~%")
;;	(convergence gmg (compute (lazy-slices *test2-data* (range 8 225 9))) 1e-10)
	
	
	(format t "--------- jump test ----------~%")
	(format t "--------------  deep multi grid --------~%")
	;;(load-weights vcycle-calc-model  "weights-vcycle/5/")
		;;(loop for i from 0 below 150 do
		;;		(format t "~S " i)
			;;	(convergence vcycle-calc-model (compute (lazy-slices *jump-data* (range i (1+ i)))) 1e-10))
	(format t "--------------  attention deep multi grid --------~%")
;;	(load-weights att-calc-model "weights-att_5its_1e-3_3/100/" )	
	;;	(loop for i from 0 below 150 do
		;;		(format t "~S " i)
			;;	(convergence att-calc-model (compute (lazy-slices *jump-data* (range i (1+ i)))) 1e-10))
	(format t "--------------  geometric multi grid --------~%")
	
	;;	(loop for i from 0 below 150 do
	;;		(format t "~S " i)
	;;			(convergence gmg (compute (lazy-slices *jump-data* (range i (1+ i)))) 1e-10))
	
	(format t "-------------- testing jump count for attention dmg ----~%")
	;;(load-weights att-calc-model "weights-multi_5its_1e-3/250/" )	
		;;(loop for i from 0 below 33 do
		;;		(format t "~S " i)
		;;		(convergence att-calc-model (compute (lazy-slices *jumpcount-data* (range i (1+ i)))) 1e-10))
				
	(format t "-------------- testing jump exp height for attention dmg ----~%")
	;;(load-weights att-calc-model "weights-att_5its_1e-3_3/100/" )	
	;;	(loop for i from 200 below 201 do
		;;		(format t "~S " i)
			;;	(convergence att-calc-model (compute (lazy-slices *jumpexp-data* (range i (1+ i)))) 1e-10))			

	))


;;(tes)
(vcycle-main)