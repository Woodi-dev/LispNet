
(in-package :common-lisp-user)

(defpackage #:lispnet.layers
  (:use
   #:common-lisp
   #:petalisp
   #:lispnet.initializers
   #:lispnet.trainable-parameter
   #:lispnet.utils
   )
  (:export
   #:softmax
   #:relu
   #:sigmoid
   #:fcn
   #:conv-2d
   #:flatten
   ))

(in-package #:lispnet.layers)

(define-modify-macro minf (&rest numbers) min)
(define-modify-macro maxf (&rest numbers) max)


(defun lazy-multi-stack (axis arrays)
  (if (> 2 (length arrays)) (lazy-reshape (first arrays) (~s (~ 1) ~s (array-shape (first arrays))))
  (let ((combined-array (lazy-stack axis (first arrays) (second arrays))))
      (loop for index from 2 to (- (length arrays) 1) do
          (setq combined-array (lazy-stack axis combined-array (nth index arrays)))
  )
    combined-array
  )
))

(defun sigmoid(input)
	(lazy #'/ 1.0 (lazy #'+ 1.0 (lazy #'exp (lazy #'- input)))))
  
(defun softmax (input)
  (let* ((c (lazy-allreduce #'max input))
		(totals (lazy #'exp (lazy #'- input c))))		
    (lazy #'/ totals (lazy-allreduce #'+ totals))))

(defun relu (input)
  (lazy #'max (coerce 0 (element-type input)) input))

(defun fcn (input &key units (activation nil))
  (let* ((n (total-size input))
         (weights
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array (list units n) :element-type (element-type input))
                  (* units n))))
         (bias
           (make-trainable-parameter
            (lazy #'/
                  (make-random-array units :element-type (element-type input))
                  units)))

         )
        
    (lazy-reduce
           #'+
           (lazy #'*
                 (lazy-reshape weights (transform A B to B A))
                  (lazy-reshape (lazy-flatten input ) (transform n to n 0) )))		
    ;; comment out equation above and use the code below to test the forward and backward pass
    #|(let* ((result
			(lazy #'+ bias
			(lazy-collapse
			(lazy-multi-stack 0
			(loop for i from 0 below units collect
				(lazy-reshape
					(lazy-reduce #'+                 
                        (lazy #'*
                              (lazy-slice weights i)
                              input
                              ))
            (~ 1))))))))
	  (if activation (funcall activation result)
	   result)
	  
      )|#
  
  ))


(defun flatten (input)
	(lazy-collapse (lazy-flatten input)))

 ;; Conv2D input_dim:[channel,height,width]
 (defun conv-2d (input &key (n-filters 1) (kernel-size 3) (strides '(1 1)) (padding "valid") (stencil '()) (activation nil))
  (let* ((rank (rank input))
         (n-weights (length stencil))
         (lower-bounds (make-array 2 :initial-element 0))
         (upper-bounds (make-array 2 :initial-element 0))
		 (n-filters-input (first(shape-dimensions(array-shape input))))
		 (input-pad input)
         )
	;; Generate stencil if not set
	(when (= (length stencil) 0)
	(setq stencil (make-2d-kernel kernel-size))
	(setf n-weights (length stencil)))
		
    (loop for offsets in stencil do
              (assert (= (length offsets) 2)))
	
    ;; Determine the bounding box of the stencil.
    (loop for offsets in stencil do
          (loop for offset in offsets
                for index from 0 do
                (minf (aref lower-bounds index) offset)
                (maxf (aref upper-bounds index) offset)))

	;; Padding
	(when (string-equal padding "same")
	(setq input-pad (pad input :paddings (append '((0 0)) (loop for lb across lower-bounds
								 for ub across upper-bounds collect
								 (list (abs lb) ub))))))

	
    ;; Use the bounding box to compute the shape of the result.
    (let* ((interior-shape
            (~l 
               (loop for lb across lower-bounds
                     for ub across upper-bounds
                     for range in (cdr(shape-ranges (array-shape input-pad)))				
                     collect
                     (if (and (integerp lb)
                              (integerp ub))
                         (let ((lo (- (range-start range) lb))
                               (hi (- (range-end range) ub)))
                           (assert (< lo hi))
                           (range lo hi))
                         range))))
          (filters
            (make-trainable-parameter
             (make-random-array
              (list n-filters
			        n-weights      
                    n-filters-input)
              :element-type (element-type input-pad)))))
				
      ;; Compute the result.
	  (let ((result
	  (lazy-collapse
	  (lazy-reshape
	  (lazy-multi-stack 0
	  (loop for filter-index below n-filters
				collect 				
				(lazy-reshape 				
				(lazy-reduce #'+
				(apply #'lazy #'+
					(loop for offsets in stencil
						  for offset-index from 0
						  collect
						  (lazy #'*
								(lazy-reshape (lazy-slice (lazy-slice filters filter-index) offset-index) (transform A to A 0 0))
								(lazy-reshape input-pad
									(make-transformation
									:offsets
									(cons 0 (mapcar #'- offsets)))
									(~r (range n-filters-input) ~s interior-shape)
									))))) (transform A B to 0 A B )))) 
	    (~r (range n-filters) ~s (stride-shape interior-shape strides))))))
		(if activation (funcall activation result)
		result))
		)))
								

						   
