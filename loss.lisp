
(in-package :common-lisp-user)

(defpackage #:lispnet.loss
  (:use
   #:common-lisp
   #:petalisp)
  (:export
   #:mse
   #:mae
   #:binary-cross-entropy
   ))
   
(in-package #:lispnet.loss)

(defun oneminusx (x) (lazy #'+ 1.0 (lazy #'* x -1.0)))
;; automatic differentiator cant resolve min function. -> max(-1 * f(x))
(defun clip (x minx maxx)
(lazy #'* -1.0 (lazy #'max (* -1 maxx) (lazy #'* -1.0 (lazy #'max minx x))))
)

(defun mse (y-true y-pred)  (lazy #'/ (lazy-allreduce #'+ (lazy #'expt (lazy #'- y-pred y-true) 2)) (shape-size (array-shape y-pred))))

(defun mae (y-true y-pred)  (lazy #'/ (lazy-allreduce #'+ (lazy #'abs (lazy #'- y-pred y-true) ))(shape-size (array-shape y-pred))))

(defun binary-cross-entropy (y-true y-pred)
(let ((y-pred-stable (clip y-pred 0.0000001 0.9999999)))
 (lazy #'* -1.0 (lazy #'/  (lazy-allreduce #'+ (lazy #'+ (lazy #'* y-true (lazy #'log y-pred-stable)) 
											           (lazy #'* (oneminusx y-true) (lazy #'log (oneminusx y-pred-stable)))
			   )) (shape-size (array-shape y-pred)))))
)