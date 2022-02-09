(defsystem "lispnet.examples"
  :description "Lispnet examples"
  :author "Michael Holzmann <michael.holzmann@fau.de>"

  :depends-on
  ("lispnet")


  :serial t
  :components
  ((:file "vcycle/packages")
  (:file "vcycle/restriction")
  (:file "vcycle/prolongation")
  (:file "vcycle/jacobi")
  (:file "vcycle/rbgs")
  (:file "vcycle/residual")
  (:file "vcycle/vcycle-model")
  (:file "vcycle/att-restriction")
  (:file "vcycle/att-prolongation")
  (:file "vcycle/attention-vcycle-model")
  (:file "vcycle/vcycle")))
