(define (problem adjacent-blocks-noclear)
  (:domain adjacent-blocks-world-noclear)
  
  (:objects b1 b2 b3 b4 b5 b6 - block)
  
  (:init
    (handempty)
  )
  
  (:goal (and
    (Nneighbor b2 b1)
    (Eneighbor b3 b1)
    (longDiag b4 b1)
    (Wneighbor b5 b4)
    (Sneighbor b6 b4)
  ))
)
