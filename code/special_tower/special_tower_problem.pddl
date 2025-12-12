(define (problem special-tower-blocks)
  (:domain special-tower-blocks-world)
  
  (:objects r1 g1 b1 o1 y1 r2 g2 b2 o2 y2 - block)
  
  (:init
    (clear r1) (clear g1) (clear b1)
    (clear o1) (clear y1)
    (clear r2) (clear g2) (clear b2)
    (clear o2) (clear y2)
    (handempty)
  )

  (:goal (and
    (oriented r1)
    (relative b1 r1)
    (relative_layered b2 r1 b1)
  ))
)