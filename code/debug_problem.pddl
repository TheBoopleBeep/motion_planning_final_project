(define (problem two-tower-blocks)
  (:domain towers-world)

  (:objects r g b y m c - block)

  (:init
    (on r g)
    (clear r)
    (on g b)
    (ontable b)
    (placedleft b)
    (on y m)
    (clear y)
    (on m c)
    (ontable c)
    (placedright c)
    (handempty)
  )

  (:goal (and
    (on g b)
    (on r g)
    (on m c)
    (on y m)
    (placedleft b)
    (placedright c)
  ))
)
