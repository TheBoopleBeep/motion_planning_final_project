(define (domain special-tower-blocks-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (on ?x - block ?y - block)  ; block x is on block y
    (clear ?x - block)  ; block x has nothing on top
    (ontable ?x - block)  ; block x is on the table
    (holding ?x - block)  ; robot is holding block x
    (handempty) ; robot hand is empty
    (placed ?x - block) ; block x is in the right place
    (oriented ?x - block) ; block x is oriented correctly
    (relative ?x - block ?y - block) ; block x is placed relative to block y in special tower formation (clockwise)
    (relative_layered ?x - block ?y - block ?z - block) ; block x is on the second layer relative to y and stacked between y and z
  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (clear ?x) (handempty)) ; block x is clear and on table and hand is empty
    :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x)) (not (handempty))) ; x is held and not on table and hand is not empty
  )

  (:action place_specific_oriented
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (clear ?x) (oriented ?x) (placed ?x) (handempty) (not (holding ?x)))  ; x is clear, oriented right in the correct position and not held
  )

  (:action place_relative_oriented
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x ) (placed ?y) (oriented ?y))
    :effect (and (clear ?x) (relative ?x ?y) (placed ?x) (oriented ?x) (handempty) (not (holding ?x)))  ; x is placed relative to y, is clear, in the correct position, and not held
  )

  (:action stack_relative_oriented
    :parameters (?x - block ?y - block ?z - block)
    :precondition (and (holding ?x) (placed ?y) (placed ?z) (oriented ?y) (oriented ?z))
    :effect (and (clear ?x) (placed ?x) (oriented ?x) (handempty) (relative_layered ?x ?y ?z) (not (holding ?x)))  ; x is placed relative to y, is clear, in the correct position, and not held
  )

)