(define (domain special-tower-blocks-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (on ?x - block ?y - block)  ; block x is on block y
    (clear ?x - block)  ; block x has nothing on top
    (ontable ?x - block)  ; block x is on the table
    (holding ?x - block)  ; robot is holding block x
    (handempty) ; robot hand is empty
    (inplace ?x - block) ; block x is in the right place
    (oriented ?x - block) ; block x is in the right place and oriented
    (relative ?x - block ?y - block) ; block x is placed relative to block y in special tower formation (clockwise)
    (relative_layered ?x - block ?y - block ?z - block) ; block x is on the second layer relative to y and stacked between y and z
  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty)) ; block x is clear and on table and hand is empty
    :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x)) (not (handempty))) ; x is held and not on table and hand is not empty
  )
  
  (:action place
    :parameters (?x - block)
    :precondition (holding ?x)  ; holding x
    :effect (and (clear ?x) (ontable ?x) (handempty) (not (holding ?x)))  ; x is clear and on table and not held
  )
  
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty)) ; x on y and x is clear and head is empty
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty))) ; x is held and not on y and y is clear and hand is not empty
  )
  
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y)) ; x is held and y is clear
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))) ; x is on y and y is not clear and x is no longer held and x is clear
  )

  (:action place_specific
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (clear ?x) (inplace ?x) (handempty) (not (holding ?x)))  ; x is clear and in place and not held
  )

  (:action place_specific_oriented
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (clear ?x) (oriented ?x) (inplace ?x) (handempty) (not (holding ?x)))  ; x is clear, oriented right in the correct position and not held
  )

  (:action place_relative_oriented
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x ) (inplace ?y))
    :effect (and (clear ?x) (relative ?x ?y) (inplace ?x) (handempty) (not (holding ?x)))  ; x is placed relative to y, is clear, in the correct position, and not held
  )

  (:action place_stacked_relative_oriented
    :parameters (?x - block ?y - block ?z - block)
    :precondition (and (holding ?x) (inplace ?y) (inplace ?z))
    :effect (and (clear ?x) (inplace ?x) (handempty) (relative_layered ?x ?y ?z) (not (holding ?x)))  ; x is placed relative to y, is clear, in the correct position, and not held
  )

)