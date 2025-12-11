(define (domain towers-world)
  (:requirements :strips :typing)
  
  (:types block)
  
  (:predicates
    (on ?x - block ?y - block)  ; block x is on block y
    (clear ?x - block)  ; block x has nothing on top
    (ontable ?x - block)  ; block x is on the table
    (holding ?x - block)  ; robot is holding block x
    (placedleft ?x - block); block is in right position
    (placedright ?x - block); block is in right position
    (handempty) ; robot hand is empty
  )
  
  (:action pick
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty)) ; block x is clear and on table and hand is empty
    :effect (and (holding ?x) (not (clear ?x)) (not (ontable ?x)) (not (handempty))) ; x is held and not on table and hand is not empty
  )
  
  (:action placeLeft
    :parameters (?x - block)
    :precondition (holding ?x)  ; holding x
    :effect (and (clear ?x) (ontable ?x) (handempty) (not (holding ?x)) (placedleft ?x))  ; x is clear and on table and not held and in leftpos
  )

  (:action placeRight
    :parameters (?x - block)
    :precondition (holding ?x)  ; holding x
    :effect (and (clear ?x) (ontable ?x) (handempty) (not (holding ?x)) (placedright ?x))  ; x is clear and on table and not held and in rightpos
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
)