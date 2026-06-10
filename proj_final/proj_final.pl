% agent_at(Where1), inventory(List), has_item(phone, List), goto(kitchen), goto(living_room), agent_at(Where2), watch_tv.

% allow changing player position
:- dynamic agent_at/1.

% starting point
agent_at(corridor).

% apartment layout
connected(corridor, living_room).
connected(living_room, kitchen).
connected(living_room, bedroom).

% objects in rooms
item_in(tv, living_room).
item_in(sofa, living_room).
item_in(fridge, kitchen).
item_in(bed, bedroom).

% spatial relations
next_to(sofa, tv).
on(remote, sofa).
between(corridor, kitchen, living_room).

% bidirectional movement
can_move(X, Y) :- connected(X, Y).
can_move(X, Y) :- connected(Y, X).

% player inventory (list)
inventory([keys, phone, wallet]).

% check if item is in list (recursive)
has_item(Item, [Item|_]).
has_item(Item, [_|Tail]) :- has_item(Item, Tail).

% recursive path finding
path(A, B) :- can_move(A, B).
path(A, B) :- can_move(A, C), path(C, B).

% navigation logic: 

% already there
goto(Room) :-
    agent_at(Room), !, 
    write('You are already in the '), write(Room), nl.

% move and update state
goto(Room) :-
    agent_at(Current),
    can_move(Current, Room),
    retract(agent_at(Current)), 
    asserta(agent_at(Room)),    
    write('You moved from '), write(Current), write(' to the '), write(Room), nl, !. 

% cannot move directly
goto(Room) :-
    write('You cannot go directly to the '), write(Room), write(' from here.'), nl.

% main task: find TV and watch it
watch_tv :-
    item_in(tv, TargetRoom),    
    goto(TargetRoom),           
    next_to(sofa, tv),          
    write('You sit on the sofa next to the TV. You are now watching TV.'), nl.