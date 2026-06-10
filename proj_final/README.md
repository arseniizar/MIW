# Final Project — Prolog Text Adventure

A logic programming text adventure implemented in Prolog. It features a spatial representation of an apartment, dynamic state management, recursive list evaluation, and pathfinding.

## Usage

You will need a Prolog interpreter such as [SWI-Prolog](https://www.swi-prolog.org/) to run this script. 

Load the script in the interactive SWI-Prolog console:

```bash
swipl -s proj_final.pl
```

Once loaded, you can test the logic using the main scenario query:

```prolog
?- agent_at(Where1), inventory(List), has_item(phone, List), goto(kitchen), goto(living_room), agent_at(Where2), watch_tv.
```

*(Note: Make sure to uncomment `:- dynamic agent_at/1.` in your code so the agent's position can be updated at runtime).*

## Features

| Feature | Notes |
|-------|-------|
| Logic Representation | Spatial layout mapped via `connected/2` and `item_in/2` facts |
| Dynamic State | Agent location is modified during runtime using `asserta` and `retract` |
| Recursion | Custom `has_item/2` for inventory lists and `path/2` for spatial pathfinding |
| Action Rules | `goto/1` manages movement constraints, state updates, and console feedback |
| Goal Chaining | `watch_tv/0` automatically resolves conditions (finding the TV) and executes actions |