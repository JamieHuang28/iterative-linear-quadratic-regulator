# Implement iLQR for Vehicle Model

## Orient
implement ilqr_vehicle_model by referring to vehicle_model.tex, linear-quadratic-regulator.tex and ilqr.py.

## Principle
* Write the python code from top to bottom level, if dependency is not given, leave as comment.
* Stop and let me check when each step is done

## Parameters
* L = 2.8
* delta_max = 0.5, delta_min = -0.5 (for trajectory test case generation)
* dot_delta_max = 0.25 (for cost tuning)

## Steps
* Activate working environment by executing "conda activate py310" command in terminal
* Implement the discrete version of vehicle model and test by generating a circular trajectory with input. plot and show.
* Implement the diff-dynamics version of vehicle model and derivate related matrix for the use of ilqr.
* Implement cost penalty considering dot_delta and velocity.
* Implement the ilqr controlled vehicle model, and test with infinite-symbole-shaped trajectory tracking.
