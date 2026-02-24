# Code Implementation Prompt

## Orient
implement ilqr.py by referring to iterative-linear-quadratic-regulator.tex, linear-quadratic-regulator.tex and lqr.py.

## Principle
* I don't want you to be too creative. I need you do basic coding work for me, so that you can help me achieve it step by step.
* Don't change the given code. Add code at the comment "# Implement Here by Agent"
* Write the python code from top to bottom level
* Stop and let me check when each step is done

## Steps
* Activate working environment by executing "conda activate py310" command in terminal
* Implement the iteration loop which is the top level.
* Leave the solving algorithm alone assuming all Ks is solved. Write forward pass code
* Implement backward pass. Assume all V-value-coefficient is solved, keep only the top level part of solve $K_t$ given $P_{t+1}$
* Complete it by implementing the solution of $p_t$ and $P_t$"
* Test if the output is equal to the output of lqr.py
* Log my orignal prompt in prompt.md file. From the first to the latest, no missing and modification.