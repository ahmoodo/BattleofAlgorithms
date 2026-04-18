# BattleofAlgorithms
Bitminers

Connect 4 plan:
Have a functioning AI vs AI program running MiniMax with Alpha-Beta pruning.
Have the code record run time (time it takes to think per move).
Have the code be able to track Player 1/Player 2 Win%
Create a Monte Carlo Tree Search algorithm.
Have the AI run at depths 1, 3, 5, and 7 (code currently runs at 5)
Run a tournament of the each AI depth going against each other.
A single match should be something like running depth 1 vs 3 for 100 or so rounds
Report the wins (Player 1 won 40/100, Player 2 won 60/100).
Report average turn time (Player 1 takes an average of 1.2s to make a move).
Run these matches till we feel like we have satisfactory data.

Objective:
Find at what MiniMax depth does the win percentage vs turn time starts to plateau.
Run MiniMax vs Mante Carlo.
For example, The depth 5 wins 80/100 with average 3s turn time, and depth 7 wins 85/100 with average of 6s turn time.
Only 5 more wins with double the turn time may not be worth it.

Battleship Plan:
Create a basic Reinforced Learning (RL) AI
Have the RL AI train/go against the Standard Population Distribution (SPD) AI

Objective:
Train the RL AI to beat the SPD AI

Run the base Connect 4 code:
1) pip install numpy
2) pip install pygame
3) cd "(insert your file path here)"
4) python3 coonect4_ai_no_comments.py

Run the base Battleship code:
1) cd "(insert your file path here)"
2) python3 -m venv .ven
3) .\.venv\Scripts\activate
4) pip install -r requirements.txt
5) python3 main.py

References:
- https://github.com/JTexpo/Python_Projects/tree/main/Algorithm_AI/SPD_Battleship (Battleship AI)
- https://www.youtube.com/watch?v=LE315dE81Bo&t=1s (Battleship AI demonstration)
- https://github.com/zakuraevs/connect4-ai (Connect 4 AI)


  
