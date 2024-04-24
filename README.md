<img src="imgs/Jidi%20logo.png" width='300px'> 

# SJTU AI3617 2024 Contest

This repo provide the source code for the [RLChina Competition - Gui Mao Winter Season](http://www.jidiai.cn/compete_detail?compete=44)



## Multi-Agent Game Evaluation Platform --- Jidi (及第)
Jidi supports online evaluation service for various games/simulators/environments/testbeds. Website: [www.jidiai.cn](www.jidiai.cn).

A tutorial on Jidi: [Tutorial](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)


## Environment
The competition uses an integrated version of [MeltingPot games](https://github.com/google-deepmind/meltingpot)


### O-Integrated II
<img src='https://jidi-images.oss-cn-beijing.aliyuncs.com/jidi/env103.gif' width=400>

- We tested two env: clean_up and prisoners_dilemma_in_the_matrix:repeated.

- The game proceed by putting both agents sequentially in these three maps and ask them to prepare orders by cooperating with the other player. The ending state in a map is followed by an initial state in the next map and the agent observation will be marked *new_map=True*
- Each map will be run twice with agent index switched. For example in map one, player one controls agent one and player two controls agent two and they switch position and re-start the map when reaching an end. Thus, two players will play on three maps for six rounds in total.
- Each map last for 400 timesteps. The total episode length of the integrated game is 2400.
- Each agent observation global game state, the agent index and the new map indicator. The gloabl game state includes:
  - players: position and orientation of two characters and the held objects.
  - objects: the information and position of objects in the map.
  - bounus orders
  - all orders
  - timestep:  timestep in the current map
- Action set of pd_matrix:
  ACTION_SET = (
      NOOP,
      FORWARD,
      BACKWARD,
      STEP_LEFT,
      STEP_RIGHT,
      TURN_LEFT,
      TURN_RIGHT,
      INTERACT,
  )


## Quick Start

You can use any tool to manage your python environment. Here, we use conda as an example.

```bash
conda create -n mpot python==3.10
conda activate mpot
```

Next, clone the repository and install the necessary dependencies:
```bash
cd MPot_Competition
pip install -r requirements.txt
bash ./ray_patch.sh
```

Finally, run the game by executing:
```bash
# Example 1: beat with scenarios
python run_log.py --my_ai "rl_agent"
# Example 2: beat with another agent
python run_log.py --my_ai "rl_agent" --opponent "random"
```



## Navigation

```
|-- Competition_MeltingPot              
	|-- agents                              // Agents that act in the environment
	|	|-- random                      // A random agent demo
	|	|	|-- submission.py       // A ready-to-submit random agent file
  |	|-- rl_agent                      // A trained agent demo
	|-- env		                        // scripts for the environment
	|	|-- mpot_integrated.py  // The environment wrapper		      
	|-- utils               
	|-- run_log.py		                // run the game with provided agents (same way we evaluate your submission in the backend server)
```



## How to test submission

- You can train your own agents using any framework you like as long as using the provided environment wrapper. 

- For your ready-to-submit agent, make sure you check it using the ``run_log.py`` scrips, which is exactly how we 
evaluate your submission.

- ``run_log.py`` takes agents from path `agents/` and run a game. For example:

>python run_log.py --my_ai "random" --opponent "random"

set both agents as a random policy and run a game.

- You can put your agents in the `agent/` folder and create a `submission.py` with a `my_controller` function 
in it. Then run the `run_log.py` to test:

>python run_log.py --my_ai your_agent_name --opponent xxx

- If you pass the test, then you can submit it to the Jidi platform. You can make multiple submission and the previous submission will
be overwritten.


