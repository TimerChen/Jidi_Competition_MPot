# Wrapper for Ray Policy in MeltingPot

## Necessary Files of Policy

Current loaded policy path of `submission.py` is `agents/rl_agent/pd_policy/`, you can modify this path in function of `init()`.
```
Path tree of agents/rl_agent/pd_policy/:
├── checkpoint_000001   # network parameters of policies
│   └── policies       
│       ├── agent_0
│       │   ├── policy_state.pkl
│       │   └── rllib_checkpoint.json
│       └── agent_1
│           ├── policy_state.pkl
│           └── rllib_checkpoint.json
└── params.json         # hyperparameter of policies
```


The policy can be obtained with ray script in Melting-Pot-Contest-2023.

There is an example of path of saved model. 
`Melting-Pot-Contest-2023/initialization_save_ckpt/torch/pd_matrix/PPO_meltingpot_2f131_00000_0_2024-04-25_16-02-38/`

You need move this to `agents/rl_agent/pd_policy`.

**Caution**: **The code will always load `checkpoint_000001`**, so your should first rename your checkpoint name in it.
You can change the fuction: init() [here](https://github.com/TimerChen/Jidi_Competition_MPot/blob/3786a06970c77ceaea5633a463d8cefdd9c2e5fe/agents/rl_agent/submission.py#L197) to load other checkpoints
