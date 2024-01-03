# A Robust Quantile Huber Loss with Interpretable Parameter Adjustment in Distributional Reinforcement Learning 

## About

The code for the Generalized Quantile Huber Loss function (denoted as GL), along with its second-order Taylor approximation (denoted as GLA), as detailed in the research paper titled "A Robust Quantile Huber Loss with Interpretable Parameter Adjustment in Distributional Reinforcement Learning," accepted for presentation at ICASSP 2024 (). The loss function is specifically crafted for distributional reinforcement learning algorithms and provides a mechanism for tuning threshold parameters in an interpretable manner, thereby improving the robustness of the model.

## Code Structure
```
D4PG-QR, DPG-GL, and D4PG-GLA Codebase
│   run.py - Run D4PG model
|   simulate_env.py - Simulate the environment
└───agent
│   │   agent.py - D4PG agent
│   │   distributional.py - distributional dependency for D4PG
│   │   learning.py - learning module for D4PG-QR, D4PG-GLA, and DPG-GL
│   │   quantile_losses.py - quantile Huber loss and our proposed loss functions (GL and GLA) learning module for D4PG

└───environment
│   │   Environment.py - Trading Environment
│   │   Trading.py - Portfolio constructions
│   │   utils.py - Stochastic Processes generation and other utility functions
```

## Dependencies
```
dm-env==1.5
gym==0.24.1
numpy==1.23.1
pytest==6.2.5
reverb==2.0.1
scipy==1.8.1
torch==1.11.0
tqdm==4.64.0
dm-acme[jax,tensorflow,envs]==0.4.0
dm-sonnet==2.0.0
dm-launchpad==0.5.0
trfl==1.2.0
pyyaml==5.4.1
xmanager==0.2.0
```

## Sample Runs
### Train & Evaluate the Reinforcement Learning Agent
```console
python run.py -spread=0.005 -obj_func=meanstd -train_sim=40000 -eval_sim=5000 -critic=qr -std_coef=1.645 -init_vol=0.3 -mu=0.0 -vov=0.0 -vega_obs=False -gbm=True -hed_ttm=30 -liab_ttms=60 -init_ttm=30 -poisson_rate=1.0 -action_space=0,1 -logger_prefix=batch1/Table1/TC05/RL/mean_std -n_step=5
```

## Credits

* The implementation of D4PG agent is from [ACME [D4PG](https://github.com/deepmind/acme/tree/master/acme/agents/tf/d4pg)]
* The trading environment utilized in this project is from the research paper available at [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4106814)]. For further details regarding the environment's implementation, refer to the corresponding repository [[gamma-vega-rl-hedging](https://github.com/rotmanfinhub/gamma-vega-rl-hedging/tree/main)].
