# Analysis of Neural ODE Performance in PDE Sequence Modeling

Paper link: https://openreview.net/forum?id=rnxAKI2kRD 

### Set up conda environment

`source environment.sh`

### Produce dataset
`python generate/generate_data.py --experiment=E1 --train_samples=2048 --valid_samples=128 --test_samples=128 --log=True --device=cuda:0`

###  Train Autoregressive Baseline Model

`python experiments/train.py --device=cuda:0 --experiment=E1 --model=GNN --base_resolution=250,100 --time_window=25 --n_time_windows_out=8 --use_odeint=False --log=True`

### Train Neural ODE Model

`python experiments/train.py --device=cuda:0 --experiment=E1 --model=GNN --base_resolution=250,100 --time_window=25 --n_time_windows_out=8 --use_odeint=True --log=True`

### Generate Visualizations

`python experiments/visualize_example.py --model_path_odeint={path to neural ode model} --model_path_autoregressive={path to autoregressive baseline model} --dataset_path={path to test dataset} --train_dataset_path={path to training dataset} --num_time_windows_out=8 --device=cuda:0`

