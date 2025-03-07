import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common.utils import GraphCreator, HDF5Dataset
from experiments.models_gnn import MP_PDE_Solver
import argparse
from equations.PDEs import *
import torch.nn.functional as F
import imageio
import os

def visualize_example(args: argparse) -> None:
    """
    Visualize example prediction as well as ground truth (on same graph) for a SINGLE TIMESTEP
    """

    # load dataset
    pde = CE(device=args.device)
    # need train dataset to get tmin, tmax, dt
    train_dataset = HDF5Dataset(path=args.train_dataset_path, pde=pde, mode='train', base_resolution=[250, 100], super_resolution=[250, 100])
    dataset = HDF5Dataset(path=args.dataset_path, pde=pde, mode='test', base_resolution=[250, 100], super_resolution=[250, 100])

    eq_variables = {}

    # load both models
    model_odeint = MP_PDE_Solver(pde=pde,
                          time_window=25,
                          eq_variables=eq_variables,
                          n_time_windows_out=args.num_time_windows_out,
                          use_odeint=True).to(args.device)
    model_odeint.load_state_dict(torch.load(args.model_path_odeint))
    model_autoregressive = MP_PDE_Solver(pde=pde,
                          time_window=25,
                          eq_variables=eq_variables,
                          n_time_windows_out=args.num_time_windows_out,
                          use_odeint=False).to(args.device)
    model_autoregressive.load_state_dict(torch.load(args.model_path_autoregressive))


    # set equation specific parameters
    pde.tmin = train_dataset.tmin
    pde.tmax = train_dataset.tmax
    pde.grid_size = train_dataset.base_resolution
    pde.dt = train_dataset.dt

    graph_creator_odeint = GraphCreator(pde=pde, neighbors=3, time_window=25, t_resolution=250, x_resolution=100, n_time_windows_out=args.num_time_windows_out)
    graph_creator_autoregressive = GraphCreator(pde=pde, neighbors=3, time_window=25, t_resolution=250, x_resolution=100, n_time_windows_out=1)
  
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    batch_size = 16

    model_odeint.eval()
    model_autoregressive.eval()

    first_iter = True
    
    criterion = torch.nn.MSELoss(reduction="sum")

    mse_autoregressive_losses_list = []
    mse_odeint_losses_list = []

    frmse_autoregressive_losses_list = []
    frmse_odeint_losses_list = []

    # initialize variables
    standard_mse_odeint_total = 0.0
    standard_mse_autoregressive_total = 0.0
    num_batches = 0
    nrm = 0.0

    # get a single example from the dataset (is there a better way to do this?)
    for (u_base, u_super, x, variables) in dataloader:
        with torch.no_grad():
            nrm += torch.mean(u_super)
            
            same_steps = [25] * batch_size
            data_odeint, labels_odeint = graph_creator_odeint.create_data(u_super, same_steps)
            data_autoregressive, labels_autoregressive = graph_creator_autoregressive.create_data(u_super, same_steps)

            if first_iter:
                n_nodes = labels_odeint.shape[2]
                time_window_odeint = labels_odeint.shape[1]
                time_window_autoregressive = labels_autoregressive.shape[1]

            graph_odeint = graph_creator_odeint.create_graph(data_odeint, labels_odeint, x, variables, same_steps).to(args.device)
            graph_autoregressive = graph_creator_autoregressive.create_graph(data_autoregressive, labels_autoregressive, x, variables, same_steps).to(args.device)

            
            # get prediction for odeint and reshape
            pred_odeint = model_odeint(graph_odeint)
            pred_odeint_reshaped = pred_odeint.view(data_odeint.shape[0], n_nodes, time_window_odeint).permute(0, 2, 1)
            labels_odeint = labels_odeint.to(args.device)
            mse_losses_tmp = F.mse_loss(pred_odeint_reshaped, labels_odeint, reduction='none')
            mse_losses_tmp = mse_losses_tmp.mean(dim=2)
            if first_iter:
                mse_losses_odeint_one_example = mse_losses_tmp[-1]

                pred_total_autoregressive_one_ex = torch.zeros(time_window_odeint, n_nodes)
                labels_total_autoregressive_one_ex = torch.zeros(time_window_odeint, n_nodes)

                mse_losses_autoregressive = torch.zeros(time_window_autoregressive * args.num_time_windows_out).to(args.device)
                mse_losses_odeint = torch.zeros(time_window_odeint).to(args.device)

                mse_losses_autoregressive_one_example = torch.zeros(time_window_autoregressive * args.num_time_windows_out)

            mse_losses_odeint += mse_losses_tmp.sum(dim=0)
            
            # get prediction for autoregressive and reshape
            pred_autoregressive = model_autoregressive(graph_autoregressive)
            pred_autoregressive_reshaped = pred_autoregressive.view(data_autoregressive.shape[0], n_nodes, time_window_autoregressive).permute(0, 2, 1)
            if first_iter:
                pred_total_autoregressive_one_ex[0:time_window_autoregressive] = pred_autoregressive_reshaped[-1]
                labels_total_autoregressive_one_ex[0:time_window_autoregressive] = labels_autoregressive[-1]
            labels_autoregressive = labels_autoregressive.to(args.device)
            mse_losses_tmp = F.mse_loss(pred_autoregressive_reshaped, labels_autoregressive, reduction='none')
            mse_losses_tmp = mse_losses_tmp.mean(dim=2)
            if first_iter:
                mse_losses_autoregressive_one_example[0:time_window_autoregressive] = mse_losses_tmp[-1]
            mse_losses_autoregressive[0:time_window_autoregressive] += mse_losses_tmp.sum(dim=0)

            mse_losses_autoregressive_list_temp = []
            ar_loss = criterion(pred_autoregressive_reshaped, labels_autoregressive) / 100 # 100: base resolution
            mse_losses_autoregressive_list_temp.append(ar_loss / batch_size)

            odeint_loss = criterion(pred_odeint_reshaped, labels_odeint) / 100 # 100: base resolution
            
            mse_odeint_losses_list.append(torch.sum(odeint_loss / batch_size))
            
            standard_mse_odeint_total += F.mse_loss(pred_odeint_reshaped, labels_odeint).item()

            # Calculate standard MSE for autoregressive (initial window)
            standard_mse_batch = F.mse_loss(pred_autoregressive_reshaped, labels_autoregressive).item()
            standard_mse_autoregressive_total += standard_mse_batch

            for stp in range(25 + graph_creator_autoregressive.tw, 25 +graph_creator_autoregressive.tw * args.num_time_windows_out, graph_creator_autoregressive.tw):
                same_steps = [stp] * batch_size
                _, labels_autoregressive = graph_creator_autoregressive.create_data(u_super, same_steps)
                if first_iter:
                    labels_total_autoregressive_one_ex[stp-graph_creator_autoregressive.tw:stp] = labels_autoregressive[-1]
                graph_autoregressive = graph_creator_autoregressive.create_next_graph(graph_autoregressive, pred_autoregressive, labels_autoregressive, same_steps)
                pred_autoregressive = model_autoregressive(graph_autoregressive)
                pred_autoregressive_reshaped = pred_autoregressive.view(data_autoregressive.shape[0], n_nodes, time_window_autoregressive).permute(0, 2, 1)
                if first_iter:
                    pred_total_autoregressive_one_ex[stp-graph_creator_autoregressive.tw:stp] = pred_autoregressive_reshaped[-1]
                labels_autoregressive = labels_autoregressive.to(args.device)
                mse_losses_tmp = F.mse_loss(pred_autoregressive_reshaped, labels_autoregressive, reduction='none')
                mse_losses_tmp = mse_losses_tmp.mean(dim=2)
                if first_iter:
                    mse_losses_autoregressive_one_example[stp-graph_creator_autoregressive.tw:stp] = mse_losses_tmp[-1]
                mse_losses_autoregressive[stp-graph_creator_autoregressive.tw:stp] += mse_losses_tmp.sum(dim=0)

                ar_loss = criterion(pred_autoregressive_reshaped, labels_autoregressive) / 100 # 100: base resolution
                mse_losses_autoregressive_list_temp.append(ar_loss / batch_size)

                # Add MSE for this window
                standard_mse_batch = F.mse_loss(pred_autoregressive_reshaped, labels_autoregressive).item()
                standard_mse_autoregressive_total += standard_mse_batch
            first_iter = False
            num_batches += 1
        mse_autoregressive_losses_list.append(torch.sum(torch.stack(mse_losses_autoregressive_list_temp)))

    pred_temp_autoregressive = pred_autoregressive[-100:]
    labels_temp_autoregressive = labels_autoregressive[-1]
    pred_temp_odeint = pred_odeint[-100:]
    labels_temp_odeint = labels_odeint[-1]

    # grab last timestep from prediction and labels
    pred_final_autoregressive = pred_temp_autoregressive[:, -1]
    labels_final_autoregressive = labels_temp_autoregressive[-1]
    pred_final_odeint = pred_temp_odeint[:, -1]
    labels_final_odeint = labels_temp_odeint[-1]
    
    nrm = nrm / num_batches
    print("nrm:", nrm)

    mse_losses_ar_stack = torch.stack(mse_autoregressive_losses_list)
    print("autoregressive num params:", sum(p.numel() for p in model_autoregressive.parameters()))
    print("MSE losses autoregressive (unrolled) [mp-pde-solvers criterion]:", torch.mean(mse_losses_ar_stack))
    print("rMSE losses autoregressive (unrolled) [mp-pde-solvers criterion]:", torch.sqrt(torch.mean(mse_losses_ar_stack)))
    print("nrMSE losses autoregressive (unrolled) [mp-pde-solvers criterion]:", torch.sqrt(torch.mean(mse_losses_ar_stack)) / nrm)
    
    mse_losses_odeint_stack = torch.stack(mse_odeint_losses_list)
    print("odeint num params:", sum(p.numel() for p in model_odeint.parameters()))
    print("MSE losses odeint (unrolled) [mp-pde-solvers criterion]:", torch.mean(mse_losses_odeint_stack))
    print("rMSE losses odeint (unrolled) [mp-pde-solvers criterion]:", torch.sqrt(torch.mean(mse_losses_odeint_stack)))
    print("nrMSE losses odeint (unrolled) [mp-pde-solvers criterion]:", torch.sqrt(torch.mean(mse_losses_odeint_stack)) / nrm)

    # perform sanity check that ground truth is the same for both models
    if not torch.allclose(labels_final_autoregressive, labels_final_odeint):
        print("WARNING! Ground truth is not the same for both models")
        return

    os.makedirs('visualizations', exist_ok=True)

    # plot
    print("Now attempting to plot")
    plt.plot(pred_final_autoregressive.cpu().numpy(), label='Prediction (autoregressive)')
    plt.plot(pred_final_odeint.cpu().numpy(), label='Prediction (odeint)')
    plt.plot(labels_final_odeint.cpu().numpy(), label='Ground Truth')
    plt.title('Predictions vs ground truth at step 200')
    plt.legend()
    plt.savefig('visualizations/step200.png')
    plt.close()

    # now plot mse losses one example
    plt.plot(mse_losses_autoregressive_one_example.cpu().numpy(), label='Autoregressive')
    plt.plot(mse_losses_odeint_one_example.cpu().numpy(), label='ODEINT')
    plt.title('MSE over time (one example)')
    plt.legend()
    plt.savefig('visualizations/mse_losses_one_example.png')
    plt.close()

    # now plot mse losses over all examples
    plt.plot(mse_losses_autoregressive.cpu().numpy(), label='Autoregressive')
    plt.plot(mse_losses_odeint.cpu().numpy(), label='ODEINT')
    plt.title('MSE over time (all examples)')
    plt.legend()
    plt.savefig('visualizations/mse_losses_total.png')
    plt.close()

    # create separate plots for autoregressive and odeint
    plt.plot(mse_losses_autoregressive.cpu().numpy())
    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.ylim(0, 0.85)
    plt.savefig('visualizations/mse_losses_autoregressive.png')
    plt.close()

    plt.plot(mse_losses_odeint.cpu().numpy())
    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.ylim(0, 0.85)
    plt.savefig('visualizations/mse_losses_odeint.png')
    plt.close()

    # print average total mse loss for both autoregressive and odeint
    autoregressive_total_mse_loss = mse_losses_autoregressive.mean()
    odeint_total_mse_loss = mse_losses_odeint.mean()
    print(f"Average total test MSE loss (autoregressive) [hierarchical]: {autoregressive_total_mse_loss}")
    print(f"Average total test MSE loss (odeint) [hierarchical]: {odeint_total_mse_loss}")

    autoregressive_total_mse_loss_standard = standard_mse_autoregressive_total / num_batches
    odeint_total_mse_loss_standard = standard_mse_odeint_total / num_batches
    print(f"Average total test MSE loss (autoregressive) [standard]: {autoregressive_total_mse_loss_standard}")
    print(f"Average total test MSE loss (odeint) [standard]: {odeint_total_mse_loss_standard}")

    odeint_data = pred_odeint_reshaped[-1].cpu().numpy()
    print("ODEINT total data min:", pred_odeint_reshaped.min())
    print("ODEINT ex data min:", odeint_data.min())
    labels_odeint_data = labels_odeint[-1].cpu().numpy()


    autoregressive_data = pred_total_autoregressive_one_ex.cpu().numpy()
    labels_autoregressive_data = labels_total_autoregressive_one_ex.cpu().numpy()

    # create dir for total
    os.makedirs('visualizations/total_gif', exist_ok=True)
    os.makedirs('visualizations/odeint_frames', exist_ok=True)
    os.makedirs('visualizations/autoregressive_frames', exist_ok=True)
    os.makedirs('visualizations/ground_truth_frames', exist_ok=True)
    filenames_total = []
    time_stamps = [0, 49, 99, 149, 199]
    for i in range(time_window_odeint):
        # save odeint prediction
        plt.figure(figsize=(8, 4))
        plt.plot(odeint_data[i], label='ODEINT')
        plt.plot(labels_odeint_data[i], label='Ground Truth')
        plt.plot(autoregressive_data[i], label='Autoregressive')
        plt.ylim(-0.2, 1.2)
        plt.ylabel('Values')
        plt.title(f'Predictions vs. Ground Truth at step {i}')
        plt.legend()
        plt.savefig(f'total_gif/step{i}.png')
        filenames_total.append(f'total_gif/step{i}.png')
        plt.close()

        if i in time_stamps:
            # save separate plots for odeint, autoregressive, and ground truth
            plt.figure(figsize=(8,4))
            plt.plot(odeint_data[i])
            plt.ylim(-0.2, 1.2)
            plt.ylabel('Values')
            plt.title(f'NeuralODE prediction at step {i+1}')
            plt.savefig(f'visualizations/odeint_frames/step{i+1}.png')
            plt.close()

            plt.figure(figsize=(8,4))
            plt.plot(autoregressive_data[i])
            plt.ylim(-0.2, 1.2)
            plt.ylabel('Values')
            plt.title(f'Autoregressive prediction at step {i+1}')
            plt.savefig(f'visualizations/autoregressive_frames/step{i+1}.png')
            plt.close()

            plt.figure(figsize=(8,4))
            plt.plot(labels_odeint_data[i])
            plt.ylim(-0.2, 1.2)
            plt.ylabel('Values')
            plt.title(f'Ground Truth at step {i+1}')
            plt.savefig(f'visualizations/ground_truth_frames/step{i+1}.png')
            plt.close()

    # create gif
    imageio.mimsave('visualizations/total_gif.gif', [imageio.imread(f) for f in filenames_total], fps=50)

    
if __name__ == "__main__":
    # get model path, dataset path, autoregressive or not using argparser
    parser = argparse.ArgumentParser("Visualize example prediction")
    parser.add_argument("--model_path_odeint", type=str, default="",help="Path to model")
    parser.add_argument("--model_path_autoregressive", type=str, default="", help="Path to model")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to dataset")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--num_time_windows_out", type=int, default=8, help="Number of time windows out (for autoregressive)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--train_dataset_path", type=str, default="", help="Path to train dataset")
    args = parser.parse_args()

    visualize_example(args)