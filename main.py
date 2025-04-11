import os
import random
import time
import numpy as np
import torch
from ogb.graphproppred import Evaluator
from tqdm import tqdm
import torch.nn as nn
from config import gene_arg, set_config

import warnings
import torch.nn.functional as F
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

from criterion.weight_loss import weighted_cross_entropy, accuracy_sbm
from data_utils.load_data import load_data
from model.Gradformer import Gradformer
from optimizer.ultra_optimizer import get_scheduler


# 添加地理距离计算函数
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


# 添加中位数计算函数
def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2


def train(loader, model, criterion, optimizer, device, task):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        node_num = data.sph.shape[-1]
        data.sph = data.sph.reshape(-1, node_num, node_num)

        optimizer.zero_grad()
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros(data.edge_index.shape[1])

        if task == "geo_regression":
            geo_out, _ = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)

            lat_pred = geo_out[:, 0]
            lon_pred = geo_out[:, 1]

            train_node_ids = data.train_node_ids
            lat_pred_train = lat_pred[train_node_ids]
            lon_pred_train = lon_pred[train_node_ids]

            y_lat = data.norm_y_lat.squeeze(-1)
            y_lon = data.norm_y_lon.squeeze(-1)

            assert len(train_node_ids) == len(y_lat), \
                f"train_node_ids length {len(train_node_ids)} does not match y_lat length {len(y_lat)}"

            lat_loss = F.mse_loss(lat_pred_train, y_lat)
            lon_loss = F.mse_loss(lon_pred_train, y_lon)

            loss = lat_loss + lon_loss
        elif task == "multi_label":
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            bce_loss = nn.BCEWithLogitsLoss()
            target = data.y.clone()
            is_labeled = target == target
            loss = bce_loss(out[is_labeled], target[is_labeled])
        elif task == 'multi_class':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            loss = F.cross_entropy(out, data.y)
        elif task == 'binary_class':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        elif task == 'multi_weight_class':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            loss = weighted_cross_entropy(out, data.y)
        else:
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            loss = (out.squeeze() - data.y).abs().mean()  # MAE

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        loss.backward()
        total_loss += loss.item() * (data.num_graphs if hasattr(data, 'num_graphs') else 1)
        optimizer.step()

    return total_loss / len(loader.dataset)


def test(loader, criterion, split, model, device):
    model.eval()

    y_true = []
    y_pred = []
    correct = 0
    sample = 0
    mae = 0

    if criterion == 'geo_distance':
        all_distances = []

    for data in tqdm(loader, desc=split, leave=False):
        data = data.to(device)
        node_num = data.sph.shape[-1]
        data.sph = data.sph.reshape(-1, node_num, node_num)
        if data.edge_attr is None:
            data.edge_attr = data.edge_index.new_zeros(data.edge_index.shape[1])

        if criterion == 'geo_distance':
            geo_out, _ = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)

            if split == 'val':
                mask = data.val_mask
            elif split == 'test':
                mask = data.test_mask
            else:
                mask = data.train_mask

            lat_pred = geo_out[:, 0]
            lon_pred = geo_out[:, 1]

            lat_pred_np = lat_pred[mask].detach().cpu().numpy().reshape(-1, 1)
            lon_pred_np = lon_pred[mask].detach().cpu().numpy().reshape(-1, 1)

            scaler_lat = data.target_scaler_lat[0] if isinstance(data.target_scaler_lat,
                                                                 list) else data.target_scaler_lat
            scaler_lon = data.target_scaler_lon[0] if isinstance(data.target_scaler_lon,
                                                                 list) else data.target_scaler_lon

            lat_pred_orig = scaler_lat.inverse_transform(lat_pred_np)
            lon_pred_orig = scaler_lon.inverse_transform(lon_pred_np)

            lat_true = data.y_lat[mask].cpu().numpy()
            lon_true = data.y_lon[mask].cpu().numpy()

            distances = []
            for i in range(lat_pred_orig.shape[0]):
                lat1, lon1 = lat_pred_orig[i][0], lon_pred_orig[i][0]
                lat2, lon2 = lat_true[i][0], lon_true[i][0]
                dist = geodistance(lon1, lat1, lon2, lat2)
                distances.append(dist)

            all_distances.extend(distances)
        elif criterion == 'accuracy':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            out = out.max(dim=1)[1]
            correct += out.eq(data.y).sum().item()
            sample += data.y.shape[0]
        elif criterion == 'ogbg':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            y_true.append(data.y.view(out.shape).detach().cpu())
            y_pred.append(out.detach().cpu())
        elif criterion == 'mae':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            mae += (out.squeeze() - data.y).abs().sum().item()
        elif criterion == 'accuracy_sbm':
            out = model(data.x, data.pe, data.edge_index, data.edge_attr, data.batch, data.sph)
            y_true.append(data.y.detach().cpu())
            y_pred.append(out.detach().cpu())

    if criterion == 'geo_distance':
        avg_distance = sum(all_distances) / len(all_distances)
        median_distance = mediannum(all_distances)
        max_distance = max(all_distances)
        result = {
            'avg_distance': avg_distance,
            'median_distance': median_distance,
            'max_distance': max_distance,
            'distances': all_distances
        }
    elif criterion == 'accuracy':
        result = correct / sample
    elif criterion == 'ogbg':
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        evaluator = Evaluator(args.dataset)
        _, result = evaluator.eval(input_dict).popitem()
    elif criterion == 'mae':
        result = mae / len(loader.dataset)
    elif criterion == 'accuracy_sbm':
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        result = accuracy_sbm(y_pred, y_true)
    else:
        raise ValueError("Invalid criterion. Supported criterion: 'accuracy', 'ogbg', 'mae', 'geo_distance'")

    return result


# 将训练结果保存到日志（需求一）
def save_epoch_results(args, epoch, loss, train_res, val_res, test_res, best_train_res=None):
    epoch_file = os.path.join(args.save_path, str(args.seed), f"epoch_{epoch:03d}_results.txt")
    with open(epoch_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Run ID: {args.run}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Training Loss: {loss:.4f}\n")
        f.write("Training Parameters:\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  N_hop: {args.n_hop}\n")
        f.write(f"  Delay_weight: {args.delay_weight}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Num Layers: {args.num_layers}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")
        f.write(f"  Slope: {args.slope}\n")

        if args.criterion == 'geo_distance':
            f.write(f"Train - Avg Distance: {train_res['avg_distance']:.4f} km, "
                    f"Median Distance: {train_res['median_distance']:.4f} km, "
                    f"Max Distance: {train_res['max_distance']:.4f} km\n")
            f.write(f"Val - Avg Distance: {val_res['avg_distance']:.4f} km, "
                    f"Median Distance: {val_res['median_distance']:.4f} km, "
                    f"Max Distance: {val_res['max_distance']:.4f} km\n")
            f.write(f"Test - Avg Distance: {test_res['avg_distance']:.4f} km, "
                    f"Median Distance: {test_res['median_distance']:.4f} km, "
                    f"Max Distance: {test_res['max_distance']:.4f} km\n")
            if best_train_res is not None:
                f.write(f"Best Train - Avg Distance: {best_train_res['avg_distance']:.4f} km, "
                        f"Median Distance: {best_train_res['median_distance']:.4f} km, "
                        f"Max Distance: {best_train_res['max_distance']:.4f} km\n")
        else:
            f.write(f"Train Result: {train_res:.4f}\n")
            f.write(f"Val Result: {val_res:.4f}\n")
            f.write(f"Test Result: {test_res:.4f}\n")
    #print(f"Epoch {epoch} results saved to {epoch_file}")


# 保存最佳结果到文件（需求二）
def results_to_file(args, val_metric, test_metric):
    result_file = os.path.join(args.save_path, str(args.seed), "best_results.txt")
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Run ID: {args.run}\n")
        f.write(f"Best Val Metric: {val_metric:.4f}\n")
        f.write(f"Best Test Metric: {test_metric:.4f}\n")
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    now = datetime.now()
    now = now.strftime("%m_%d-%H_%M_%S")
    warnings.filterwarnings('ignore')

    args = gene_arg()
    set_config(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    print(args)

    run_name = f"{args.dataset}"
    args.save_path = f"exps/{run_name}-{now}"
    os.makedirs(os.path.join(args.save_path, str(args.seed)), exist_ok=True)

    # 使用单 GPU，指定设备
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader, num_tasks, num_features, num_edge_features = load_data(args)

    model = Gradformer(args=args, node_dim=num_features, edge_dim=num_edge_features, num_tasks=num_tasks,
                       mpnn=args.mpnn, pool=args.pool).to(device)

    if args.dataset in ['NCI1', 'IMDB-BINARY', 'COLLAB', 'PROTEINS', 'MUTAG']:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay)
        scheduler = get_scheduler(optimizer, args.warmup_epoch, args.epochs * len(train_loader), -1)

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_param}")

    start_time = time.time()
    best_val = 1e5
    best_train_res = None  # 初始化最佳训练结果

    for epoch in range(1, args.epochs + 1):
        loss = train(train_loader, model, args.criterion, optimizer, device, args.task)

        if args.criterion == 'geo_distance':
            train_res = test(train_loader, args.criterion, "train", model, device)
            val_res = test(val_loader, args.criterion, "val", model, device)
            test_res = test(test_loader, args.criterion, "test", model, device)

            val_metric = val_res['median_distance']

            # 更新最佳训练结果
            if best_val > val_metric:
                best_val = val_metric
                best_train_res = train_res
                state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))
                np.save(os.path.join(args.save_path, str(args.seed), "best_test_distances.npy"),
                        np.array(test_res['distances']))

            # 保存当前 epoch 结果，包含最佳训练结果
            save_epoch_results(args, epoch, loss, train_res, val_res, test_res, best_train_res)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: [avg={train_res["avg_distance"]:.4f}, median={train_res["median_distance"]:.4f}, max={train_res["max_distance"]:.4f}], '
                  f'Val: [avg={val_res["avg_distance"]:.4f}, median={val_res["median_distance"]:.4f}, max={val_res["max_distance"]:.4f}], '
                  f'Test: [avg={test_res["avg_distance"]:.4f}, median={test_res["median_distance"]:.4f}, max={test_res["max_distance"]:.4f}]')
        else:
            val_res = test(val_loader, args.criterion, "val", model, device)
            test_res = test(test_loader, args.criterion, "test", model, device)
            val_metric = val_res
            if args.dataset in ['ZINC', 'IP_GEO']:
                if best_val > val_metric:
                    best_val = val_metric
                    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                    torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))
            else:
                if best_val < val_metric:
                    best_val = val_metric
                    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
                    torch.save(state_dict, os.path.join(args.save_path, str(args.seed), "best_model.pt"))

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_res:.4f}, Test: {test_res:.4f}')

        if scheduler is not None:
            scheduler.step()

    # 加载最佳模型并计算最终结果
    state_dict = torch.load(os.path.join(args.save_path, str(args.seed), "best_model.pt"))
    model.load_state_dict(state_dict["model"])

    best_train_res = test(train_loader, args.criterion, "best_train", model, device)
    best_val_res = test(val_loader, args.criterion, "best_val", model, device)
    best_test_res = test(test_loader, args.criterion, "best_test", model, device)

    # 保存最佳结果到单独文件
    with open(os.path.join(args.save_path, str(args.seed), "best_train_results.txt"), 'w') as f:
        f.write(f"Best Train - Avg Distance: {best_train_res['avg_distance']:.4f} km, "
                f"Median Distance: {best_train_res['median_distance']:.4f} km, "
                f"Max Distance: {best_train_res['max_distance']:.4f} km\n")
    with open(os.path.join(args.save_path, str(args.seed), "best_val_results.txt"), 'w') as f:
        f.write(f"Best Val - Avg Distance: {best_val_res['avg_distance']:.4f} km, "
                f"Median Distance: {best_val_res['median_distance']:.4f} km, "
                f"Max Distance: {best_val_res['max_distance']:.4f} km\n")
    with open(os.path.join(args.save_path, str(args.seed), "best_test_results.txt"), 'w') as f:
        f.write(f"Best Test - Avg Distance: {best_test_res['avg_distance']:.4f} km, "
                f"Median Distance: {best_test_res['median_distance']:.4f} km, "
                f"Max Distance: {best_test_res['max_distance']:.4f} km\n")

    # 打印最佳结果
    print(f'Best Train: [avg={best_train_res["avg_distance"]:.4f}, median={best_train_res["median_distance"]:.4f}, '
          f'max={best_train_res["max_distance"]:.4f}]')
    print(f'Best Val: [avg={best_val_res["avg_distance"]:.4f}, median={best_val_res["median_distance"]:.4f}, '
          f'max={best_val_res["max_distance"]:.4f}]')
    print(f'Best Test: [avg={best_test_res["avg_distance"]:.4f}, median={best_test_res["median_distance"]:.4f}, '
          f'max={best_test_res["max_distance"]:.4f}]')
    results_to_file(args, best_val_res["median_distance"], best_test_res["median_distance"])

    print(f"Total time elapsed: {time.time() - start_time:.4f}s")