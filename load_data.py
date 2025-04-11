import os
import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import random_split, DataLoader

from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset
from torch_geometric.utils import degree

from data_utils.wrapper import NewDataset
from utils.process import process_sph
from math import radians, cos, sin, asin, sqrt
from sklearn import preprocessing


# 全局变量（保留以兼容可能的其他代码，但优先使用 Data 对象传递）
target_scaler_lat = None
target_scaler_lon = None

# Add the geodistance calculation function from GNN-Geo
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # Earth's average radius, 6371km
    distance = round(distance / 1000, 3)
    return distance


# Add median function for error calculation
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


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def fn(data_list):
    #print(f"Processing {len(data_list)} items in data_list")
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for i, data in enumerate(data_list):
        #print(f"Data {i} keys before: {data.keys()}")
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)

        keys_to_convert = []
        for key in list(data.keys()):
            if not isinstance(key, str):
                print(f"Found non-string key in Data {i}: {key} (type: {type(key)})")
                keys_to_convert.append(key)

        for key in keys_to_convert:
            data[str(key)] = data[key]
            del data[key]

        #print(f"Data {i} keys after: {data.keys()}")

    batched_data = Batch.from_data_list(data_list)
    return batched_data


def load_data(args):
    if args.dataset in ['NCI1', 'NCI109', 'Mutagenicity', 'PTC_MR', 'AIDS', 'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB',
                        'PROTEINS', 'DD', 'MUTAG', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_tudataset(args)
    elif args.dataset[:4] == 'ogbg':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_ogbg(args)
    elif args.dataset == 'ZINC':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_zinc(args)
    elif args.dataset in ['CLUSTER', 'PATTERN']:
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_node_cls(args)
    elif args.dataset == 'IP_GEO':
        num_tasks, num_features, edge_features, training_set, validation_set, test_set = load_ip_geo(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=fn)
    val_loader = DataLoader(validation_set, batch_size=args.eval_batch_size, collate_fn=fn)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, collate_fn=fn)
    return train_loader, val_loader, test_loader, num_tasks, num_features, edge_features


def load_ip_geo(args):
    global target_scaler_lat, target_scaler_lon  # 使用全局变量（可选）

    filepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if os.name == 'nt':  # Windows
        node_file = os.path.join(filepath, 'data', args.dataset, 'ip_feature.txt')
        edge_file = os.path.join(filepath, 'data', args.dataset, 'edge_feature.txt')
        target_file = os.path.join(filepath, 'data', args.dataset, 'dstip_id_allinfo.txt')
    else:  # Linux/Unix
        node_file = os.path.join(filepath, 'data', args.dataset, 'ip_feature.txt')
        edge_file = os.path.join(filepath, 'data', args.dataset, 'edge_feature.txt')
        target_file = os.path.join(filepath, 'data', args.dataset, 'dstip_id_allinfo.txt')

    node_id_delay = {}
    node_id_ip = {}
    node_list = []
    node_attr_list = []

    with open(node_file, 'r') as fr:
        for line in fr.readlines():
            str_list = line.strip('\r\n').split(sep=',')
            temp_ip = str(str_list[0])
            temp_nodeID = int(str_list[1])
            node_id_ip[temp_nodeID] = temp_ip
            temp_node_attr = list(map(float, str_list[2:]))
            temp_node_delay = float(str_list[2])
            node_id_delay[temp_nodeID] = temp_node_delay
            node_list.append(temp_nodeID)
            node_attr_list.append(temp_node_attr)

    node_attr_array = np.array(node_attr_list)

    src_list = []
    dst_list = []
    edge_attr_list = []

    with open(edge_file, 'r') as fr:
        for line in fr.readlines():
            str_list = line.strip('\r\n').split(sep=',')
            temp_src = int(str_list[0])
            temp_dst = int(str_list[1])
            temp_edge_attr = list(map(float, str_list[2:]))

            src_list.append(temp_src)
            dst_list.append(temp_dst)
            edge_attr_list.append(temp_edge_attr)

            src_list.append(temp_dst)
            dst_list.append(temp_src)
            edge_attr_list.append(temp_edge_attr)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr_array = np.array(edge_attr_list)

    target_lat_label_dict = {}
    target_lon_label_dict = {}
    label_lat_array = []
    label_lon_array = []

    with open(target_file, 'r', encoding='UTF-8') as fr:
        for line in fr.readlines():
            str_list = line.strip('\r\n').split(sep=',')
            node_id = int(str_list[1])
            lat_label = float(str_list[3])
            lon_label = float(str_list[4])
            label_lat_array.append(lat_label)
            label_lon_array.append(lon_label)

            target_lat_label_dict[node_id] = lat_label
            target_lon_label_dict[node_id] = lon_label

    target_scaler1 = preprocessing.MinMaxScaler()
    target_scaler2 = preprocessing.MinMaxScaler()

    train_node_id_list = []
    lat_train_node_label_list = []
    lon_train_node_label_list = []

    val_node_id_list = []
    lat_val_node_label_list = []
    lon_val_node_label_list = []

    test_node_id_list = []
    lat_test_node_label_list = []
    lon_test_node_label_list = []

    np.random.seed(0)

    for key in target_lat_label_dict.keys():
        node_id = int(key)
        lat_label = target_lat_label_dict[node_id]
        lon_label = target_lon_label_dict[node_id]

        random_val = np.random.random()
        if random_val < 0.1:
            test_node_id_list.append(node_id)
            lat_test_node_label_list.append(lat_label)
            lon_test_node_label_list.append(lon_label)
        elif random_val < 0.3:
            val_node_id_list.append(node_id)
            lat_val_node_label_list.append(lat_label)
            lon_val_node_label_list.append(lon_label)
        else:
            train_node_id_list.append(node_id)
            lat_train_node_label_list.append(lat_label)
            lon_train_node_label_list.append(lon_label)

    lat_train_array = np.array(lat_train_node_label_list).reshape(-1, 1)
    lon_train_array = np.array(lon_train_node_label_list).reshape(-1, 1)

    norm_lat_train_array = target_scaler1.fit_transform(lat_train_array)
    norm_lon_train_array = target_scaler2.fit_transform(lon_train_array)

    norm_lat_val_array = target_scaler1.transform(np.array(lat_val_node_label_list).reshape(-1, 1))
    norm_lon_val_array = target_scaler2.transform(np.array(lon_val_node_label_list).reshape(-1, 1))
    norm_lat_test_array = target_scaler1.transform(np.array(lat_test_node_label_list).reshape(-1, 1))
    norm_lon_test_array = target_scaler2.transform(np.array(lon_test_node_label_list).reshape(-1, 1))

    norm_lat_train_array = norm_lat_train_array.astype(np.float32)
    norm_lon_train_array = norm_lon_train_array.astype(np.float32)
    norm_lat_val_array = norm_lat_val_array.astype(np.float32)
    norm_lon_val_array = norm_lon_val_array.astype(np.float32)
    norm_lat_test_array = norm_lat_test_array.astype(np.float32)
    norm_lon_test_array = norm_lon_test_array.astype(np.float32)

    train_node_ids = torch.tensor(train_node_id_list, dtype=torch.long)
    val_node_ids = torch.tensor(val_node_id_list, dtype=torch.long)
    test_node_ids = torch.tensor(test_node_id_list, dtype=torch.long)

    from torch_geometric.data import Data

    x = torch.zeros((max(node_list) + 1, len(node_attr_list[0])), dtype=torch.float)
    for i, node_id in enumerate(node_list):
        x[node_id] = torch.tensor(node_attr_list[i], dtype=torch.float)

    edge_attr = torch.tensor(edge_attr_array, dtype=torch.float)

    train_mask = torch.zeros(max(node_list) + 1, dtype=torch.bool)
    val_mask = torch.zeros(max(node_list) + 1, dtype=torch.bool)
    test_mask = torch.zeros(max(node_list) + 1, dtype=torch.bool)

    train_mask[train_node_ids] = True
    val_mask[val_node_ids] = True
    test_mask[test_node_ids] = True

    y_lat = torch.zeros(max(node_list) + 1, 1, dtype=torch.float)
    y_lon = torch.zeros(max(node_list) + 1, 1, dtype=torch.float)
    norm_y_lat = torch.zeros(max(node_list) + 1, 1, dtype=torch.float)
    norm_y_lon = torch.zeros(max(node_list) + 1, 1, dtype=torch.float)

    for i, node_id in enumerate(train_node_id_list):
        y_lat[node_id] = torch.tensor([lat_train_node_label_list[i]], dtype=torch.float)
        y_lon[node_id] = torch.tensor([lon_train_node_label_list[i]], dtype=torch.float)
        norm_y_lat[node_id] = torch.tensor([norm_lat_train_array[i][0]], dtype=torch.float)
        norm_y_lon[node_id] = torch.tensor([norm_lon_train_array[i][0]], dtype=torch.float)

    for i, node_id in enumerate(val_node_id_list):
        y_lat[node_id] = torch.tensor([lat_val_node_label_list[i]], dtype=torch.float)
        y_lon[node_id] = torch.tensor([lon_val_node_label_list[i]], dtype=torch.float)
        norm_y_lat[node_id] = torch.tensor([norm_lat_val_array[i][0]], dtype=torch.float)
        norm_y_lon[node_id] = torch.tensor([norm_lon_val_array[i][0]], dtype=torch.float)

    for i, node_id in enumerate(test_node_id_list):
        y_lat[node_id] = torch.tensor([lat_test_node_label_list[i]], dtype=torch.float)
        y_lon[node_id] = torch.tensor([lon_test_node_label_list[i]], dtype=torch.float)
        norm_y_lat[node_id] = torch.tensor([norm_lat_test_array[i][0]], dtype=torch.float)
        norm_y_lon[node_id] = torch.tensor([norm_lon_test_array[i][0]], dtype=torch.float)

    y = torch.cat([norm_y_lat, norm_y_lon], dim=1)

    # 定义 scaler 并赋值给全局变量（可选）
    target_scaler_lat = target_scaler1
    target_scaler_lon = target_scaler2

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        y_lat=y_lat,
        y_lon=y_lon,
        norm_y_lat=torch.tensor(norm_lat_train_array, dtype=torch.float),  # 只包含训练节点的归一化值
        norm_y_lon=torch.tensor(norm_lon_train_array, dtype=torch.float),  # 只包含训练节点的归一化值
        train_node_ids=train_node_ids,
        val_node_ids=val_node_ids,
        test_node_ids=test_node_ids,
        target_scaler_lat=target_scaler1,  # 添加到 Data 对象
        target_scaler_lon=target_scaler2   # 添加到 Data 对象
    )

    max_node_id = max(node_list) + 1
    node_id_delay_tensor = torch.zeros(max_node_id, dtype=torch.float)
    for nid, delay in node_id_delay.items():
        node_id_delay_tensor[nid] = delay
    data.node_id_delay = node_id_delay_tensor

    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    data = transform(data)

    training_set = NewDataset([data])
    validation_set = NewDataset([data])
    test_set = NewDataset([data])

    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')

    num_tasks = 2
    num_features = x.size(1)
    edge_features = edge_attr.size(1) if edge_attr.size(0) > 0 else 0

    return num_tasks, num_features, edge_features, training_set, validation_set, test_set


def load_tudataset(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    dataset = TUDataset(os.path.join(args.data_root, args.dataset),
                        name=args.dataset,
                        pre_transform=transform)
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)
    num_tasks = dataset.num_classes
    num_features = dataset.num_features
    num_edge_features = 1
    data = NewDataset(dataset)
    process_sph(args, data)
    num_training = int(len(data) * 0.7)
    num_val = int(len(data) * 0.2)
    num_test = len(data) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(data, [num_training, num_val, num_test])
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_ogbg(args):
    if args.dataset not in ['ogbg-ppa', 'ogbg-code2']:
        transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    else:
        transform = None
    dataset = PygGraphPropPredDataset(name=args.dataset, root=os.path.join(args.data_root, args.dataset),
                                      pre_transform=transform)
    num_tasks = dataset.num_tasks
    num_features = dataset.num_features
    num_edge_features = dataset.num_edge_features
    split_idx = dataset.get_idx_split()
    training_data = dataset[split_idx['train']]
    validation_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_zinc(args):
    transform = T.AddRandomWalkPE(walk_length=args.pe_origin_dim, attr_name='pe')
    training_data = ZINC(os.path.join(args.data_root, args.dataset), split='train', subset=True,
                         pre_transform=transform)
    validation_data = ZINC(os.path.join(args.data_root, args.dataset), split='val', subset=True,
                           pre_transform=transform)
    test_data = ZINC(os.path.join(args.data_root, args.dataset), split='test', subset=True,
                     pre_transform=transform)
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')
    num_tasks = 1
    num_features = 28
    num_edge_features = 4
    return num_tasks, num_features, num_edge_features, training_set, validation_set, test_set


def load_node_cls(args):
    transform = T.AddLaplacianEigenvectorPE(k=args.pe_origin_dim, attr_name='pe', is_undirected=True)
    training_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='train',
                                        pre_transform=transform)
    validation_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='val',
                                          pre_transform=transform)
    test_data = GNNBenchmarkDataset(os.path.join(args.data_root, args.dataset), name=args.dataset, split='test',
                                    pre_transform=transform)
    num_task = training_data.num_classes
    num_feature = training_data.num_features
    num_edge_features = 1
    training_set = NewDataset(training_data)
    validation_set = NewDataset(validation_data)
    test_set = NewDataset(test_data)
    process_sph(args, training_set, 'train')
    process_sph(args, validation_set, 'val')
    process_sph(args, test_set, 'test')

    return num_task, num_feature, num_edge_features, training_set, validation_set, test_set