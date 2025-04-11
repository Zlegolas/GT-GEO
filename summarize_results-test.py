import os
import re
import pandas as pd


def extract_metrics(file_path):
    """从 best_*.txt 文件中提取 Avg, Median, Max Distance"""
    metrics = {'Avg': None, 'Median': None, 'Max': None}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            avg_match = re.search(r'Avg Distance: ([\d.]+) km', content)
            median_match = re.search(r'Median Distance: ([\d.]+) km', content)
            max_match = re.search(r'Max Distance: ([\d.]+) km', content)
            if avg_match:
                metrics['Avg'] = float(avg_match.group(1))
            if median_match:
                metrics['Median'] = float(median_match.group(1))
            if max_match:
                metrics['Max'] = float(max_match.group(1))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return metrics


def extract_epoch_params(file_path):
    """从 epoch_001_results.txt 中提取参数"""
    params = {
        'Dataset': None, 'Gamma': None, 'N_hop': None, 'Delay_weight': None,
        'Learning Rate': None, 'Dropout': None, 'Batch Size': None,
        'Num Layers': None, 'Weight Decay': None, 'Slope': None
    }
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            params['Dataset'] = re.search(r'Dataset: (\S+)', content).group(1)
            params['Gamma'] = float(re.search(r'Gamma: ([\d.]+)', content).group(1))
            params['N_hop'] = int(re.search(r'N_hop: (\d+)', content).group(1))
            params['Delay_weight'] = float(re.search(r'Delay_weight: ([\d.]+)', content).group(1))
            params['Learning Rate'] = float(re.search(r'Learning Rate: ([\d.]+)', content).group(1))
            params['Dropout'] = float(re.search(r'Dropout: ([\d.]+)', content).group(1))
            params['Batch Size'] = int(re.search(r'Batch Size: (\d+)', content).group(1))
            params['Num Layers'] = int(re.search(r'Num Layers: (\d+)', content).group(1))
            params['Weight Decay'] = float(re.search(r'Weight Decay: ([\d.e-]+)', content).group(1))
            params['Slope'] = float(re.search(r'Slope: ([\d.]+)', content).group(1))
    except (FileNotFoundError, AttributeError) as e:
        print(f"Error parsing {file_path}: {e}")
    return params


def generate_excel():
    exp_dir = "exps"
    output_excel = "experiment_summary.xlsx"

    # 定义 Excel 表格列
    columns = [
        'Dataset', 'Gamma', 'N_hop', 'Delay_weight', 'Learning Rate', 'Dropout',
        'Batch Size', 'Num Layers', 'Weight Decay', 'Slope',
        'Best Test-Avg', 'Best Test-Median', 'Best Test-Max',
        'Best Train-Avg', 'Best Train-Median', 'Best Train-Max',
        'Best Val-Avg', 'Best Val-Median', 'Best Val-Max'
    ]
    data = []

    # 遍历 exps 目录
    for run_dir in os.listdir(exp_dir):
        if not os.path.isdir(os.path.join(exp_dir, run_dir)):
            continue
        if not run_dir.startswith("IP_GEO-"):
            continue

        result_dir = os.path.join(exp_dir, run_dir, "12344")
        if not os.path.exists(result_dir):
            continue

        # 提取文件路径
        best_test_file = os.path.join(result_dir, "best_test_results.txt")
        best_train_file = os.path.join(result_dir, "best_train_results.txt")
        best_val_file = os.path.join(result_dir, "best_val_results.txt")
        epoch_001_file = os.path.join(result_dir, "epoch_001_results.txt")

        # 提取参数和结果
        params = extract_epoch_params(epoch_001_file)
        if not params['Dataset']:
            print(f"Skipping {run_dir} due to missing epoch_001_results.txt data")
            continue

        test_metrics = extract_metrics(best_test_file)
        train_metrics = extract_metrics(best_train_file)
        val_metrics = extract_metrics(best_val_file)

        # 构建一行数据
        row = [
            params['Dataset'],
            params['Gamma'],
            params['N_hop'],
            params['Delay_weight'],
            params['Learning Rate'],
            params['Dropout'],
            params['Batch Size'],
            params['Num Layers'],
            params['Weight Decay'],
            params['Slope'],
            test_metrics['Avg'],
            test_metrics['Median'],
            test_metrics['Max'],
            train_metrics['Avg'],
            train_metrics['Median'],
            train_metrics['Max'],
            val_metrics['Avg'],
            val_metrics['Median'],
            val_metrics['Max']
        ]
        data.append(row)

    # 创建 DataFrame 并保存为 Excel
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(output_excel, index=False)
    print(f"Excel summary saved to {output_excel}")


if __name__ == "__main__":
    generate_excel()