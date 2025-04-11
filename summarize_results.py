#整理每次脚本运行的最好结果


import os

def summarize_results():
    exp_dir = "exps"
    summary_file = "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Summary of All Experiments\n")
        f.write("=" * 50 + "\n")
        for run_dir in os.listdir(exp_dir):
            if not os.path.isdir(os.path.join(exp_dir, run_dir)):
                continue
            result_file = os.path.join(exp_dir, run_dir, "12344", "best_results.txt")
            if os.path.exists(result_file):
                with open(result_file, 'r') as rf:
                    lines = rf.readlines()
                    params = run_dir.split('-')[0].split('_')[1:]  # 提取参数部分，例如 "g0.5_n4_dw0.01_lr0.0005_do0.05_bs16"
                    gamma = params[0][1:]  # "0.5"
                    n_hop = params[1][1:]  # "4"
                    delay_weight = params[2][2:]  # "0.01"
                    lr = params[3][2:]  # "0.0005"
                    dropout = params[4][2:]  # "0.05"
                    batch_size = params[5][2:]  # "16"

                    # 从 main.py 的最终输出中获取最佳结果
                    best_train_file = os.path.join(exp_dir, run_dir, "12344", "best_train_results.txt")
                    best_val_file = os.path.join(exp_dir, run_dir, "12344", "best_val_results.txt")
                    best_test_file = os.path.join(exp_dir, run_dir, "12344", "best_test_results.txt")

                    f.write(f"Run ID: {run_dir}\n")
                    f.write(f"Parameters: gamma={gamma}, n_hop={n_hop}, delay_weight={delay_weight}, lr={lr}, dropout={dropout}, batch_size={batch_size}\n")
                    if os.path.exists(best_train_file):
                        with open(best_train_file, 'r') as tf:
                            f.write(tf.read())
                    if os.path.exists(best_val_file):
                        with open(best_val_file, 'r') as vf:
                            f.write(vf.read())
                    if os.path.exists(best_test_file):
                        with open(best_test_file, 'r') as tf:
                            f.write(tf.read())
                    f.write("-" * 50 + "\n")
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    summarize_results()