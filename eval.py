import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                    main_args.e_dim)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            # Lấy thêm y_pred (nhãn dự đoán) và y_true (nhãn thật) từ hàm đánh giá
            test_auc, test_std, y_pred, y_true = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test, result_y_test)
    
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")

    # ==========================================
    # ĐOẠN CODE THÊM VÀO ĐỂ LƯU FILE OUTPUT
    # ==========================================
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    # 1. Lưu eval.log (ghi lại kết quả AUC dạng text)
    with open("eval.log", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Test AUC: {test_auc:.4f} ± {test_std:.4f}\n")

    # Kiểm tra xem hàm evaluate có trả về y_pred và y_true không
    if y_pred is not None and y_true is not None:
        # 2. Sinh Metrics (Classification Report) và lưu ra CSV & JSON
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        
        # Lưu ra JSON
        with open("metrics.json", "w") as f:
            json.dump(report_dict, f, indent=4)
        
        # Lưu ra CSV
        df_metrics = pd.DataFrame(report_dict).transpose()
        df_metrics.to_csv("metrics.csv", index=True)

        # 3. Vẽ và lưu Confusion Matrix (Ma trận nhầm lẫn)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign (0)", "Malware (1)"], yticklabels=["Benign (0)", "Malware (1)"])
        plt.title("Confusion Matrix")
        plt.ylabel("Thực tế (True Label)")
        plt.xlabel("Dự đoán (Predicted Label)")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.close()

        # 4. Vẽ Metrics Bar Chart (Biểu đồ cột cho Precision, Recall, F1)
        metrics_to_plot = ['precision', 'recall', 'f1-score']
        macro_avg = report_dict['macro avg']
        values = [macro_avg[m] for m in metrics_to_plot]
        
        plt.figure(figsize=(6,4))
        plt.bar(metrics_to_plot, values, color=['#4C72B0', '#DD8452', '#55A868'])
        plt.ylim(0, 1.1)
        plt.title("Evaluation Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig("metrics_bar.png", dpi=300)
        plt.close()

        print("Đã tạo xong các file output: eval.log, metrics.csv, metrics.json, confusion_matrix.png, metrics_bar.png")
    else:
        print("Đã tạo eval.log. (Không thể vẽ biểu đồ vì hàm evaluate không trả về y_pred và y_true).")

    return


if __name__ == '__main__':
    args = build_args()
    main(args)
