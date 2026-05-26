import time
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


def log(message):
    print('[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), message), flush=True)


def describe_graph(g):
    try:
        return 'nodes={}, edges={}'.format(g.number_of_nodes(), g.number_of_edges())
    except Exception as exc:
        return 'graph_info_unavailable={}'.format(repr(exc))


def main(main_args):
    start_time = time.time()
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)
    dataset_name = main_args.dataset
    y_pred = None
    y_true = None

    log('START eval.py')
    log('dataset={}'.format(dataset_name))
    log('requested_device_arg={}'.format(main_args.device))
    log('torch_device={}'.format(device))
    log('torch_cuda_available={}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        try:
            log('cuda_device_name={}'.format(torch.cuda.get_device_name(device)))
        except Exception:
            log('cuda_device_name={}'.format(torch.cuda.get_device_name(0)))

    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    log('model_config num_hidden={}, num_layers={}'.format(main_args.num_hidden, main_args.num_layers))

    set_random_seed(0)
    log('random_seed=0')

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        log('[BATCH] loading batch-level dataset')
        t0 = time.time()
        dataset = load_batch_level_dataset(dataset_name)
        log('[BATCH] dataset loaded in {:.2f}s'.format(time.time() - t0))

        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        log('[BATCH] n_node_feat={}, n_edge_feat={}'.format(n_node_feat, n_edge_feat))

        log('[MODEL] building model')
        model = build_model(main_args)

        checkpoint_path = "./checkpoints/checkpoint-{}.pt".format(dataset_name)
        log('[MODEL] loading checkpoint {}'.format(checkpoint_path))
        t0 = time.time()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log('[MODEL] checkpoint loaded in {:.2f}s'.format(time.time() - t0))

        model = model.to(device)
        pooler = Pooling(main_args.pooling)

        log('[BATCH] starting batch-level evaluation')
        t0 = time.time()
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], dataset_name, main_args.n_dim,
                                                    main_args.e_dim)
        log('[BATCH] batch-level evaluation finished in {:.2f}s'.format(time.time() - t0))
    else:
        log('[ENTITY] loading metadata')
        t0 = time.time()
        metadata = load_metadata(dataset_name)
        log('[ENTITY] metadata loaded in {:.2f}s'.format(time.time() - t0))

        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']
        log('[ENTITY] node_feature_dim={}, edge_feature_dim={}, n_train={}, n_test={}, malicious_count={}'.format(
            main_args.n_dim, main_args.e_dim, n_train, n_test, len(malicious)
        ))

        log('[MODEL] building model')
        t0 = time.time()
        model = build_model(main_args)
        log('[MODEL] model built in {:.2f}s'.format(time.time() - t0))

        checkpoint_path = "./checkpoints/checkpoint-{}.pt".format(dataset_name)
        log('[MODEL] loading checkpoint {}'.format(checkpoint_path))
        t0 = time.time()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log('[MODEL] checkpoint loaded in {:.2f}s'.format(time.time() - t0))

        log('[MODEL] moving model to {}'.format(device))
        t0 = time.time()
        model = model.to(device)
        log('[MODEL] model moved to device in {:.2f}s'.format(time.time() - t0))

        model.eval()

        with torch.no_grad():
            x_train = []
            log('[TRAIN] start embedding {} train graphs'.format(n_train))
            train_start = time.time()
            for i in range(n_train):
                t0 = time.time()
                log('[TRAIN] {}/{} loading graph train{}.pkl'.format(i + 1, n_train, i))
                g = load_entity_level_dataset(dataset_name, 'train', i)
                log('[TRAIN] {}/{} loaded graph: {} in {:.2f}s'.format(
                    i + 1, n_train, describe_graph(g), time.time() - t0
                ))

                t0 = time.time()
                log('[TRAIN] {}/{} moving graph to {}'.format(i + 1, n_train, device))
                g = g.to(device)
                log('[TRAIN] {}/{} embedding graph'.format(i + 1, n_train))
                emb = model.embed(g).cpu().numpy()
                x_train.append(emb)
                log('[TRAIN] {}/{} embedding done: shape={} in {:.2f}s'.format(
                    i + 1, n_train, emb.shape, time.time() - t0
                ))
                del g

            log('[TRAIN] concatenating train embeddings')
            t0 = time.time()
            x_train = np.concatenate(x_train, axis=0)
            log('[TRAIN] x_train shape={} concat_time={:.2f}s total_train_time={:.2f}s'.format(
                x_train.shape, time.time() - t0, time.time() - train_start
            ))

            skip_benign = 0
            x_test = []
            log('[TEST] start embedding {} test graphs'.format(n_test))
            test_start = time.time()
            for i in range(n_test):
                t0 = time.time()
                log('[TEST] {}/{} loading graph test{}.pkl'.format(i + 1, n_test, i))
                g = load_entity_level_dataset(dataset_name, 'test', i)
                log('[TEST] {}/{} loaded graph: {} in {:.2f}s'.format(
                    i + 1, n_test, describe_graph(g), time.time() - t0
                ))

                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                    log('[TEST] {}/{} skip_benign updated to {}'.format(i + 1, n_test, skip_benign))

                t0 = time.time()
                log('[TEST] {}/{} moving graph to {}'.format(i + 1, n_test, device))
                g = g.to(device)
                log('[TEST] {}/{} embedding graph'.format(i + 1, n_test))
                emb = model.embed(g).cpu().numpy()
                x_test.append(emb)
                log('[TEST] {}/{} embedding done: shape={} in {:.2f}s'.format(
                    i + 1, n_test, emb.shape, time.time() - t0
                ))
                del g

            log('[TEST] concatenating test embeddings')
            t0 = time.time()
            x_test = np.concatenate(x_test, axis=0)
            log('[TEST] x_test shape={} concat_time={:.2f}s total_test_time={:.2f}s'.format(
                x_test.shape, time.time() - t0, time.time() - test_start
            ))

            n = x_test.shape[0]
            log('[LABEL] building y_test length={}'.format(n))
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            log('[LABEL] total_positive_labels_before_filter={}'.format(int(y_test.sum())))

            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            log('[FILTER] filtering test nodes with skip_benign={}'.format(skip_benign))
            t0 = time.time()
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            log('[FILTER] selected {} / {} test nodes in {:.2f}s'.format(
                len(test_idx), x_test.shape[0], time.time() - t0
            ))

            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            log('[FILTER] result_x_test shape={}, positives={}/{}'.format(
                result_x_test.shape, int(result_y_test.sum()), len(result_y_test)
            ))

            del x_test, y_test

            # Lấy thêm y_pred (nhãn dự đoán) và y_true (nhãn thật) từ hàm đánh giá
            log('[EVAL] starting evaluate_entity_level_using_knn')
            t0 = time.time()
            test_auc, test_std, y_pred, y_true = evaluate_entity_level_using_knn(
                dataset_name, x_train, result_x_test, result_y_test
            )
            log('[EVAL] evaluate_entity_level_using_knn finished in {:.2f}s'.format(time.time() - t0))

    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}", flush=True)
    log('total_runtime={:.2f}s'.format(time.time() - start_time))

    # ==========================================
    # ĐOẠN CODE THÊM VÀO ĐỂ LƯU FILE OUTPUT
    # ==========================================
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    # 1. Lưu eval.log (ghi lại kết quả AUC dạng text)
    log('[OUTPUT] writing eval.log')
    with open("eval.log", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Test AUC: {test_auc:.4f} ± {test_std:.4f}\n")

    # Kiểm tra xem hàm evaluate có trả về y_pred và y_true không
    if y_pred is not None and y_true is not None:
        log('[OUTPUT] writing metrics.json, metrics.csv, confusion_matrix.png, metrics_bar.png')
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
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign (0)", "Malware (1)"],
                    yticklabels=["Benign (0)", "Malware (1)"])
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

        plt.figure(figsize=(6, 4))
        plt.bar(metrics_to_plot, values, color=['#4C72B0', '#DD8452', '#55A868'])
        plt.ylim(0, 1.1)
        plt.title("Evaluation Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig("metrics_bar.png", dpi=300)
        plt.close()

        print("Đã tạo xong các file output: eval.log, metrics.csv, metrics.json, confusion_matrix.png, metrics_bar.png", flush=True)
    else:
        print("Đã tạo eval.log. (Không thể vẽ biểu đồ vì hàm evaluate không trả về y_pred và y_true).", flush=True)

    return


if __name__ == '__main__':
    args = build_args()
    main(args)
