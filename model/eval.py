import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset


def log(message):
    print('[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), message), flush=True)


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    model.eval()
    x_list = []
    y_list = []
    log('[BATCH_EVAL] loading batch-level dataset {}'.format(dataset))
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    log('[BATCH_EVAL] embedding {} graphs'.format(len(full)))
    with torch.no_grad():
        for pos, i in enumerate(full):
            t0 = time.time()
            log('[BATCH_EVAL] {}/{} loading/transforming graph index={}'.format(pos + 1, len(full), i))
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
            log('[BATCH_EVAL] {}/{} done: out_shape={}, label={}, time={:.2f}s'.format(
                pos + 1, len(full), out.shape, label, time.time() - t0
            ))
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    log('[BATCH_EVAL] embeddings shape={}, labels shape={}'.format(x.shape, y.shape))
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(100, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = min(int(train_count * 0.02), 10)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    log('[BATCH_KNN] dataset={}, repeat={}, train_count={}, n_neighbors={}, benign={}, attack={}'.format(
        dataset, repeat, train_count, n_neighbors, len(benign_idx), len(attack_idx)
    ))
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            t_repeat = time.time()
            log('[BATCH_KNN] repeat {}/{} start'.format(s + 1, repeat))
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)
            log('[BATCH_KNN] repeat {}/{} done in {:.2f}s auc={} f1={}'.format(
                s + 1, repeat, time.time() - t_repeat, auc, f1[max_f1_idx]
            ))

        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)), flush=True)
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)), flush=True)
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)), flush=True)
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)), flush=True)
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)), flush=True)
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)), flush=True)
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)), flush=True)
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)), flush=True)
        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        log('[BATCH_KNN] fitting NearestNeighbors')
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        log('[BATCH_KNN] computing train distances')
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        log('[BATCH_KNN] computing test distances')
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        print('AUC: {}'.format(auc), flush=True)
        print('F1: {}'.format(f1[best_idx]), flush=True)
        print('PRECISION: {}'.format(prec[best_idx]), flush=True)
        print('RECALL: {}'.format(rec[best_idx]), flush=True)
        print('TN: {}'.format(tn), flush=True)
        print('FN: {}'.format(fn), flush=True)
        print('TP: {}'.format(tp), flush=True)
        print('FP: {}'.format(fp), flush=True)
        return auc, 0.0


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    """Entity-level KNN evaluation with a fixed, train-calibrated threshold.

    Important evaluation policy:
    - AUC is computed on the test labels and does not require a threshold.
    - Precision/Recall/F1 are computed using a threshold selected ONLY from
      the benign training-score distribution.
    - No test labels are used to choose the threshold.

    Environment variables:
    - MAGIC_TRAIN_THRESHOLD_QUANTILE: primary threshold quantile, default 0.999.
    - MAGIC_REPORT_QUANTILES: comma-separated sensitivity quantiles, default
      0.99,0.995,0.999.
    """
    log('[ENTITY_KNN] start dataset={}'.format(dataset))
    log('[ENTITY_KNN] input x_train shape={}, x_test shape={}, y_test shape={}, positives={}'.format(
        x_train.shape, x_test.shape, y_test.shape, int(np.sum(y_test))
    ))

    primary_quantile = float(os.environ.get('MAGIC_TRAIN_THRESHOLD_QUANTILE', '0.999'))
    if not 0.0 < primary_quantile < 1.0:
        raise ValueError('MAGIC_TRAIN_THRESHOLD_QUANTILE must be in (0, 1), got {}'.format(primary_quantile))

    report_quantiles_raw = os.environ.get('MAGIC_REPORT_QUANTILES', '0.99,0.995,0.999')
    report_quantiles = []
    for q in report_quantiles_raw.split(','):
        q = q.strip()
        if not q:
            continue
        qv = float(q)
        if not 0.0 < qv < 1.0:
            raise ValueError('MAGIC_REPORT_QUANTILES values must be in (0, 1), got {}'.format(qv))
        report_quantiles.append(qv)
    if primary_quantile not in report_quantiles:
        report_quantiles.append(primary_quantile)
    report_quantiles = sorted(set(report_quantiles))

    log('[ENTITY_KNN] threshold policy=train_score_quantile')
    log('[ENTITY_KNN] primary_quantile={}'.format(primary_quantile))
    log('[ENTITY_KNN] report_quantiles={}'.format(report_quantiles))

    log('[ENTITY_KNN] standardizing embeddings')
    t0 = time.time()
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
    x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)
    log('[ENTITY_KNN] standardization done in {:.2f}s'.format(time.time() - t0))

    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10
    log('[ENTITY_KNN] n_neighbors={}'.format(n_neighbors))

    log('[ENTITY_KNN] fitting NearestNeighbors on x_train')
    t0 = time.time()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)
    log('[ENTITY_KNN] fit done in {:.2f}s'.format(time.time() - t0))

    q_tag = str(primary_quantile).replace('.', 'p')
    save_dict_path = './eval_result/distance_save_{}_trainq{}.pkl'.format(dataset, q_tag)
    log('[ENTITY_KNN] distance cache path={}'.format(save_dict_path))

    if not os.path.exists(save_dict_path):
        log('[ENTITY_KNN] no cache found; computing distances')

        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        train_sample_size = min(50000, x_train.shape[0])
        train_sample_idx = idx[:train_sample_size]

        log('[ENTITY_KNN] computing train calibration distances: sample_size={}'.format(train_sample_size))
        t0 = time.time()
        train_distances, _ = nbrs.kneighbors(x_train[train_sample_idx], n_neighbors=n_neighbors)
        log('[ENTITY_KNN] train calibration distances done in {:.2f}s'.format(time.time() - t0))

        mean_distance = train_distances.mean()
        train_scores = train_distances.mean(axis=1) / mean_distance
        log('[ENTITY_KNN] mean_distance={}'.format(mean_distance))
        log('[ENTITY_KNN] train_score min={:.6f}, max={:.6f}, mean={:.6f}'.format(
            float(np.min(train_scores)), float(np.max(train_scores)), float(np.mean(train_scores))
        ))

        del train_distances
        del x_train

        log('[ENTITY_KNN] computing test distances for {} samples'.format(x_test.shape[0]))
        t0 = time.time()
        test_distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        log('[ENTITY_KNN] test distances done in {:.2f}s'.format(time.time() - t0))

        score = test_distances.mean(axis=1) / mean_distance
        del test_distances

        save_dict = {
            'mean_distance': mean_distance,
            'train_scores': train_scores,
            'score': score,
            'primary_quantile': primary_quantile,
            'report_quantiles': report_quantiles,
            'n_neighbors': n_neighbors,
            'threshold_policy': 'train_score_quantile',
        }

        log('[ENTITY_KNN] saving distance cache')
        t0 = time.time()
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
        log('[ENTITY_KNN] cache saved in {:.2f}s'.format(time.time() - t0))
    else:
        log('[ENTITY_KNN] loading cached distances')
        t0 = time.time()
        with open(save_dict_path, 'rb') as f:
            save_dict = pkl.load(f)
        mean_distance = save_dict['mean_distance']
        train_scores = save_dict['train_scores']
        score = save_dict['score']
        log('[ENTITY_KNN] cache loaded in {:.2f}s'.format(time.time() - t0))

    log('[ENTITY_KNN] score min={:.6f}, max={:.6f}, mean={:.6f}'.format(
        float(np.min(score)), float(np.max(score)), float(np.mean(score))
    ))

    log('[ENTITY_KNN] computing ROC-AUC and PR curve')
    t0 = time.time()
    auc = roc_auc_score(y_test, score)
    prec_curve, rec_curve, threshold_curve = precision_recall_curve(y_test, score)
    f1_curve = 2 * prec_curve * rec_curve / (rec_curve + prec_curve + 1e-9)
    log('[ENTITY_KNN] metrics curves computed in {:.2f}s'.format(time.time() - t0))
    log('[ENTITY_KNN] len(score)={}, len(threshold_curve)={}, len(f1_curve)={}'.format(
        len(score), len(threshold_curve), len(f1_curve)
    ))

    # This is reported only as diagnostic separability information. It is not
    # used for the final test-set Precision/Recall/F1 because it uses y_test.
    if len(threshold_curve) > 0:
        valid_f1 = f1_curve[:len(threshold_curve)]
        oracle_idx = int(np.argmax(valid_f1))
        print('ORACLE_MAX_F1_DEBUG_ONLY: threshold={} precision={} recall={} f1={}'.format(
            threshold_curve[oracle_idx], prec_curve[oracle_idx], rec_curve[oracle_idx], valid_f1[oracle_idx]
        ), flush=True)

    def confusion_at_threshold(thres):
        y_pred_local = (score >= thres).astype(int)
        tp = int(np.sum((y_test == 1.0) & (y_pred_local == 1)))
        fn = int(np.sum((y_test == 1.0) & (y_pred_local == 0)))
        tn = int(np.sum((y_test == 0.0) & (y_pred_local == 0)))
        fp = int(np.sum((y_test == 0.0) & (y_pred_local == 1)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        return y_pred_local, precision, recall, f1_score, tn, fn, tp, fp

    log('[ENTITY_KNN] evaluating fixed train-quantile operating points')
    primary_result = None
    primary_threshold = None

    for q in report_quantiles:
        thres = float(np.quantile(train_scores, q))
        y_pred_q, precision_q, recall_q, f1_q, tn_q, fn_q, tp_q, fp_q = confusion_at_threshold(thres)
        print(
            'OPERATING_POINT: threshold_source=train_scores quantile={} threshold={} '
            'precision={} recall={} f1={} TN={} FN={} TP={} FP={}'.format(
                q, thres, precision_q, recall_q, f1_q, tn_q, fn_q, tp_q, fp_q
            ),
            flush=True
        )
        if abs(q - primary_quantile) < 1e-12:
            primary_result = (y_pred_q, precision_q, recall_q, f1_q, tn_q, fn_q, tp_q, fp_q)
            primary_threshold = thres

    if primary_result is None:
        primary_threshold = float(np.quantile(train_scores, primary_quantile))
        primary_result = confusion_at_threshold(primary_threshold)

    y_pred, precision, recall, f1_score, tn, fn, tp, fp = primary_result

    print('THRESHOLD_SELECTION: train_quantile_{}'.format(primary_quantile), flush=True)
    print('BEST_THRESHOLD: {}'.format(primary_threshold), flush=True)
    print('BEST_INDEX: {}'.format('N/A_train_quantile'), flush=True)
    print('AUC: {}'.format(auc), flush=True)
    print('F1: {}'.format(f1_score), flush=True)
    print('PRECISION: {}'.format(precision), flush=True)
    print('RECALL: {}'.format(recall), flush=True)
    print('TN: {}'.format(tn), flush=True)
    print('FN: {}'.format(fn), flush=True)
    print('TP: {}'.format(tp), flush=True)
    print('FP: {}'.format(fp), flush=True)
    log('[ENTITY_KNN] done')

    return auc, 0.0, y_pred, y_test
