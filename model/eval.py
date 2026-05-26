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
    log('[ENTITY_KNN] start dataset={}'.format(dataset))
    log('[ENTITY_KNN] input x_train shape={}, x_test shape={}, y_test shape={}, positives={}'.format(
        x_train.shape, x_test.shape, y_test.shape, int(y_test.sum())
    ))

    t0 = time.time()
    log('[ENTITY_KNN] standardizing embeddings')
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
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

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    log('[ENTITY_KNN] distance cache path={}'.format(save_dict_path))

    if not os.path.exists(save_dict_path):
        log('[ENTITY_KNN] no cache found; computing distances')
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)

        train_sample_size = min(50000, x_train.shape[0])
        log('[ENTITY_KNN] computing train sample distances: sample_size={}'.format(train_sample_size))
        t0 = time.time()
        distances, _ = nbrs.kneighbors(x_train[idx][:train_sample_size], n_neighbors=n_neighbors)
        log('[ENTITY_KNN] train sample distances done in {:.2f}s'.format(time.time() - t0))

        del x_train
        mean_distance = distances.mean()
        log('[ENTITY_KNN] mean_distance={}'.format(mean_distance))
        del distances

        log('[ENTITY_KNN] computing test distances for {} samples'.format(x_test.shape[0]))
        t0 = time.time()
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        log('[ENTITY_KNN] test distances done in {:.2f}s'.format(time.time() - t0))

        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)

        log('[ENTITY_KNN] saving distance cache')
        t0 = time.time()
        os.makedirs(os.path.dirname(save_dict_path), exist_ok=True)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
        log('[ENTITY_KNN] cache saved in {:.2f}s'.format(time.time() - t0))
    else:
        log('[ENTITY_KNN] loading cached distances')
        t0 = time.time()
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
        log('[ENTITY_KNN] cache loaded in {:.2f}s'.format(time.time() - t0))

    log('[ENTITY_KNN] computing anomaly scores')
    score = distances / mean_distance
    del distances

    log('[ENTITY_KNN] computing ROC-AUC and PR curve')
    t0 = time.time()
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    log('[ENTITY_KNN] metrics curves computed in {:.2f}s'.format(time.time() - t0))
    log('[ENTITY_KNN] len(score)={}, len(threshold)={}, len(f1)={}'.format(len(score), len(threshold), len(f1)))

    # NOTE:
    # precision_recall_curve returns len(threshold) == len(prec) - 1.
    # The final precision/recall point has no corresponding threshold, so all
    # threshold-based selections must use only f1[:len(threshold)].
    #
    # The original MAGIC code hard-coded recall targets for trace/theia/cadets
    # to reproduce the paper's peak performance. New entity-level datasets such
    # as fivedirections do not match any of those branches; previously best_idx
    # stayed -1, which selected the last threshold and could make recall/F1 zero.
    if len(threshold) == 0:
        raise ValueError('precision_recall_curve returned no thresholds; check y_test and score.')

    valid_f1 = f1[:len(threshold)]

    recall_targets = {
        'trace': 0.99979,
        'theia': 0.99996,
        'cadets': 0.9976,
    }

    threshold_selection = 'max_f1'
    best_idx = -1

    if dataset in recall_targets:
        # Keep the original paper-reproduction behavior for known datasets.
        target_recall = recall_targets[dataset]
        for i in range(len(threshold)):
            if rec[i] < target_recall:
                best_idx = max(i - 1, 0)
                threshold_selection = 'target_recall_{}'.format(target_recall)
                break

    if best_idx < 0:
        # Safe fallback for new datasets, e.g. fivedirections.
        # This uses labels to choose a threshold, so it is appropriate for
        # debugging/model-separability checks. For final strict evaluation,
        # choose this threshold on a validation/calibration split instead.
        best_idx = int(np.argmax(valid_f1))
        if dataset not in recall_targets:
            threshold_selection = 'max_f1_new_dataset'
        else:
            threshold_selection = 'max_f1_fallback'

    best_thres = threshold[best_idx]
    print('THRESHOLD_SELECTION: {}'.format(threshold_selection), flush=True)
    print('BEST_THRESHOLD: {}'.format(best_thres), flush=True)
    print('BEST_INDEX: {}'.format(best_idx), flush=True)

    log('[ENTITY_KNN] building confusion matrix counts')
    t0 = time.time()
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
    log('[ENTITY_KNN] confusion counts done in {:.2f}s'.format(time.time() - t0))

    print('AUC: {}'.format(auc), flush=True)
    print('F1: {}'.format(f1[best_idx]), flush=True)
    print('PRECISION: {}'.format(prec[best_idx]), flush=True)
    print('RECALL: {}'.format(rec[best_idx]), flush=True)
    print('TN: {}'.format(tn), flush=True)
    print('FN: {}'.format(fn), flush=True)
    print('TP: {}'.format(tp), flush=True)
    print('FP: {}'.format(fp), flush=True)
    y_pred = (score >= best_thres).astype(int)
    log('[ENTITY_KNN] done')
    return auc, 0.0, y_pred, y_test
