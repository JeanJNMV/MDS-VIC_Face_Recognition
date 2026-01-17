from vic.dataloader import make_fixed_test_indices, split_with_fixed_test
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np


def get_metrics_vs_train_size(model, train_sizes, data, n_test=3, seed=0):
    test_idx, pool_idx = make_fixed_test_indices(data, n_test=n_test, seed=seed)
    accuracy_scores = {}
    conf_matrices = {}

    for train_size in train_sizes:
        Xtr, ytr, Xte, yte = split_with_fixed_test(
            data, test_idx, pool_idx, train_size, seed=seed
        )
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        accuracy_scores[train_size] = accuracy_score(yte, y_pred)
        conf_matrices[train_size] = confusion_matrix(yte, y_pred)

    return accuracy_scores, conf_matrices


def get_average_acc_vs_train_size(
    model, train_sizes, data, n_exp=10, n_test=3, seed_master=0
):
    master_rng = np.random.default_rng(seed_master)

    avg_accuracy_scores = {}

    for i in tqdm(range(n_exp), desc="Experiments"):
        seed = master_rng.integers(0, 1e6)

        accuracy_scores, _ = get_metrics_vs_train_size(
            model,
            train_sizes,
            data,
            n_test=n_test,
            seed=seed,
        )

        for train_size, acc in accuracy_scores.items():
            avg_accuracy_scores.setdefault(train_size, []).append(acc)

    mean_accuracy_scores = {
        size: np.mean(accs) for size, accs in avg_accuracy_scores.items()
    }
    std_accuracy_scores = {
        size: np.std(accs) for size, accs in avg_accuracy_scores.items()
    }

    return mean_accuracy_scores, std_accuracy_scores
