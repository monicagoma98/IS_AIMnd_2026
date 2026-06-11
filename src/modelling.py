### supervised modelling: svm, xgb
### FFNN
### XLSTM
from datetime import datetime

import audeer

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import audmetric

import os
import joblib

import utils as ut
import plots


def bootstrap_regression_metrics(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    metrics = {
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "CCC": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        y_t = y_true[idx]
        y_p = y_pred[idx]

        metrics["MAE"].append(mean_absolute_error(y_t, y_p))
        metrics["RMSE"].append(np.sqrt(mean_squared_error(y_t, y_p)))
        metrics["R2"].append(r2_score(y_t, y_p))
        metrics["CCC"].append(audmetric.concordance_cc(y_t, y_p))

    ci = {}
    for m, vals in metrics.items():
        ci[m] = {
            "mean": np.mean(vals),
            "lower_95": np.percentile(vals, 2.5),
            "upper_95": np.percentile(vals, 97.5),
        }

    return ci


def bootstrap_prediction_intervals(
    y_true,
    y_pred,
    n_bootstrap=1000,
    alpha=0.05,
    seed=42,
):
    """
    Returns lower and upper prediction intervals for each prediction.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    residuals = y_true - y_pred

    boot_residuals = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_residuals[b] = residuals[idx]

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)

    lower_err = np.percentile(boot_residuals, lower_q, axis=0)
    upper_err = np.percentile(boot_residuals, upper_q, axis=0)

    y_lower = y_pred + lower_err
    y_upper = y_pred + upper_err

    return y_lower, y_upper


def clinical_error_analysis(y_true, y_pred, thresholds=(2, 3, 5)):
    abs_err = np.abs(y_true - y_pred)

    results = {}
    for t in thresholds:
        results[f"within_±{t}"] = np.mean(abs_err <= t)

    return results


def evaluation_regression(X_test, y_test, y_pred, save_dir, model, seed):

    ######
    # session-level metrics: this is also not considering the different vads!!!!
    ######
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    pearson_r, pearson_p = pearsonr(y_test, y_pred)
    ccc_session = audmetric.concordance_cc(y_test, y_pred)

    # Bootstrap confidence intervals
    y_lower, y_upper = bootstrap_prediction_intervals(
        y_true=y_test,
        y_pred=y_pred,
        n_bootstrap=1000,
        seed=seed,
    )

    ci_metrics = bootstrap_regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
        n_bootstrap=1000,
        seed=seed,
    )

    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

    clinical = clinical_error_analysis(y_test, y_pred)

    #########
    # speaker-level metrics
    #########
    spk_df = X_test.copy()
    spk_df.reset_index(inplace=True)
    spk_df = spk_df[["file", "session", "IDs"]]
    spk_df["references"] = y_test
    spk_df["predictions"] = y_pred

    spk_eval = spk_df.groupby(["IDs", "session"]).agg(
        {
            "references": lambda x: list(x),
            "predictions": lambda x: list(x),
        }
    )
    print(spk_eval)
    # average predictions and references per speaker
    spk_eval["agg_ref"] = spk_eval["references"].apply(lambda x: np.mean(x))
    spk_eval["agg_pred"] = spk_eval["predictions"].apply(lambda x: np.mean(x))
    spk_eval["abs_error"] = np.abs(spk_eval["agg_ref"] - spk_eval["agg_pred"])

    ref_spk = spk_eval["agg_ref"].to_numpy()
    pred_spk = spk_eval["agg_pred"].to_numpy()
    mse_spk = mean_squared_error(ref_spk, pred_spk)
    rmse_spk = np.sqrt(mse_spk)
    mae_spk = mean_absolute_error(ref_spk, pred_spk)
    r2_spk = r2_score(ref_spk, pred_spk)
    pearson_r_spk, pearson_p_spk = pearsonr(ref_spk, pred_spk)
    ccc_spk = audmetric.concordance_cc(ref_spk, pred_spk)

    save_dir_plots = audeer.mkdir(f"{save_dir}/plots")
    plots.plot_per_speaker_trajectories(spk_eval, model=model, odir=save_dir_plots)
    plots.plot_error_vs_session_with_sem(spk_eval, model=model, odir=save_dir_plots)
    plots.plot_predicted_vs_reference(spk_eval, model=model, odir=save_dir_plots)
    plots.plot_population_trajectory(spk_eval, model=model, odir=save_dir_plots)

    speaker_clinical = clinical_error_analysis(ref_spk, pred_spk)

    speaker_ci_metrics = bootstrap_regression_metrics(
        y_true=ref_spk, y_pred=pred_spk, n_bootstrap=1000, seed=seed
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_file = f"{save_dir}/{model}_results.txt"

    with open(out_file, "w") as f:
        f.write("======================\n")
        f.write(f"{model} Regression Results Session-Level\n")
        f.write("======================\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Point Estimates:\n")
        f.write(f"MAE  : {mae:.3f}\n")
        f.write(f"RMSE : {rmse:.3f}\n")
        f.write(f"R2   : {r2:.3f}\n\n")
        f.write(f"Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})\n\n")
        f.write(f"CCC: {ccc_session:.3f}\n\n")

        f.write("95% Confidence Intervals (Bootstrap):\n")
        for m, stats in ci_metrics.items():
            f.write(
                f"{m}: {stats['mean']:.3f} "
                f"[{stats['lower_95']:.3f}, {stats['upper_95']:.3f}]\n"
            )
        f.write("\n")

        f.write("Clinical Accuracy:\n")
        for k, v in clinical.items():
            f.write(f"{k}: {v*100:.1f}%\n")
        f.write("\n")

        f.write("Prediction Interval:\n")
        f.write(f"Nominal coverage: 95%\n")
        f.write(f"Empirical coverage: {coverage*100:.1f}%\n")
        f.write("\n")
        f.write("======================\n")
        f.write(f"{model} Regression Results Speaker-Level\n")
        f.write("======================\n\n")
        f.write("Point Estimates speaker-level:\n")
        f.write(f"MAE  : {mae_spk:.3f}\n")
        f.write(f"RMSE : {rmse_spk:.3f}\n")
        f.write(f"R2   : {r2_spk:.3f}\n\n")
        f.write(f"Pearson r: {pearson_r_spk:.3f} (p={pearson_p_spk:.3e})\n\n")
        f.write(f"CCC: {ccc_spk:.3f}\n\n")

        f.write("95% Confidence Intervals (Bootstrap) Speaker-level:\n")
        for m, stats in speaker_ci_metrics.items():
            f.write(
                f"{m}: {stats['mean']:.3f} "
                f"[{stats['lower_95']:.3f}, {stats['upper_95']:.3f}]\n"
            )
        f.write("\n")

        f.write("Speaker-Level Clinical Accuracy:\n")
        for k, v in speaker_clinical.items():
            f.write(f"{k}: {v*100:.1f}%\n")

    return print(f"Evaluation results saved in {out_file}.")


def main_supervised_modeling(
    train, test, X_col, target, save_dir, n_jobs, seed, normalisation
):
    if seed:
        ut.set_seed(seed)

    n_jobs = 1

    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    # input(train["file"])
    train.set_index(["file", "session", "IDs"], inplace=True)
    test.set_index(["file", "session", "IDs"], inplace=True)

    # print(train)
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    X_train = train[X_col].copy()
    X_test = test[X_col].copy()

    # print(y_train)
    # input(X_test)

    if normalisation == "standard":
        scaler = StandardScaler()
    elif normalisation == "robust":
        scaler = RobustScaler(quantile_range=(10, 90))

    scaler.fit(X_train)
    X_train = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # SVM modeling
    print("Starting SVM modeling...")
    y_pred_svm = svm_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed)

    # XGB modeling
    print("Starting XGB modeling...")
    y_pred_xgb = xgb_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed=seed)
    
    # RF modeling
    print("Starting RF modeling...")
    y_pred_rf = rf_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed)

    ## MLNN modeling
    #print("Starting FFNN modeling...")
    #y_pred_ffnn = ffnn_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed)

    # save predictions
    ut.save_predictions(X_test, y_pred_svm, y_test, save_dir, "svm")
    ut.save_predictions(X_test, y_pred_xgb, y_test, save_dir, "xgb")
    ut.save_predictions(X_test, y_pred_rf, y_test, save_dir, "rf")
    

    # Evaluation SVM
    print("Evaluating SVM model...")
    evaluation_regression(X_test, y_test, y_pred_svm, save_dir, "svm", seed)

    ## Evaluation XGB
    print("Evaluating XGB model...")
    evaluation_regression(X_test, y_test, y_pred_xgb, save_dir, "xgb", seed)

    ## Evaluation RF
    print("Evaluating RF model...")
    evaluation_regression(X_test, y_test, y_pred_rf, save_dir, "rf", seed)
    return


def svm_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed):

    audeer.mkdir(f"{save_dir}/model")
    model_file = f"{save_dir}/model/model_svm.sav"

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)

    if os.path.exists(model_file):
        mod = joblib.load(model_file)
    else:
        base = svm.SVR()
        scoring = "neg_mean_squared_error"

        param_dist = {
            "C": np.logspace(-2, 2, 20),
            "epsilon": [0.05, 0.1, 0.2, 0.5],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=30,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=seed,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train)

        mod = search.best_estimator_

        # save model
        joblib.dump(mod, model_file)
        del search  # Remove RandomizedSearchCV object

        # Free memory
        import gc

        # del X_train, y_train
        gc.collect()

    # predictions

    y_pred = mod.predict(X_test)

    return y_pred


def xgb_modeling(X_train, X_test, y_train, save_dir, n_jobs, task=None, seed=None):

    audeer.mkdir(f"{save_dir}/model")
    model_file = f"{save_dir}/model/model_xgb.sav"

    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=2,  # reduced like SVM
        random_state=seed,
    )

    if os.path.exists(model_file):
        best_xgb_model = joblib.load(model_file)
    else:
        scoring = "neg_mean_squared_error"
        model = xgboost.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )

        param_dist = {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.6, 0.8],
            "colsample_bytree": [0.6, 0.8],
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=30,  # key speedup
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=seed,
            verbose=1,
        )

        search.fit(X_train, y_train)

        # best_params = grid_search.best_params_

        best_xgb_model = search.best_estimator_
        joblib.dump(best_xgb_model, model_file)

        del search  # Remove RandomizedSearchCV object

    y_pred = best_xgb_model.predict(X_test)

    return y_pred

def rf_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed):
    
    audeer.mkdir(f"{save_dir}/model")
    model_file = f"{save_dir}/model/model_rf.sav"

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)

    if os.path.exists(model_file):
        best_rf_model = joblib.load(model_file)
    else:
    
        scoring = "neg_mean_squared_error"
        model = RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
        )

        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=30,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=seed,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_rf_model = search.best_estimator_
        joblib.dump(best_rf_model, model_file)

        del search

    y_pred = best_rf_model.predict(X_test)

    return y_pred

def ffnn_modeling(X_train, X_test, y_train, save_dir, n_jobs, seed):

    audeer.mkdir(f"{save_dir}/model")
    model_file = f"{save_dir}/model/model_ffnn.sav"

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)

    if os.path.exists(model_file):
        best_ffnn_model = joblib.load(model_file)
    else:
        scoring = "neg_mean_squared_error"
        model = MLPRegressor(
            random_state=seed,
            max_iter=500,
        )
        n_features = X_train.shape[1]

        param_dist = {
            "hidden_layer_sizes": [
                (n_features,),
                (n_features * 2,),
                (n_features, n_features // 2),
                (n_features * 2, n_features),
                (n_features * 2, n_features, n_features // 2),
                (128, 64),
                (256, 128, 64),
            ],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "alpha": [0.00001, 0.0001, 0.001],  # L2 regularization
            "solver": ["adam"],
            "batch_size": ["auto", 32, 64],
            "early_stopping": [True],
            "validation_fraction": [0.1],
        }
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=30,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=seed,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_ffnn_model = search.best_estimator_
        joblib.dump(best_ffnn_model, model_file)

        del search

    y_pred = best_ffnn_model.predict(X_test)

    return y_pred
