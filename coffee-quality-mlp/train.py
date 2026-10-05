# 1. 导入库和基本设置
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Notebook 中使用富文本显示；普通 Python 环境没有 IPython 时退回 print。
try:
    from IPython.display import display
except ImportError:
    display = print

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

SEED = 42
TARGET = "Total.Cup.Points"
HIGH_QUALITY_THRESHOLD = 82.5
DATASET_HANDLE = "volpatto/coffee-quality-database-from-cqi"

random.seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

PROJECT_DIR = Path.cwd()
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "results_notebook"
PLOT_DIR = OUTPUT_DIR / "plots"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 3. 定义使用和排除的特征

# 十项杯测子分数几乎直接相加得到目标，禁止进入模型。
LEAKAGE_COLUMNS = [
    "Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
    "Balance", "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points",
]

# 7 个数值输入。
NUMERIC_FEATURES = [
    "Number.of.Bags",
    "Moisture",
    "Category.One.Defects",
    "Quakers",
    "Category.Two.Defects",
    "altitude_mean_meters",
    "Harvest.Year.Numeric",
]

# 6 个类别输入。
CATEGORICAL_FEATURES = [
    "Species",
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
    "unit_of_measurement",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# 这些字段保留供复核，但不直接进入当前模型。
EXCLUDED_METADATA_COLUMNS = [
    "Owner", "Farm.Name", "Lot.Number", "Mill", "ICO.Number", "Company",
    "Altitude", "Region", "Producer", "Bag.Weight", "In.Country.Partner",
    "Harvest.Year", "Grading.Date", "Owner.1", "Expiration",
    "Certification.Body", "Certification.Address", "Certification.Contact",
    "altitude_low_meters", "altitude_high_meters",
]

feature_plan = pd.DataFrame(
    {
        "feature_group": [
            "Numeric inputs",
            "Categorical inputs",
            "Target leakage",
            "Other excluded metadata",
        ],
        "count": [7, 6, 10, len(EXCLUDED_METADATA_COLUMNS)],
        "processing": [
            "Median imputation + RobustScaler",
            "Mode imputation + One-Hot Encoding",
            "Audit only; excluded from training",
            "Retained for audit; excluded from X",
        ],
    }
)
display(feature_plan)

# 4. 定位或下载数据（不在代码中保存 Token）
def locate_or_download_data() -> Path:
    # 先查找本地 data/raw。
    matches = sorted(RAW_DATA_DIR.rglob("merged_data_cleaned.csv"))
    if matches:
        return matches[0]

    # 再查找 Kaggle Notebook 附加的数据。
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob("merged_data_cleaned.csv"))
        if matches:
            return matches[0]

    # 仍未找到时，尝试下载公开数据集。
    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            "Place merged_data_cleaned.csv in data/raw/ or install kagglehub."
        ) from exc

    download_path = Path(
        kagglehub.dataset_download(DATASET_HANDLE, output_dir=str(RAW_DATA_DIR))
    )
    for root in (download_path, RAW_DATA_DIR):
        matches = sorted(root.rglob("merged_data_cleaned.csv"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "The dataset was downloaded, but merged_data_cleaned.csv was not found."
    )


CSV_PATH = locate_or_download_data()
print("Data file:", CSV_PATH.resolve())

# 5. 原始数据初检
df_raw = pd.read_csv(CSV_PATH)

print("Original data shape:", df_raw.shape)  # 正式数据应为 1339×44。
display(df_raw.head())

missing_before = df_raw.isna().sum().sort_values(ascending=False)
display(missing_before.head(20).rename("missing_count").to_frame())
print("Exact duplicate rows in raw data:", int(df_raw.duplicated().sum()))

# 6. 数据清洗
def save_json(data: dict, path: Path) -> None:
    # 保存包含 NumPy 数值的 JSON。
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def load_and_clean(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    original_shape = df.shape

    # 保存清洗前缺失值统计。
    df.isna().sum().sort_values(ascending=False).rename("missing_count").to_csv(
        OUTPUT_DIR / "missing_values_before.csv"
    )

    # 源行号只用于追踪，不作为特征。
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "record_id"})
    else:
        df.insert(0, "record_id", np.arange(len(df)))

    # 判断重复时忽略 record_id。
    duplicate_subset = [column for column in df.columns if column != "record_id"]
    duplicate_count = int(df.duplicated(subset=duplicate_subset).sum())
    df = df.drop_duplicates(subset=duplicate_subset).copy()

    # 从 2014/2015 等文本中提取第一个四位年份。
    df["Harvest.Year.Numeric"] = pd.to_numeric(
        df["Harvest.Year"]
        .astype("string")
        .str.extract(r"((?:19|20)\d{2})", expand=False),
        errors="coerce",
    )
    invalid_year = df["Harvest.Year.Numeric"].notna() & ~df[
        "Harvest.Year.Numeric"
    ].between(1990, 2030)
    invalid_year_count = int(invalid_year.sum())
    df.loc[invalid_year, "Harvest.Year.Numeric"] = np.nan

    # 超过 0—5000 米的海拔视为录入/单位错误，只设为缺失，不删整行。
    invalid_altitude = df["altitude_mean_meters"].notna() & ~df[
        "altitude_mean_meters"
    ].between(0, 5000)
    invalid_altitude_count = int(invalid_altitude.sum())
    df.loc[invalid_altitude, "altitude_mean_meters"] = np.nan

    # 目标是训练答案，缺失、≤0 或 >100 时不能填补，必须删除。
    invalid_target = df[TARGET].isna() | ~df[TARGET].between(0.000001, 100)
    invalid_target_count = int(invalid_target.sum())
    df = df.loc[~invalid_target].reset_index(drop=True)

    # 验证杯测子分数和源行号的数据泄漏风险。
    component_sum = df[LEAKAGE_COLUMNS].sum(axis=1)
    not_used = [
        column
        for column in df.columns
        if column not in MODEL_FEATURES and column != TARGET
    ]

    audit = {
        "original_shape": list(original_shape),
        "rows_after_cleaning": len(df),
        "duplicate_rows_removed": duplicate_count,
        "invalid_year_values_set_to_missing": invalid_year_count,
        "invalid_altitude_values_set_to_missing": invalid_altitude_count,
        "invalid_target_rows_removed": invalid_target_count,
        "target_component_sum_correlation": float(df[TARGET].corr(component_sum)),
        "target_component_sum_mean_absolute_difference": float(
            (df[TARGET] - component_sum).abs().mean()
        ),
        "source_index_target_correlation": float(df[TARGET].corr(df["record_id"])),
        "model_features": MODEL_FEATURES,
        "columns_not_used_by_model": not_used,
    }

    save_json(audit, OUTPUT_DIR / "cleaning_audit.json")
    df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)
    df.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "summary_statistics.csv"
    )
    return df, audit


df, cleaning_audit = load_and_clean(CSV_PATH)
print("Data shape after cleaning:", df.shape)  # 正式数据应为 1338 行。
display(pd.Series(cleaning_audit, name="value").to_frame())

# 7. 基础数据分析
target_summary = pd.Series(
    {
        "count": df[TARGET].count(),
        "mean": df[TARGET].mean(),
        "median": df[TARGET].median(),
        "variance": df[TARGET].var(),
        "standard_deviation": df[TARGET].std(),
        "minimum": df[TARGET].min(),
        "q1": df[TARGET].quantile(0.25),
        "q3": df[TARGET].quantile(0.75),
        "maximum": df[TARGET].max(),
    },
    name=TARGET,
)
target_summary.to_csv(OUTPUT_DIR / "target_summary.csv")
display(target_summary.to_frame())

# 目标分布。
plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=30, kde=True, color="#6f4e37")
plt.axvline(df[TARGET].mean(), color="red", linestyle="--", label="Mean")
plt.axvline(df[TARGET].median(), color="navy", linestyle=":", label="Median")
plt.title("Distribution of Total Cup Points")
plt.xlabel("Total Cup Points")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_target_distribution.png", dpi=160)
plt.show()

# 记录最多的十个国家。
top_countries = df["Country.of.Origin"].value_counts().head(10).sort_values()
plt.figure(figsize=(9, 6))
top_countries.plot(kind="barh", color="#8b5a2b")
plt.title("Top 10 Countries by Number of Records")
plt.xlabel("Number of records")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_top_countries.png", dpi=160)
plt.show()

# 泄漏列只用于相关性审计，不进入模型。
corr_columns = LEAKAGE_COLUMNS + [
    TARGET,
    "Moisture",
    "Category.One.Defects",
    "Category.Two.Defects",
    "altitude_mean_meters",
]
plt.figure(figsize=(12, 9))
sns.heatmap(df[corr_columns].corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_correlation_heatmap.png", dpi=160)
plt.show()

# 清洗后缺失最多的十五列。
missing = df.isna().sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x=missing.values, y=missing.index, color="#4c72b0")
plt.title("Top 15 Columns with Missing Values")
plt.xlabel("Missing values")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_missing_values.png", dpi=160)
plt.show()

# 8. 建立无泄漏预处理和 MLP
def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # 未列出的字段全部不进入模型。
    )


def make_model(hidden_layers: tuple[int, ...]) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=SEED,
                ),
            ),
        ]
    )
    # 目标也标准化，预测时自动还原成原始杯测分数。
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(y_true - y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "within_1_point_accuracy": float(np.mean(absolute_error <= 1.0)),
        "within_2_points_accuracy": float(np.mean(absolute_error <= 2.0)),
    }

# 9. 60% train / 20% validation / 20% sealed test
def split_indices(y: pd.Series):
    indices = np.arange(len(y))

    # 连续目标先按排名分成五个区间，再用于分层。
    bins_all = pd.qcut(y.rank(method="first"), q=5, labels=False)
    development_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=bins_all,
    )

    development_bins = pd.qcut(
        y.iloc[development_idx].rank(method="first"), q=5, labels=False
    )
    train_idx, val_idx = train_test_split(
        development_idx,
        test_size=0.25,  # 80%×25%=总数据20%。
        random_state=SEED,
        stratify=development_bins,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


# X 只包含明确批准的 13 个字段。
X = df[MODEL_FEATURES].copy()
y = df[TARGET].astype(float)
train_idx, val_idx, test_idx = split_indices(y)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

split_summary = pd.DataFrame(
    {
        "split": ["train", "validation", "test"],
        "rows": [len(train_idx), len(val_idx), len(test_idx)],
        "target_mean": [y_train.mean(), y_val.mean(), y_test.mean()],
        "target_std": [y_train.std(), y_val.std(), y_test.std()],
    }
)
split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
display(split_summary)

# 检查三个集合没有索引重叠。
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx).isdisjoint(test_idx)
assert set(val_idx).isdisjoint(test_idx)

# 10. 只用 validation RMSE 选择隐藏层
CANDIDATE_LAYERS = [(32,), (64, 32), (128, 64, 32)]
validation_rows = []
best_layers = None
best_validation_rmse = float("inf")

for hidden_layers in CANDIDATE_LAYERS:
    model = make_model(hidden_layers)
    model.fit(X_train, y_train)  # 只在训练集拟合。
    validation_prediction = model.predict(X_val)
    metrics = regression_metrics(y_val, validation_prediction)
    validation_rows.append(
        {"hidden_layers": "-".join(map(str, hidden_layers)), **metrics}
    )

    if metrics["RMSE"] < best_validation_rmse:
        best_validation_rmse = metrics["RMSE"]
        best_layers = hidden_layers

assert best_layers is not None
validation_table = pd.DataFrame(validation_rows).sort_values("RMSE")
validation_table.to_csv(
    OUTPUT_DIR / "validation_model_selection.csv", index=False
)
display(validation_table)
print("Selected hidden layers:", best_layers)

# 11. 锁定结构后用 train+validation 重训，最后测试一次
development_idx = np.sort(np.concatenate([train_idx, val_idx]))

final_model = make_model(best_layers)
final_model.fit(X.iloc[development_idx], y.iloc[development_idx])

# sealed test 到这里才第一次参与预测。
test_prediction = final_model.predict(X_test)
test_metrics = regression_metrics(y_test, test_prediction)

# 均值基线：所有测试记录都预测开发数据的目标均值。
baseline_prediction = np.full(len(test_idx), y.iloc[development_idx].mean())
baseline_metrics = regression_metrics(y_test, baseline_prediction)

display(
    pd.DataFrame(
        [baseline_metrics, test_metrics],
        index=["Mean baseline", "Final MLP"],
    )
)

# 12. 保存预测并计算 82.5 分派生筛选指标
test_predictions = pd.DataFrame(
    {
        "record_id": df.iloc[test_idx]["record_id"].to_numpy(),
        "actual_total_cup_points": y_test.to_numpy(),
        "predicted_total_cup_points": test_prediction,
    }
)
test_predictions["error"] = (
    test_predictions["predicted_total_cup_points"]
    - test_predictions["actual_total_cup_points"]
)
test_predictions["absolute_error"] = test_predictions["error"].abs()
test_predictions["actual_high_quality"] = (
    test_predictions["actual_total_cup_points"] >= HIGH_QUALITY_THRESHOLD
).astype(int)
test_predictions["predicted_high_quality"] = (
    test_predictions["predicted_total_cup_points"] >= HIGH_QUALITY_THRESHOLD
).astype(int)
test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

actual_high = test_predictions["actual_high_quality"]
predicted_high = test_predictions["predicted_high_quality"]
screening_confusion = confusion_matrix(actual_high, predicted_high)
screening_metrics = {
    "threshold": HIGH_QUALITY_THRESHOLD,
    "accuracy": float(accuracy_score(actual_high, predicted_high)),
    "precision": float(
        precision_score(actual_high, predicted_high, zero_division=0)
    ),
    "recall": float(recall_score(actual_high, predicted_high, zero_division=0)),
    "F1": float(f1_score(actual_high, predicted_high, zero_division=0)),
    "confusion_matrix_TN_FP_FN_TP": screening_confusion.ravel().tolist(),
}

print("Sealed-test regression metrics:")
display(pd.Series(test_metrics, name="Final MLP").to_frame())
print("Quality-screening metrics at 82.5 points:")
display(pd.Series(screening_metrics, name="value").to_frame())

# 13. 查看 One-Hot 后实际进入 MLP 的特征
fitted_preprocessor = final_model.regressor_.named_steps["preprocess"]
processed_feature_names = fitted_preprocessor.get_feature_names_out()

print("Original model input fields:", len(MODEL_FEATURES))
print("Processed columns entering the MLP:", len(processed_feature_names))

processed_features = pd.DataFrame(
    {"processed_feature": processed_feature_names}
)
processed_features.to_csv(
    OUTPUT_DIR / "processed_feature_names.csv", index=False
)
display(processed_features)

# 14. 最终评价图
final_mlp = final_model.regressor_.named_steps["mlp"]

# 训练损失。
plt.figure(figsize=(8, 5))
plt.plot(final_mlp.loss_curve_, color="#6f4e37")
plt.title("Final MLP Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_training_loss.png", dpi=160)
plt.show()

# 真实值与预测值。
low = min(y_test.min(), test_prediction.min()) - 1
high = max(y_test.max(), test_prediction.max()) + 1
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=test_prediction, alpha=0.7, color="#6f4e37")
plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
plt.xlim(low, high)
plt.ylim(low, high)
plt.xlabel("Actual Total Cup Points")
plt.ylabel("Predicted Total Cup Points")
plt.title("Actual vs Predicted on Sealed Test Set")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_actual_vs_predicted.png", dpi=160)
plt.show()

# 测试残差。
plt.figure(figsize=(8, 5))
sns.histplot(test_predictions["error"], bins=25, kde=True, color="#4c72b0")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Prediction error (predicted - actual)")
plt.title("Test Residual Distribution")
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_test_residuals.png", dpi=160)
plt.show()

# High/Standard 混淆矩阵。
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(
    screening_confusion,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Standard", "High"],
    yticklabels=["Standard", "High"],
)
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title(f"Quality Screening at {HIGH_QUALITY_THRESHOLD} Points")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_quality_screening_confusion_matrix.png", dpi=160)
plt.show()

# 15. 保存最终模型和指标
MODEL_PATH = OUTPUT_DIR / "coffee_quality_mlp.joblib"
joblib.dump(final_model, MODEL_PATH)

final_results = {
    "random_seed": SEED,
    "prediction_target": TARGET,
    "model_features": MODEL_FEATURES,
    "processed_feature_count": len(processed_feature_names),
    "split_rows": {
        "train": len(train_idx),
        "validation": len(val_idx),
        "test": len(test_idx),
    },
    "candidate_hidden_layers": [list(item) for item in CANDIDATE_LAYERS],
    "selected_hidden_layers": list(best_layers),
    "selected_validation_RMSE": best_validation_rmse,
    "final_model_iterations": int(final_mlp.n_iter_),
    "test_metrics": test_metrics,
    "derived_high_quality_screening_metrics": screening_metrics,
    "mean_baseline_test_metrics": baseline_metrics,
}

save_json(final_results, OUTPUT_DIR / "model_metrics.json")
print("Model saved to:", MODEL_PATH.resolve())
print("Results directory:", OUTPUT_DIR.resolve())

