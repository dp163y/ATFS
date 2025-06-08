import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from scipy.stats import ttest_rel, bootstrap
from scipy.stats import pearsonr
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ====================== 对比特征选择器类 ======================
class AllFeaturesSelector:
    """使用所有特征的选择器"""
    def __init__(self):
        self.selected_features = None
        
    def fit(self, X, y=None):
        self.selected_features = X.columns.tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features]
    
class RFRFE_Selector:
    """随机森林递归特征消除选择器"""
    def __init__(self, k_features):
        self.k = k_features
        self.selected_features = None
        
    def fit(self, X, y):
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.selector = RFE(model, n_features_to_select=self.k, step=0.1)
        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.support_].tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features]
    
class LassoSelector:
    """Lasso特征选择器"""
    def __init__(self, k_features):
        self.k = k_features
        self.selected_features = None
        
    def fit(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Lasso(alpha=0.01, max_iter=5000)
        model.fit(X_scaled, y)
        coef = np.abs(model.coef_)
        top_k_indices = coef.argsort()[-self.k:][::-1]
        self.selected_features = X.columns[top_k_indices].tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features]
    
class XGBoostBuiltInSelector:
    """XGBoost内置特征重要性选择器"""
    def __init__(self, k_features):
        self.k = k_features
        self.selected_features = None
        
    def fit(self, X, y):
        scale_pos_weight = sum(y == 0) / sum(y == 1)
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X, y)
        importance = model.feature_importances_
        top_k_indices = importance.argsort()[-self.k:][::-1]
        self.selected_features = X.columns[top_k_indices].tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features]
    
class mRMRSelector:
    """mRMR特征选择器"""
    def __init__(self, k_features):
        self.k = k_features
        self.selected_features = None
        
    def fit(self, X, y):
        # 计算互信息
        mi_scores = mutual_info_classif(X, y)
        # 计算特征间相关性
        corr_matrix = np.abs(np.corrcoef(X.values.T))
        
        selected_indices = []
        remaining_indices = list(range(X.shape[1]))
        
        # 第一步：选择互信息最大的特征
        first_idx = np.argmax(mi_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # 迭代选择后续特征
        for _ in range(1, min(self.k, X.shape[1])):
            scores = []
            for idx in remaining_indices:
                # 相关性部分
                redundancy = 0.0
                for sel_idx in selected_indices:
                    redundancy += corr_matrix[idx, sel_idx]
                redundancy /= len(selected_indices)
                
                # mRMR得分：相关性 - 冗余性
                score = mi_scores[idx] - redundancy
                scores.append(score)
            
            # 选择得分最高的特征
            best_idx = remaining_indices[np.argmax(scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        self.selected_features = X.columns[selected_indices].tolist()
        return self
        
    def transform(self, X):
        return X[self.selected_features]

# ====================== ATFS类 ======================
class AdaptiveThresholdFeatureSelector:
    def __init__(self, target_pr_auc=0.90, corr_threshold=0.8, verbose=True):
        self.target_pr_auc = target_pr_auc
        self.corr_threshold = corr_threshold
        self.selected_features = None
        self.feature_ranking = None
        self.scaler = StandardScaler()
        self.verbose = verbose
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)

    def fit(self, X, y):
        """主特征选择流程"""
        if self.verbose:
            print("开始特征选择流程...")
        X_scaled = self.scaler.fit_transform(X)
        if self.verbose:
            print("完成特征标准化")
        
        # 互信息特征排序
        mi_scores = mutual_info_classif(X_scaled, y)
        self.feature_ranking = np.argsort(mi_scores)[::-1]
        ranked_features = X.columns[self.feature_ranking]
        if self.verbose:
            print(f"前10个重要特征: {ranked_features[:10].tolist()}")
        
        # 自适应特征数量选择
        if self.verbose:
            print("寻找最佳特征数量...")
        k_opt, pr_auc_scores = self._find_optimal_k(X_scaled, y, ranked_features)
        if self.verbose:
            print(f"初步选择{k_opt}个特征")
        
        k_features = ranked_features[:k_opt]
        # 特征相关性筛选
        if self.verbose:
            print("进行特征相关性筛选...")
        self.selected_features = self._remove_correlated_features(
            X_scaled[:, self.feature_ranking[:k_opt]], k_features
        )
        if self.verbose:
            print(f"最终选择特征数: {len(self.selected_features)}")
            print(f"最终所选特征: {self.selected_features}")
        return self

    def transform(self, X):
        """应用特征选择"""
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, [i for i, col in enumerate(X.columns) 
                          if col in self.selected_features]]

    def _find_optimal_k(self, X, y, ranked_features):
        """确定最佳特征数量（使用PR-AUC）"""
        k_values = []
        best_feat_num = -100
        pr_auc_scores = {'LR': [], 'SVC': [], 'DT': [], 'avg': []}
        
        for k in range(1, min(500, len(ranked_features))):
            X_sub = X[:, self.feature_ranking[:k]]
            models = {
                'LR': LogisticRegression(class_weight='balanced', max_iter=1000),
                'SVC': SVC(class_weight='balanced', probability=True),
                'DT': DecisionTreeClassifier(class_weight='balanced')
            }
            model_scores = {}
            
            for name, model in models.items():
                scores = cross_val_score(model, X_sub, y, cv=5, 
                                        scoring='average_precision', n_jobs=-1)
                model_scores[name] = np.mean(scores)
                pr_auc_scores[name].append(np.mean(scores))
            
            avg_score = np.mean(list(model_scores.values()))
            pr_auc_scores['avg'].append(avg_score)
            k_values.append(k)
            
            if avg_score > self.target_pr_auc and k > 10:
                break
            
            if len(pr_auc_scores['avg']) >= 5:
                last_n_tolerances = [
                    abs(pr_auc_scores['avg'][i] - pr_auc_scores['avg'][i-1])
                    for i in range(-5+1, 0)
                ]
                if all(t < 0.0001 for t in last_n_tolerances):
                    best_feat_num = k - 5 + 1
                    break
        
        plt.figure(figsize=(12, 8))
        for name, scores in pr_auc_scores.items():
            if name != 'avg' and scores:
                plt.plot(range(1, len(scores)+1), scores, label=f'{name} PR-AUC', 
                        marker='o', linestyle='-', alpha=0.7)
        plt.plot(k_values, pr_auc_scores['avg'], label='Average PR-AUC', 
                marker='s', linestyle='--', color='black', linewidth=2)
        plt.xlabel('Number of Features')
        plt.ylabel('Average Precision (PR-AUC)')
        plt.title('Feature Count vs PR-AUC Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.results_dir / 'feature_vs_prauc.png')
        plt.close()
        
        best_idx = np.argmax(pr_auc_scores['avg']) if best_feat_num == -100 else k_values.index(best_feat_num)
        best_k = k_values[best_idx] if best_feat_num == -100 else best_feat_num
        
        if self.verbose:
            print(f"最佳特征数量: {best_k}, 对应平均PR-AUC: {pr_auc_scores['avg'][best_idx]:.4f}")
        return best_k, pr_auc_scores

    def _remove_correlated_features(self, X, features):
        """去除高度相关特征"""
        selected = []
        remaining = list(range(len(features)))
        while remaining:
            best_idx = remaining[0]
            selected.append(features[best_idx])
            if self.verbose:
                print(f"保留特征: {features[best_idx]}")
            
            # 计算与其余特征的相关系数
            j_list = remaining[1:]
            corrs = []
            for j in j_list:
                corr, _ = pearsonr(X[:, best_idx], X[:, j])
                corrs.append(abs(corr))
            
            # 找出需移除的特征及其相关系数
            to_remove_pairs = [(j, corr) for j, corr in zip(j_list, corrs) if corr > self.corr_threshold]
            to_remove = [j for j, _ in to_remove_pairs]
            
            # 打印移除信息
            for j, corr in to_remove_pairs:
                if self.verbose:
                    print(f"移除高度相关特征: {features[j]} (与 {features[best_idx]} 相关系数: {corr:.2f})")
            remaining = [j for j in remaining[1:] if j not in to_remove]
            
        return selected

# ====================== 评估函数 ======================
def evaluate_model(model, X_test, y_test, dataset_name, verbose=True):
    """评估模型并绘制曲线"""
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]
    
    results_dir = Path('results')
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / f'{dataset_name}_pr_curve.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / f'{dataset_name}_roc_curve.png')
    plt.close()
    
    y_pred = (y_proba >= best_threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if verbose:
        print("\n评估结果:")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"最佳阈值: {best_threshold:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'best_threshold': best_threshold
    }

# ====================== 统计函数 ======================
def calculate_confidence_intervals(scores, confidence_level=0.95):
    """计算置信区间"""
    mean = np.mean(scores)
    std_err = np.std(scores, ddof=1) / np.sqrt(len(scores))
    margin = std_err * 1.96  # 95% CI
    return (mean - margin, mean + margin)

def format_ci(mean, ci):
    """格式化置信区间输出"""
    return f"{mean:.4f} ({ci[0]:.4f}-{ci[1]:.4f})"

# ====================== 主训练评估函数 ======================
def train_and_evaluate_cv(dataset_path, dataset_name, verbose=True, n_splits=5):
    """完整训练评估流程"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"正在处理数据集: {dataset_name}")
        print(f"{'='*60}")
    
    data = pd.read_csv(dataset_path)
    X = data.drop('label', axis=1)
    y = data['label']
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储所有方法的结果
    method_results = {
        'ATFS': {'pr_auc': [], 'roc_auc': [], 'f1': []},
        'All Features': {'pr_auc': [], 'roc_auc': [], 'f1': []},
        'RF-RFE': {'pr_auc': [], 'roc_auc': [], 'f1': []},
        'LASSO': {'pr_auc': [], 'roc_auc': [], 'f1': []},
        'XGBoost BuiltIn': {'pr_auc': [], 'roc_auc': [], 'f1': []},
        'mRMR': {'pr_auc': [], 'roc_auc': [], 'f1': []}
    }
    
    fold_idx = 1
    for train_idx, val_idx in kf.split(X, y):
        if verbose:
            print(f"\nFold {fold_idx} 处理中...")
            print(f"训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. ATFS方法 (自适应阈值特征选择)
        if verbose:
            print("\n[方法1] ATFS特征选择...")
        atfs_selector = AdaptiveThresholdFeatureSelector(verbose=verbose)
        atfs_selector.fit(X_train, y_train)
        X_train_atfs = atfs_selector.transform(X_train)
        X_val_atfs = atfs_selector.transform(X_val)
        k_features = len(atfs_selector.selected_features)
        if verbose:
            print(f"ATFS选择特征数: {k_features}")
        
        # 2. 所有特征
        if verbose:
            print("\n[方法2] 使用所有特征...")
        all_selector = AllFeaturesSelector()
        all_selector.fit(X_train, y_train)
        X_train_all = all_selector.transform(X_train)
        X_val_all = all_selector.transform(X_val)
        
        # 3. RF-RFE
        if verbose:
            print("\n[方法3] RF-RFE特征选择...")
        rf_rfe_selector = RFRFE_Selector(k_features=k_features)
        rf_rfe_selector.fit(X_train, y_train)
        X_train_rf = rf_rfe_selector.transform(X_train)
        X_val_rf = rf_rfe_selector.transform(X_val)
        
        # 4. LASSO
        if verbose:
            print("\n[方法4] LASSO特征选择...")
        lasso_selector = LassoSelector(k_features=k_features)
        lasso_selector.fit(X_train, y_train)
        X_train_lasso = lasso_selector.transform(X_train)
        X_val_lasso = lasso_selector.transform(X_val)
        
        # 5. XGBoost内置特征选择
        if verbose:
            print("\n[方法5] XGBoost内置特征选择...")
        xgb_selector = XGBoostBuiltInSelector(k_features=k_features)
        xgb_selector.fit(X_train, y_train)
        X_train_xgb = xgb_selector.transform(X_train)
        X_val_xgb = xgb_selector.transform(X_val)
        
        # 6. mRMR
        if verbose:
            print("\n[方法6] mRMR特征选择...")
        mrmr_selector = mRMRSelector(k_features=k_features)
        mrmr_selector.fit(X_train, y_train)
        X_train_mrmr = mrmr_selector.transform(X_train)
        X_val_mrmr = mrmr_selector.transform(X_val)
        
        # 训练和评估所有方法
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        
        # 方法1: ATFS
        model_atfs = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_atfs.fit(X_train_atfs, y_train)
        result_atfs = evaluate_model(model_atfs, X_val_atfs, y_val, f"{dataset_name}_ATFS", verbose=False)
        
        # 方法2: 所有特征
        model_all = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_all.fit(X_train_all, y_train)
        result_all = evaluate_model(model_all, X_val_all, y_val, f"{dataset_name}_All", verbose=False)
        
        # 方法3: RF-RFE
        model_rf = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_rf.fit(X_train_rf, y_train)
        result_rf = evaluate_model(model_rf, X_val_rf, y_val, f"{dataset_name}_RF", verbose=False)
        
        # 方法4: LASSO
        model_lasso = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_lasso.fit(X_train_lasso, y_train)
        result_lasso = evaluate_model(model_lasso, X_val_lasso, y_val, f"{dataset_name}_LASSO", verbose=False)
        
        # 方法5: XGBoost内置
        model_xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_xgb.fit(X_train_xgb, y_train)
        result_xgb = evaluate_model(model_xgb, X_val_xgb, y_val, f"{dataset_name}_XGB", verbose=False)
        
        # 方法6: mRMR
        model_mrmr = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model_mrmr.fit(X_train_mrmr, y_train)
        result_mrmr = evaluate_model(model_mrmr, X_val_mrmr, y_val, f"{dataset_name}_mRMR", verbose=False)
        
        # 收集结果
        method_results['ATFS']['pr_auc'].append(result_atfs['pr_auc'])
        method_results['ATFS']['roc_auc'].append(result_atfs['roc_auc'])
        method_results['ATFS']['f1'].append(result_atfs['f1'])
        
        method_results['All Features']['pr_auc'].append(result_all['pr_auc'])
        method_results['All Features']['roc_auc'].append(result_all['roc_auc'])
        method_results['All Features']['f1'].append(result_all['f1'])
        
        method_results['RF-RFE']['pr_auc'].append(result_rf['pr_auc'])
        method_results['RF-RFE']['roc_auc'].append(result_rf['roc_auc'])
        method_results['RF-RFE']['f1'].append(result_rf['f1'])
        
        method_results['LASSO']['pr_auc'].append(result_lasso['pr_auc'])
        method_results['LASSO']['roc_auc'].append(result_lasso['roc_auc'])
        method_results['LASSO']['f1'].append(result_lasso['f1'])
        
        method_results['XGBoost BuiltIn']['pr_auc'].append(result_xgb['pr_auc'])
        method_results['XGBoost BuiltIn']['roc_auc'].append(result_xgb['roc_auc'])
        method_results['XGBoost BuiltIn']['f1'].append(result_xgb['f1'])
        
        method_results['mRMR']['pr_auc'].append(result_mrmr['pr_auc'])
        method_results['mRMR']['roc_auc'].append(result_mrmr['roc_auc'])
        method_results['mRMR']['f1'].append(result_mrmr['f1'])
        
        fold_idx += 1
    
    # 计算统计指标和p值
    comparison_results = []
    
    # ATFS作为基准
    atfs_pr_auc = method_results['ATFS']['pr_auc']
    
    for method, results in method_results.items():
        # 计算均值
        mean_pr_auc = np.mean(results['pr_auc'])
        mean_roc_auc = np.mean(results['roc_auc'])
        mean_f1 = np.mean(results['f1'])
        
        # 计算置信区间
        ci_pr_auc = calculate_confidence_intervals(results['pr_auc'])
        ci_roc_auc = calculate_confidence_intervals(results['roc_auc'])
        ci_f1 = calculate_confidence_intervals(results['f1'])
        
        # 计算与ATFS的p值 (仅对PR-AUC)
        if method == 'ATFS':
            p_value = 1.0  # ATFS与自身比较
        else:
            _, p_value = ttest_rel(atfs_pr_auc, results['pr_auc'])
        
        comparison_results.append({
            'Method': method,
            'PR-AUC (CI)': format_ci(mean_pr_auc, ci_pr_auc),
            'ROC-AUC (CI)': format_ci(mean_roc_auc, ci_roc_auc),
            'F1 Score (CI)': format_ci(mean_f1, ci_f1),
            'p-value (vs ATFS)': p_value
        })
    
    results_df = pd.DataFrame(comparison_results)
    
    excel_path = results_dir / f'{dataset_name}_feature_selection_comparison.xlsx'
    results_df.to_excel(excel_path, index=False)
    
    if verbose:
        print(f"\n特征选择方法对比结果已保存至: {excel_path}")
        print("\n对比结果汇总:")
        print(results_df)
    
    # 可视化对比结果
    plt.figure(figsize=(14, 8))
    metrics = ['pr_auc', 'roc_auc', 'f1']
    titles = ['PR-AUC', 'ROC-AUC', 'F1 Score']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        data = []
        methods = []
        for method, results in method_results.items():
            data.append(results[metric])
            methods.append(method)
        
        plt.boxplot(data, labels=methods)
        plt.title(titles[i])
        plt.xticks(rotation=45)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / f'{dataset_name}_metrics_comparison.png')
    plt.close()
    
    if verbose:
        print(f"指标对比图已保存至: {results_dir / f'{dataset_name}_metrics_comparison.png'}")
    
    return results_df

# ====================== 主程序 ======================
if __name__ == "__main__":
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    root_path_ = Path.cwd().parent
    FILEPATH = str(Path(root_path_) / 'data/all_feature_09_cn_113k.txt')
    
    datasets = {
        "NewData1": FILEPATH
    }
    
    VERBOSE = True
    all_comparison_results = {}
    
    for name, path in datasets.items():
        print(f"\n{'='*60}")
        print(f"开始处理数据集: {name}")
        print(f"{'='*60}")
        
        comparison_df = train_and_evaluate_cv(path, name, verbose=VERBOSE)
        all_comparison_results[name] = comparison_df
        
        dataset_result_path = results_dir / f'{name}_comparison_results.xlsx'
        comparison_df.to_excel(dataset_result_path, index=False)
        print(f"数据集 {name} 的对比结果已保存至: {dataset_result_path}")
    
    final_results = pd.concat(all_comparison_results.values(), keys=all_comparison_results.keys())
    final_results_path = results_dir / 'all_datasets_comparison_results.xlsx'
    final_results.to_excel(final_results_path)
    print(f"\n所有数据集的汇总对比结果已保存至: {final_results_path}")