import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
# 输入年度数据集
data = pd.read_csv("D:\BaiduNetdiskDownload\oco2018table\oco2201806copy.csv")
# 特征选择
selected_features = ["dem","em","evi","GPP","ndvi","NO2","PET","pre","SP","tem","WD"]
# 获取选择的特征和目标变量
X = data[selected_features]
y = data["Xco2"]
# 进行数据归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
df0 = min_max_scaler.fit_transform(data)
df = pd.DataFrame(df0, columns=data.columns)
# 对类别特征进行独热编码
X = pd.get_dummies(X)
# 识别数值特征和分类特征
# 假设：数值特征是数字类型，分类特征是对象/字符串类型
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"数值特征: {numerical_features}")
# 划分训练集和测试集（先划分，防止数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=46, shuffle=True
)
# 创建列转换器，分别处理数值和分类特征
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # 随机森林不需要标准化数值特征
    ]
)
# 定义参数网格
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1,2],
    'model__max_features': ['sqrt', 'log2'],
    'model__bootstrap': [True, False]
}

# 创建模型
TAF_model = RandomForestRegressor(random_state=42, n_jobs=-1)
# 创建完整的Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', TAF_model)
])
# ========== 网格搜索调参 ==========
# 创建GridSearchCV对象
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,  # 3折交叉验证
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1,  # 使用所有可用的CPU核心
    return_train_score=True
)
# 执行网格搜索
grid_search.fit(X_train, y_train)
# ========== 输出网格搜索结果 ==========
print("\n网格搜索完成！")
print("最佳参数组合:", grid_search.best_params_)
print("最佳交叉验证分数 (负MSE):", grid_search.best_score_)
print("最佳模型R²分数:", grid_search.best_estimator_.score(X_train, y_train))
# 获取最佳模型
best_model = grid_search.best_estimator_
# ========== 在测试集上评估最佳模型 ==========
print("\n在测试集上评估最佳模型:")
# 预测
y_pred = best_model.predict(X_test)
# 计算评估指标
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R² 分数: {r2:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")

# ========== 特征重要性 ==========
print("\n特征重要性分析:")
# 获取预处理后的特征名称

# 获取模型的特征重要性
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = best_model.named_steps['model'].feature_importances_

    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': numerical_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 10重要特征:")
    print(feature_importance_df.head(10))

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('特征重要性')
    plt.title('Top 20 特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ========== 可视化预测结果 ==========
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='预测值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='完美预测线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'随机森林模型预测结果 (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========== 保存最佳模型 ==========
print("\n保存最佳模型...")
# 保存GridSearchCV结果
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df.to_csv('grid_search_results.csv', index=False)
print("网格搜索结果已保存到 'grid_search_results.csv'")
# 保存最佳模型
# with open('best_model.pkl', 'wb') as f:
#     pickle.dump({
#         'model': best_model,
#         'preprocessor': preprocessor,
#         'best_params': grid_search.best_params_,
#         'feature_names': numerical_features,
#         'cv_score': grid_search.best_score_
#     }, f)
# print("最佳模型已保存到 'best_model.pkl'")

# ========== 输出调参分析 ==========
print("\n" + "=" * 60)
print("调参分析报告")
print("=" * 60)

# 分析不同参数的影响
print("\n不同参数组合的表现（前5名）:")
top_results = cv_results_df.sort_values('mean_test_score', ascending=False).head()
for i, row in top_results.iterrows():
    print(f"\n第{i}名:")
    print(f"  参数: {row['params']}")
    print(f"  平均测试分数: {-row['mean_test_score']:.4f} (MSE)")
    print(f"  训练分数: {-row['mean_train_score']:.4f} (MSE)")
# 计算过拟合程度
best_result = cv_results_df[cv_results_df['rank_test_score'] == 1].iloc[0]
overfit_score = best_result['mean_train_score'] - best_result['mean_test_score']
print(f"\n过拟合程度 (训练分数 - 测试分数): {overfit_score:.4f}")
print("\n调参完成！")