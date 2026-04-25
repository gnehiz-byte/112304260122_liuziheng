import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

print("=" * 60)
print("第一部分：词袋模型 - 情感分析")
print("=" * 60)

print("\n[1] 加载数据...")
train = pd.read_csv('labeledTrainData.tsv', delimiter='\t', encoding='utf-8')
test = pd.read_csv('testData.tsv', delimiter='\t', encoding='utf-8')

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

print("\n[2] 文本预处理...")

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("清洗训练集...")
train_clean = [clean_text(r) for r in train['review'].tolist()]
print("清洗测试集...")
test_clean = [clean_text(r) for r in test['review'].tolist()]

print("清洗后的评论示例:")
print(train_clean[0][:200])

print("\n[3] 创建TF-IDF词袋模型...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_clean)
X_test = vectorizer.transform(test_clean)
y_train = train['sentiment'].values

print(f"特征数量: {X_train.shape[1]}")

print("\n[4] 训练逻辑回归分类器...")
clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, C=1.0)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
clf.fit(X_tr, y_tr)

val_score = clf.score(X_val, y_val)
print(f"验证集准确率: {val_score:.4f}")

val_proba = clf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_proba)
print(f"验证集 ROC-AUC: {val_auc:.4f}")

print("\n训练最终模型...")
clf.fit(X_train, y_train)

print("\n[5] 生成预测结果...")
predictions = clf.predict(X_test)

result = pd.DataFrame({
    'id': test['id'],
    'sentiment': predictions
})

result.to_csv('submission_bow.csv', index=False)
print(f"\n预测结果已保存到 submission_bow.csv")
print(f"正面评价数量: {sum(predictions == 1)}")
print(f"负面评价数量: {sum(predictions == 0)}")

print("\n" + "=" * 60)
print("第一部分完成！")
print("=" * 60)
