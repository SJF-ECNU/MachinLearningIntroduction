# 导入必要的库
from sklearn.datasets import load_wine  # 导入葡萄酒数据集
from sklearn.model_selection import train_test_split  # 用于数据划分
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 用于评估模型性能

# 步骤 1: 加载葡萄酒数据集
wine = load_wine()
X = wine.data  # 特征
y = wine.target  # 目标变量（葡萄酒的种类）

# 步骤 2: 数据预处理
# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤 3: 选择和训练模型
# 创建随机森林分类器
clf = RandomForestClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 步骤 4: 模型评估
# 使用模型进行预测
y_pred = clf.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f'准确性：{accuracy:.2f}')

# 打印分类报告，包括精确度、召回率和F1分数等
print(classification_report(y_test, y_pred))

# 打印混淆矩阵，展示分类性能
conf_matrix = confusion_matrix(y_test, y_pred)
print('混淆矩阵：\n', conf_matrix)
