机器学习实验：基于 Word2Vec 的情感预测

1. 学生信息
姓名：刘梓恒
学号：112304260122
班级：数据1231



2. 实验任务
本实验基于给定文本数据，使用 Word2Vec 将文本转为向量特征，再结合分类模型完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

3. 比赛与提交信息
比赛名称：Bag of Words Meets Bags of Popcorn Use Google's Word2Vec for movie reviews
比赛链接：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
提交日期：4.25

GitHub 仓库地址：https://github.com/gnehiz-byte/112304260122_liuziheng.git
GitHub README 地址：

注意：GitHub 仓库首页或 README 页面中，必须能看到"姓名 + 学号"，否则无效。

4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：0.89696

Public Score：
Private Score（如有）：
排名（如能看到可填写）：

5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。



建议将截图保存在 images 文件夹中。
截图文件名示例：2023123456_张三_kaggle_score.png

6. 实验方法说明

（1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

我的做法：
1. 使用正则表达式去除HTML标签 <[^>]+>
2. 去除所有非字母字符，只保留a-z和空格
3. 转换为小写
4. 去除多余空格
5. 按空格分词，去除长度小于2的单词

（2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

我的做法：
1. 使用gensim库自己训练Word2Vec模型
2. 词向量维度：150维
3. 训练参数：window=5, min_count=3, epochs=15
4. 将训练集和无标签数据合并后一起训练Word2Vec
5. 句子向量 = 句子中所有词向量的平均值
6. 如果句子中没有词汇表中的词，则用零向量表示

（3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

我的做法：
使用 Logistic Regression（逻辑回归）作为分类模型
- 参数：C=1.0, max_iter=1000
- 使用80%数据训练，20%数据验证
- 输出二分类结果（0或1）

7. 实验流程
请简要说明你的实验流程。

我的实验流程：
1. 读取训练集、测试集和无标签数据
2. 对文本进行预处理（去HTML、转小写、分词）
3. 合并训练集和无标签数据，使用gensim训练Word2Vec模型
4. 将每条评论表示为词向量的平均值，得到句子向量
5. 使用训练集训练逻辑回归分类器
6. 在验证集上评估模型效果
7. 用全部训练数据重新训练模型
8. 在测试集上预测结果
9. 生成submission文件并提交Kaggle

8. 文件说明
请说明仓库中各文件或文件夹的作用。

我的项目结构：
project/
├─ part1_bag_of_words.py    词袋模型代码
├─ submission_bow.csv       词袋模型预测结果
└─ README.md                 实验报告

9. 实验结果
模型              验证集准确率   验证集ROC-AUC
词袋模型(TF-IDF)  ~0.88        ~0.93
Word2Vec          ~0.86        ~0.91

10. 总结与反思
本次实验学习了：
1. 文本预处理的基本方法
2. Word2Vec词向量的训练与使用
3. 如何将变长文本转换为固定维度向量
4. 使用逻辑回归进行情感分类

改进方向：
- 可以尝试使用预训练的Word2Vec模型（如Google News）
- 可以尝试Doc2Vec直接生成文档向量
- 可以尝试结合TF-IDF和Word2Vec的特征
- 可以尝试其他分类器如SVM、Random Forest等
