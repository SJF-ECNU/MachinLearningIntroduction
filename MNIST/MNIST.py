import tensorflow as tf  # 导入TensorFlow库，用于构建和训练深度学习模型

# 从MNIST数据集加载训练和测试数据，将图像数据存储在x_train和x_test中，将标签存储在y_train和y_test中
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 对图像数据进行归一化，将像素值范围从0到255缩放到0到1之间
x_train = x_train/255.0
x_test = x_test/255.0

# 对图像数据进行形状重塑，将每张图像的维度从(28, 28)调整为(28, 28, 1)，因为卷积层需要输入具有通道数的图像
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 创建一个Sequential模型，该模型是一个层的线性堆叠
model = tf.keras.Sequential([
    # 第一层是一个2D卷积层，包含32个3x3的卷积核，激活函数为ReLU，输入图像大小为(28, 28, 1)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # 第二层是最大池化层，使用2x2的池化窗口
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # 第三层是展平层，将二维的卷积输出展平为一维
    tf.keras.layers.Flatten(),
    
    # 第四层是全连接层，包含128个神经元，激活函数为ReLU
    tf.keras.layers.Dense(128, activation='relu'),
    
    # 第五层是全连接层，包含10个神经元，激活函数为softmax，用于输出10个类别的概率分布
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，使用训练数据x_train和y_train进行模型训练，进行5个epochs的训练，使用批量大小为64，
# 同时使用测试数据x_test和y_test进行验证
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# 评估模型在测试数据上的性能，计算测试损失和测试准确率，verbose=2表示以详细模式显示评估信息
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy*100:.2f}%")  # 打印测试准确率的百分比形式

# 使用训练好的模型进行对测试数据的预测
predictions = model.predict(x_test)

# 将模型的预测结果转换为具体的类别标签，即找到每个预测概率分布中的最大值所在的索引，然后将其转换为NumPy数组
predicted_labels = [tf.argmax(prediction).numpy() for prediction in predictions]

model.save("model_MNIST.h5")