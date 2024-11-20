# 连接到MySQL数据库
import numpy as np
import pymysql
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

connection = pymysql.connect(
    host='localhost',
    user='root',
    passwd='Qpwoeiruty123!',
    database='py-db',
)

try:
    with connection.cursor() as cursor:
        # 查询数据
        query = "SELECT `金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"
        cursor.execute(query)
        result = cursor.fetchall()

        # 提取数据
        data = [row[0] for row in result]
        indices = list(range(len(data)))  # 用索引代表 x 轴

        # 检查是否有数据
        if not data:
            print("没有符合条件的数据。")
        else:
            # 绘制散点图
            plt.scatter(indices, data, color='blue', label='data')

            # 拟合线性回归模型
            coefficients = np.polyfit(indices, data, 3)  # 一次多项式拟合 (线性)
            polynomial = np.poly1d(coefficients)
            fitted_values = polynomial(indices)

            # 绘制拟合曲线
            plt.plot(indices, fitted_values, color='red', label='curve')

            # 预测下一个消费金额
            next_index = len(indices)  # 下一个数据点的索引
            next_value = polynomial(next_index)  # 使用模型预测值
            print(f"预测的下一个消费金额: {next_value:.2f}")

            # 标注预测值在图表中
            plt.scatter([next_index], [next_value], color='green', label='预测值', zorder=5)
            plt.annotate(f"{next_value:.2f}",
                         (next_index, next_value),
                         textcoords="offset points",
                         xytext=(5, 10),
                         ha='center',
                         fontsize=10,
                         color='green')

            # 图表设置
            plt.xlabel('data Index')
            plt.ylabel('Expense')
            plt.title('prediction Curve')
            plt.legend()
            plt.grid(True)

            # 保存图片为 PNG 文件
            plt.savefig("scatter_fit.png", format="png", dpi=300, bbox_inches="tight")

            # 显示图表
            plt.show()

# 线性机器学习预测
    with connection.cursor() as cursor:
        # 查询数据
        query = "SELECT `金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"
        cursor.execute(query)
        result = cursor.fetchall()

        # 提取数据
        data = [row[0] for row in result]
        indices = np.array(range(len(data))).reshape(-1, 1)  # 输入特征 X：数据索引，二维

        if not data:
            print("没有符合条件的数据。")
        else:
            # 数据划分
            X_train, X_test, y_train, y_test = train_test_split(
                indices, data, test_size=0.2, random_state=42
            )

            # 初始化线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)  # 训练模型

            # 测试模型性能
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"模型均方误差 (MSE): {mse:.2f}")

            # 使用模型预测下一个消费金额
            next_index = np.array([[len(data)]])  # 下一个索引
            next_value = model.predict(next_index)[0]  # 模型预测值
            print(f"预测的下一个消费金额: {next_value:.2f}")

            # 绘制数据与预测结果
            plt.scatter(indices, data, color="blue", label="data")
            plt.plot(indices, model.predict(indices), color="red", label="curve")
            plt.scatter(next_index, next_value, color="green", label="prediction", zorder=5)
            plt.annotate(f"{next_value:.2f}",
                         (next_index[0][0], next_value),
                         textcoords="offset points",
                         xytext=(5, 10),
                         ha='center',
                         fontsize=10,
                         color='green')

            # 图表设置
            plt.xlabel("data Index")
            plt.ylabel("expense")
            plt.title("ml linear model prediction")
            plt.legend()
            plt.grid(True)

            # 保存图像
            plt.savefig("ml_prediction.png", format="png", dpi=300, bbox_inches="tight")

            # 显示图表
            plt.show()

# 使用随机森林做拟合预测
    with connection.cursor() as cursor:
        query = "SELECT `金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"
        cursor.execute(query)
        result = cursor.fetchall()

        data = [row[0] for row in result]
        indices = np.array(range(len(data))).reshape(-1, 1)

        if not data:
            print("没有符合条件的数据。")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                indices, data, test_size=0.2, random_state=42
            )

            # 使用随机森林替代线性回归
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"模型均方误差 (MSE): {mse:.2f}")

            next_index = np.array([[len(data)]])
            next_value = model.predict(next_index)[0]
            print(f"预测的下一个消费金额: {next_value:.2f}")

            dense_indices = np.linspace(0, len(data), 500).reshape(-1, 1)
            dense_predictions = model.predict(dense_indices)

            plt.scatter(indices, data, color="blue", label="data")
            plt.plot(dense_indices, dense_predictions, color="red", label="curve")
            plt.scatter(next_index, next_value, color="green", label="predicion", zorder=5)
            plt.annotate(f"{next_value:.2f}",
                         (next_index[0][0], next_value),
                         textcoords="offset points",
                         xytext=(5, 10),
                         ha='center',
                         fontsize=10,
                         color='green')

            plt.xlabel("data Index")
            plt.ylabel("expense")
            plt.title("ml random forest model prediction")
            plt.legend()
            plt.grid(True)
            plt.savefig("ml_rf_prediction.png", format="png", dpi=300, bbox_inches="tight")
            plt.show()

# 使用SVR向量机预测
    with connection.cursor() as cursor:
        query = "SELECT `金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"
        cursor.execute(query)
        result = cursor.fetchall()

        data = [row[0] for row in result]
        indices = np.array(range(len(data))).reshape(-1, 1)

        if not data:
            print("没有符合条件的数据。")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                indices, data, test_size=0.2, random_state=42
            )

            # 使用随机森林替代线性回归
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"模型均方误差 (MSE): {mse:.2f}")

            next_index = np.array([[len(data)]])
            next_value = model.predict(next_index)[0]
            print(f"预测的下一个消费金额: {next_value:.2f}")

            dense_indices = np.linspace(0, len(data), 500).reshape(-1, 1)
            dense_predictions = model.predict(dense_indices)

            plt.scatter(indices, data, color="blue", label="data")
            plt.plot(dense_indices, dense_predictions, color="red", label="curve")
            plt.scatter(next_index, next_value, color="green", label="prediction", zorder=5)
            plt.annotate(f"{next_value:.2f}",
                         (next_index[0][0], next_value),
                         textcoords="offset points",
                         xytext=(5, 10),
                         ha='center',
                         fontsize=10,
                         color='green')

            plt.xlabel("data Index")
            plt.ylabel("expense")
            plt.title("ml svr model prediction")
            plt.legend()
            plt.grid(True)
            plt.savefig("ml_rf_SVR_prediction.png", format="png", dpi=300, bbox_inches="tight")
            plt.show()


finally:
    connection.close()