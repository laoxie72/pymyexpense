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
    # 存储所有模型的预测值
    predictions = []
#多项式预测模型
    with connection.cursor() as cursor:
        # 查询数据
        query = "SELECT `金额（可执行）` FROM `table_detail` WHERE `项目` != '小结' AND `金额（可执行）` < 0;"
        cursor.execute(query)
        result = cursor.fetchall()

        # 提取数据
        data = [row[0] for row in result]
        indices = list(range(len(data)))  # 用索引代表 x 轴
        # print(data)

        # 检查是否有数据
        if not data:
            print("没有符合条件的数据。")
        else:
            # 绘制散点图
            plt.scatter(indices, data, color='blue', label='data')

            # 拟合线性回归模型
            coefficients = np.polyfit(indices, data, 5)  # 一次多项式拟合 (线性)
            polynomial = np.poly1d(coefficients)
            fitted_values = polynomial(indices)

            # 绘制拟合曲线
            plt.plot(indices, fitted_values, color='red', label='curve')

            # 预测下一个消费金额
            next_index = len(indices)  # 下一个数据点的索引
            next_value = polynomial(next_index)  # 使用模型预测值
            predictions.append(next_value)  # 添加三次多项式的预测值

            print(f"多项式拟合预测的下一个消费金额: {next_value:.2f}")

            # 标注预测值在图表中
            plt.scatter([next_index], [next_value], color='green', label='prediction', zorder=5)
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
            plt.savefig("polynomial_prediction.png", format="png", dpi=300, bbox_inches="tight")

            # 显示图表
            # plt.show()

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
            print(f"ml_lr模型均方误差 (MSE): {mse:.2f}")

            # 使用模型预测下一个消费金额
            next_index = np.array([[len(data)]])  # 下一个索引
            next_value = model.predict(next_index)[0]  # 模型预测值
            predictions.append(next_value)  # 添加线性回归的预测值

            print(f"ml_lr预测的下一个消费金额: {next_value:.2f}")

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
            # plt.show()

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
            print(f"ml_rf模型均方误差 (MSE): {mse:.2f}")

            next_index = np.array([[len(data)]])
            next_value = model.predict(next_index)[0]
            predictions.append(next_value)  # 添加随机森林的预测值

            print(f"ml_rf预测的下一个消费金额: {next_value:.2f}")

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
            # plt.show()

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
            print(f"ml_rf_SVR模型均方误差 (MSE): {mse:.2f}")

            next_index = np.array([[len(data)]])
            next_value = model.predict(next_index)[0]
            predictions.append(next_value)  # 添加SVR的预测值

            print(f"ml_rf_SVR预测的下一个消费金额: {next_value:.2f}")

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
            # plt.show()
    # # 截断到小数点后 5 位
    # formatted_predictions = [round(value, 5) for value in predictions]
    #
    # # 计算均值和方差
    # mean_value = round(np.mean(predictions), 5)  # 均值
    # variance_value = round(np.var(predictions), 5)  # 总体方差
    # sample_variance = round(np.var(predictions, ddof=1), 5)  # 样本方差

    # 计算均值和方差
    mean_value = np.mean(predictions)  # 计算均值
    variance_value = np.var(predictions)  # 计算总体方差
    sample_variance = np.var(predictions, ddof=1)  # 计算样本方差

    # 构建 Markdown 表格
    table_header = "| 序号 | 预测值 |\n| -------- | ----------------------- |"
    table_rows = "\n".join(
        f"| {i + 1}    | {value:.5f}              |" for i, value in enumerate(predictions)
    )

    # 输出结果
    output_text = f"""
# 模型预测结果分析
    
## 预测值列表
{table_header}
{table_rows}
    
## 统计分析
- **均值**: {mean_value:.2f}
- **总体方差**: {variance_value:.2f}
- **样本方差**: {sample_variance:.2f}
"""

    # 将结果写入 .md 文件
    prediction_output = "model_predictions.md"
    with open(prediction_output, "w", encoding="utf-8") as file:
        file.write(output_text)

    print(f"结果已保存到 {prediction_output}")

    # 输出结果
    # print(f"预测值列表: {predictions}")
    # print(f"均值: {mean_value:.2f}")
    # print(f"总体方差: {variance_value:.2f}")
    # print(f"样本方差: {sample_variance:.2f}")

finally:
    connection.close()