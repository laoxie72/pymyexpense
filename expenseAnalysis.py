# 连接到MySQL数据库
import numpy as np
import pymysql
from matplotlib import pyplot as plt

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
            # plt.savefig("scatter_fit.png", format="png", dpi=300, bbox_inches="tight")

            # 显示图表
            plt.show()

finally:
    connection.close()