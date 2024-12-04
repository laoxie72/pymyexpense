import collections
import pandas as pd
import pymysql
import statistics
from matplotlib import pyplot as plt

# 连接到 MySQL 数据库
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="Qpwoeiruty123!",
    database="py-db"
)

try:
    # 查询 1: 计算金额的平均值和方差
    with connection.cursor() as cursor:
        query = "SELECT `金额（可执行）` FROM table_detail WHERE `项目` = '小结' AND NOT `金额（可执行）` > 0"
        cursor.execute(query)
        result = cursor.fetchall()
        ages = [row[0] for row in result]

        # 检查数据是否为空
        if ages:
            avg_cash = statistics.mean(ages)
            variance_cash = statistics.variance(ages)
        else:
            avg_cash = None
            variance_cash = None

    # 查询 2: 日期出现次数统计
    with connection.cursor() as cursor:
        query = "SELECT `日期` FROM table_detail WHERE `金额（可执行）` <= -15 AND NOT `项目` = '小结';"
        cursor.execute(query)
        result = cursor.fetchall()
        overspend = [row[0] for row in result]

        if overspend:
            date_counts = collections.Counter(overspend)
            df = pd.DataFrame(date_counts.items(), columns=['日期', '出现次数'])
            df = df.sort_values(by='日期', ascending=True)
        else:
            df = None

    # 查询 3: 日期与金额的变化曲线
    with connection.cursor() as cursor:
        query = "SELECT `日期`, `金额（可执行）` FROM table_detail WHERE `项目` = '小结' GROUP BY `日期`, `金额（可执行）`;"
        cursor.execute(query)
        result = cursor.fetchall()
        dates = [row[0] for row in result]
        amounts = [row[1] for row in result]

    # 绘制金额变化曲线
    if dates and amounts:
        plt.figure(figsize=(10, 6))
        plt.plot(dates, amounts, marker='o', linestyle='-', color='b', label="vary")
        plt.xlabel("date")
        plt.ylabel("Expense")
        plt.title("variation")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        # 保存图片为 PNG 文件
        plt.savefig("output.png", format="png", dpi=300, bbox_inches="tight")

        # plt.show()
    else:
        print("没有数据可绘制。")

    # 生成 Markdown 内容
    markdown_content = "# 数据分析结果\n\n"

    # 添加金额分析结果
    markdown_content += "## 金额（可执行）数据分析\n\n"
    if avg_cash is not None and variance_cash is not None:
        markdown_content += f"- 平均值: {avg_cash}\n"
        markdown_content += f"- 方差: {variance_cash}\n\n"
    else:
        markdown_content += "- 没有符合条件的数据。\n\n"

    # 添加日期统计结果
    markdown_content += "## 日期出现次数统计\n\n"
    if df is not None:
        markdown_content += df.to_markdown(index=False)
    else:
        markdown_content += "- 没有符合条件的数据。\n"

    # 保存 Markdown 文件
    with open("result.md", "w", encoding="utf-8") as file:
        file.write(markdown_content)
    print("Markdown 文件已生成，保存为 'result.md'")

finally:
    # 关闭数据库连接
    connection.close()
