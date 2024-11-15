import collections

import pandas as pd
import pymysql
import statistics

# 连接到 MySQL 数据库
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="Qpwoeiruty123!",
    database="py-db"
)

try:
    # 创建游标并执行查询
    with connection.cursor() as cursor:
        query = "SELECT `金额（可执行）` FROM table_detail WHERE `项目` = '小结'AND NOT `金额（可执行）` > 0"
        cursor.execute(query)

        # 获取查询结果并转换为列表
        result = cursor.fetchall()
        ages = [row[0] for row in result]
        # 输出查询的结果
        # print(result)

        # 检查数据是否为空
        if ages:
            # 计算平均值和方差
            avg_cash = statistics.mean(ages)
            variance_cash = statistics.variance(ages)

            # print(f"花费的平均值: {avg_cash}")
            # print(f"花费的方差: {variance_cash}")
        else:
            print("没有符合条件的数据。")

        # 创建第二个游标
        with connection.cursor() as cursor1:
            query1 = "SELECT `日期` FROM table_detail WHERE `金额（可执行）` <-20 AND NOT `项目`= '小结';"
            cursor1.execute(query1)

        # 获取查询结果并转换为列表（数组）
        result1 = cursor1.fetchall()
        overspend = [row[0] for row in result1]
        # print("筛选的日期:", overspend)

        # 检查数据是否为空
        if overspend:
            # 使用 Counter 统计每个日期出现的次数
            date_counts = collections.Counter(overspend)

            # 将统计结果转换为 DataFrame 表格
            df = pd.DataFrame(date_counts.items(), columns=['日期', '出现次数'])

            # 按日期排序并输出
            df = df.sort_values(by='日期', ascending=True)
            # print("日期出现次数表格：\n", df)
        else:
            print("没有符合条件的数据。")

    # 生成 Markdown 内容
    markdown_content = "# 数据分析结果\n\n"

    # 添加第一个查询结果
    markdown_content += "## 金额（可执行）数据分析\n\n"
    if avg_cash is not None and variance_cash is not None:
        markdown_content += f"- 平均值: {avg_cash}\n"
        markdown_content += f"- 方差: {variance_cash}\n\n"
    else:
        markdown_content += "- 没有符合条件的数据。\n\n"

    # 添加第二个查询结果
    markdown_content += "## 日期出现次数统计\n\n"
    markdown_content += df.to_markdown(index=False)

    # 保存为 Markdown 文件
    with open("result.md", "w", encoding="utf-8") as file:
        file.write(markdown_content)

    print("Markdown 文件已生成，保存为 'result.md'")



finally:
    # 关闭数据库连接
    connection.close()
