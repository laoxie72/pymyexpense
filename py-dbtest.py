import pymysql

# 连接数据库
connection = pymysql.connect(
    host='localhost',         # 主机地址
    user='root',     # 用户名
    password='Qpwoeiruty123!', # 密码
    database='py-db', # 数据库名称
    port=3306                 # 端口号（默认3306）
)

# 检查连接是否成功
if connection.open:
    print("连接成功！")
else:
    print("连接失败！")
