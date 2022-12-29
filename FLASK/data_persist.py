"""通过将数据保存在数据库中，保证了持久行"""
import sqlite3


class DataPersist:
    def __init__(self, database_path):
        self.max_data_limit = 5  # 每张表最多保存的数据量

        self.database_path = database_path
        self.ai_interface_table_name = "ai_interface"
        self.complete_table_name = "upload_completed"  # 数据被消费上传后, 原始数据移动至此表
        self.table_columns = ["createTime", "image", "result"]

        self.conn = None
        self.create_database()  # 创建数据库
        self.create_table(self.ai_interface_table_name)  # 创建表
        self.create_table(self.complete_table_name)

    def __del__(self):
        self.conn.close()  # 关闭连接

    def create_database(self):
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        return

    def create_table(self, table_name):
        if not self.is_table_exist(table_name):
            c = self.conn.cursor()
            # 设置 id 为主键，自动增加
            c.execute(
                f"""CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, {self.table_columns[0]}, {self.table_columns[1]}, {self.table_columns[2]})""")
            self.conn.commit()  # 执行
            c.close()
        return

    def insert_data(self, table_name, info):
        if not self.is_table_exist(table_name):
            return
        c = self.conn.cursor()
        if table_name == self.ai_interface_table_name:
            c.execute(
                f"""INSERT INTO {table_name} (id, createTime, image, result) 
                    values(NULL, '{info["createTime"]}', '{info["image"]}', '{info["result"]}')""")
        elif table_name == self.complete_table_name:
            c.execute(
                f"""INSERT INTO {table_name} (id, createTime, image, result) 
                    values(NULL, '{info["createTime"]}', '{info["image"]}', '{info["result"]}')""")
        else:
            print(f"unknown table name: {table_name}")
        self.conn.commit()  # 执行
        c.close()
        self.remove_surplus_data(table_name)
        return

    def get_data(self, table_name):  # 返回第一条数据
        if not self.is_table_exist(table_name):
            return {}
        c = self.conn.cursor()
        recs = c.execute(f"""SELECT * FROM {table_name} LIMIT 1""")
        col_names = self.get_column_name(table_name)
        for row in recs:
            res = {col_names[i]: row[i] for i in range(len(col_names))}
            return res
        return {}

    def is_table_exist(self, table_name):
        c = self.conn.cursor()
        c.execute("select name from sqlite_master where type='table' order by name")  # 查看一个数据库中有哪些表
        res = c.fetchall()
        res = [item[0] for item in res if len(res) > 0]
        if len(res) > 0 and table_name in res:
            c.close()
            return True
        return False

    def get_column_name(self, table_name):  # 获取表每列的名称
        c = self.conn.cursor()
        c.execute(f"select * from {table_name}")
        col_name_list = [t[0] for t in c.description]
        c.close()
        return col_name_list

    def remove_surplus_data(self, table_name):
        """当表中数据超过指定条数时，则取消"""
        c = self.conn.cursor()
        c.execute(f"select * from {table_name}")
        results = c.fetchall()
        if len(results) > self.max_data_limit:
            surplus_num = len(results) - self.max_data_limit
            c.execute(f"delete from {table_name} where id IN (select id from {table_name} limit {surplus_num})")
        c.close()
        self.conn.commit()  # 执行
        return

    def remove_top_nums_data(self, table_name, remove_nums):
        c = self.conn.cursor()
        c.execute(f"select * from {table_name}")
        results = c.fetchall()
        if len(results) >= remove_nums:
            c.execute(f"delete from {table_name} where id IN (select id from {table_name} limit {remove_nums})")
        c.close()
        self.conn.commit()  # 执行
        return

    def create_table_example(self, table_name, columns):  # TODO:Example
        # 创建一个表：表的名字为 table_name, 新建 start\end\score三列
        c = self.conn.cursor()
        c.execute("select name from sqlite_master where type='table' order by name")  # 查看一个数据库中有哪些表
        if len(c.fetchall()) != 0 and table_name not in c.fetchall()[0]:
            c.execute(f"""CREATE TABLE {table_name} (start, end, score)""")
            self.conn.commit()  # 执行
            self.conn.close()  # 关闭连接
        return
