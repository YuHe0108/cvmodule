# import sqlite3
# import time
#
# conn = sqlite3.connect('interface_data.db')
# c = conn.cursor()
# c.execute("select name from sqlite_master where type='table' order by name")
# print(c.fetchall())
#
# table_name = "Person"
# columns = ["id", "start", "end", "score"]
# # c.execute(f"""-- CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, {columns[1]}, {columns[2]}, {columns[3]})""")
#
# for i in range(10):
#     c.execute(f"""INSERT INTO {table_name} (id, {columns[1]}, {columns[2]}, {columns[3]}) values(NULL, 1, "2", 3.3)""")
#
# # c.execute(f"delete from {table_name} where id IN (select id from {table_name} limit 5)")
#
# c.close()
# # 执行
# conn.commit()
#
# # recs = c.execute(f"""SELECT * FROM {table_name} LIMIT 2""")
# # for row in recs:
# #     print(row, type(row))
# conn.close()
