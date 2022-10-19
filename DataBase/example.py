import sqlite3
import os


def run(table_name):
    db_path = r'test.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("select name from sqlite_master where type='table' order by name")
    table_names = [x[0] for x in c.fetchall()]
    print(table_names)

    # 创建一个表：表的名字为 mytable, 新建 start\end\score三列
    name = "mytable"
    if name not in table_names:
        c.execute("""CREATE TABLE mytable (start, end, score)""")

    # 插入数据之后要commit
    c.execute(f"""INSERT INTO {name} (start, end, score) values(1, 99, 123)""")
    conn.commit()

    # 查询数据
    recs = c.execute("""SELECT * FROM mytable""")
    for row in recs:
        print(row)

    # 删除数据
    start_id = 4
    query = f"""DELETE FROM {name} WHERE start like {start_id}"""  # 查询语句
    c.execute(query)  # 通过 face-id 匹配
    conn.commit()

    # 更新
    # query = "UPDATE face SET createTime=?, retCode=?, retDes=?, faceFeature=?, fileId=?, filePath=?, phone=?" \
    #         "where personId=?"  # 只有 person-id 进行更新
    # c.execute(query, (data['createTime'], data['retCode'], data['retDes'], data['faceFeature'],
    #                   data['fileId'], data['filePath'], data['phone'], data['personId']))
    query = f"""UPDATE {name} set start={2} where end={99}"""
    c.execute(query)
    conn.commit()

    c.close()
    return c, conn


if __name__ == '__main__':
    run('tabelname')
