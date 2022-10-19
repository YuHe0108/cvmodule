import sqlite3

DATABASE_PATH = "face.db"
COLUMN_NAMES = ["age", "grade", "sex", "score"]


def insert(table_name, data):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # 查询有哪些 表
    c.execute("select name from sqlite_master where type='table' order by name")
    table_names = [x[0] for x in c.fetchall()]
    # 如果插入的表不存在则新建
    if table_name not in table_names:
        c.execute(f"""CREATE TABLE {table_name} (age, grade, sex, score)""")

    # 插入数据
    c.execute(f"""INSERT INTO {table_name} (age, grade, sex, score) values(?,?,?,?)""",
              (data["age"], data["grade"], data["sex"], data["score"]))
    conn.commit()
    c.close()
    return


if __name__ == "__main__":
    insert("scores", {"age": 18, "grade": 3, "sex": 0, 'score': 99})
