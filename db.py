import numpy as np
import psycopg2
import pymysql
import predicte
import os

"""
DATABASE_HOST= "192.168.13.213"
DATABASE_PORT= "5432"
DATABASE = "compound"
TABLE_EXACT = "exact_feature"
DATABASE_USER = "mol_hww"
DATABASE_PWD = "cUmY8G3kZ"
"""
DATABASE_HOST= "127.0.0.1"
DATABASE_PORT= "3306"
DATABASE = "molbase_search"
TABLE_EXACT = "exact_feature"
DATABASE_USER = "root"
DATABASE_PWD = "123456"

def saveData():
    #conn = psycopg2.connect(database=DATABASE, user=DATABASE_USER, password=DATABASE_PWD, host=DATABASE_HOST, port=DATABASE_PORT)
    conn = pymysql.connect(DATABASE_HOST, DATABASE_USER, DATABASE_PWD, DATABASE, charset='utf8')
    # 连接配置
    cur = conn.cursor()
    #cur.execute("SELECT * FROM table1 LIMIT 10")
    model = predicte.ModelAndWeight()
    encodings = []
    encodings.append(predicte.img_to_encoding_2("E:/image-all/1.png", model))
    encodings.append(predicte.img_to_encoding_2("E:/image-all/2.png", model))
    encoding_count = len(encodings)
    value_count = len(encodings[0])

    # sql_0 = "INSERT INTO " + str(TABLE_EXACT) + " VALUES " \
    #     "(" + str(id) + ", " + str(encoding) + "), " \
    #     "(" + str(id2) + ", \'" + str(encoding) + "\') " \
    #     "ON DUPLICATE KEY UPDATE feature = VALUES(feature), f1=VALUES(f1)"

    values = ""
    for index in range(encoding_count):
        value = ''
        for value_index in range(value_count):
            if(value_index == value_count-1):
                value = ''.join([value, str(encodings[index][value_index])])
            else:
                value = ''.join([value, str(encodings[index][value_index]), ','])

        if (index == encoding_count - 1):
            value = ''.join(["(" , str(index+1) , ", " , str(value) , ")"])
        else:
            value = ''.join(["(", str(index+1), ", ", str(value), "),"])

        values = ''.join([values, value])

    titles = "id,"
    features = ""
    for index in range(value_count):
        if (index == value_count-1):
            titles = ''.join([titles, 'f', str(index + 1)])
            features = ''.join([features, 'f', str(index+1), '=VALUES(f', str(index+1), ')'])
        else:
            titles = ''.join([titles, 'f', str(index + 1), ','])
            features = ''.join([features, 'f', str(index+1), '=VALUES(f', str(index+1), ')', ','])


    sql = ''.join(["INSERT INTO " , str(TABLE_EXACT), " (" ,titles,") VALUES ", values, " ON DUPLICATE KEY UPDATE ", str(features)])
    cur.execute(sql);
    rows = cur.fetchall()
    # 饭回游标
    print(rows)
    conn.commit()
    cur.close()
    conn.close()



if __name__ == "__main__":
    print("================db.py start ==================")
    saveData()

    print("")














