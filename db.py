import numpy as np
import psycopg2
import pymysql
import pymongo
import predicte
from img_retrieval_chemical import ModelAndWeight
import os
from pathlib import Path
import tool

"""
DATABASE_HOST= "192.168.13.213"
DATABASE_PORT= "5432"
DATABASE = "compound"
TABLE_EXACT = "exact_feature"
DATABASE_USER = "mol_hww"
DATABASE_PWD = "cUmY8G3kZ"
"""
DATABASE_HOST_SAVE= "127.0.0.1"
DATABASE_PORT_SAVE= "3306"
DATABASE_SAVE = "molbase_search"
TABLE_EXACT_SAVE = "exact_feature128"
DATABASE_USER_SAVE = "root"
DATABASE_PWD_SAVE = "123456"

DATABASE_HOST_READ= "192.168.13.213"
DATABASE_PORT_READ = 27017
DATABASE_READ = "compound"
TABLE_EXACT_READ = "compound"
DATABASE_USER_READ= "root"
DATABASE_PWD_READ = "123456"

IMAGE_PATH = "E:/image-all/"
FEATURE_SIZE = 128 #512

"""
:param startId: 化合物ID
:param endId: 化合物ID
:return: 
"""
def saveData(startId, endId=0):
    print(tool.Time() + "saveData begin startId:" + str(startId) + " endId:" + str(endId))
    """
    数据来源配置
    """
    client = pymongo.MongoClient(host=DATABASE_HOST_READ, port=DATABASE_PORT_READ, username=None, password=None)
    db = client[DATABASE_READ]
    collection = db[TABLE_EXACT_READ]

    """
    保存数据配置
    """
    #conn = psycopg2.connect(database=DATABASE, user=DATABASE_USER, password=DATABASE_PWD, host=DATABASE_HOST, port=DATABASE_PORT)
    conn = pymysql.connect(DATABASE_HOST_SAVE, DATABASE_USER_SAVE, DATABASE_PWD_SAVE, DATABASE_SAVE, charset='utf8')

    """
    load model and weight
    """
    model = ModelAndWeight()

    pageSize=1000
    while(True):
        #db.getCollection('compound').find({"_id":{"$gt":0}},{id:1,cas:1,structImage:1}).sort({"_id": 1}).limit(5)

        if(endId > 0):
            dataSet = collection.find({"_id":{"$gt":startId, "$lte": endId}},{"id":1,"cas":1,"structImage":1}).sort([("_id",1)]).limit(pageSize)
        else:
            dataSet = collection.find({"_id":{"$gt":startId}},{"id":1,"cas":1,"structImage":1}).sort([("_id",1)]).limit(pageSize)

        print(tool.Time() + "saveData finded startId:" + str(startId) + " endId:" + str(endId) + " total:" + str(dataSet.count()))
        #save
        encodings = []
        for compound in dataSet:
            startId = compound["_id"]
            if("structImage" in compound and compound["structImage"].startswith("http://saasimg.molbase.net")):
                #print(compound)
                filePath = IMAGE_PATH + str(compound["_id"])+".png"
                #downloadImage(compound["structImage"], IMAGE_PATH, str(compound["id"])+".png")
                if Path(filePath).exists():
                    encoding = predicte.img_to_encoding(filePath, model)
                    #if(encoding != None):
                    encodings.append({"compound":compound,"encoding":encoding[0]})

        encoding_count = len(encodings)
        print(tool.Time() + "saveData getted encoding_count:" + str(encoding_count))
        if(encoding_count > 0):
            f_count = len(encodings[0]["encoding"]) #512

            # INSERT INTO exact_feature(id,cas,structImage,result,f1,f2,...f512) VALUES
            # (1,'cas', '',0,.....),
            # (2,'cas', '',0,.....)
            # ON DUPLICATE KEY UPDATE cas=VALUES(cas),structImage=VALUES(structImage),result=VALUES(result), f1=VALUES(f1)....."

            values = ""
            for index in range(encoding_count):
                value = ''
                compound_id = encodings[index]["compound"]["_id"]
                compound_cas = ""
                if("cas" in encodings[index]["compound"]):
                    compound_cas = encodings[index]["compound"]["cas"]
                compound_struct_image = encodings[index]["compound"]["structImage"]
                for value_index in range(f_count):
                    if(value_index == f_count-1):
                        value = ''.join([value, str(encodings[index]["encoding"][value_index])])
                    else:
                        value = ''.join([value, str(encodings[index]["encoding"][value_index]), ','])

                if (index == encoding_count - 1):
                    value = ''.join(["(", str(compound_id), ",'",str(compound_cas), "','",str(compound_struct_image), "',", str(value), ")"])
                else:
                    value = ''.join(["(", str(compound_id), ",'",str(compound_cas), "','",str(compound_struct_image), "',", str(value), "),"])

                values = ''.join([values, value])

            titles = "id,cas,structImage,"
            features = "cas=VALUES(cas),structImage=VALUES(structImage),"
            for index in range(f_count):
                if (index == f_count-1):
                    titles = ''.join([titles, 'f', str(index + 1)])
                    features = ''.join([features, 'f', str(index+1), '=VALUES(f', str(index+1), ')'])
                else:
                    titles = ''.join([titles, 'f', str(index + 1), ','])
                    features = ''.join([features, 'f', str(index+1), '=VALUES(f', str(index+1), ')', ','])


            sql = ''.join(["INSERT INTO " , str(TABLE_EXACT_SAVE), " (" ,titles,") VALUES ", values, " ON DUPLICATE KEY UPDATE ", str(features)])
            cur = conn.cursor()
            cur.execute(sql);
            #rows = cur.fetchall()
            # 饭回游标
            #print(rows)
            conn.commit()
            cur.close()
            print(tool.Time() + "saveData saved encoding_count:" + str(encoding_count))

        if(startId >= endId):# while end
            break

    print(tool.Time() + "saveData endAll endId:" + str(endId))
    conn.close()


from urllib.request import urlretrieve
def downloadImage(image_link, saveDir, fileName):
    #urlretrieve('http://saasimg.molbase.net/mol_command/0db3c8a13a784daabe03b6e69400dfc3.png', 'download-image/6.png')
    # 注意urlretrieve缺点：如果要下载网站设置了反爬虫，就会失败
    # 方式二： urllib.request.urlopen
    return urlretrieve(image_link, saveDir + fileName)

def downloadImageStart(startId, n=30000):
    """
    获取数据
    """
    client = pymongo.MongoClient(host=DATABASE_HOST_READ, port=DATABASE_PORT_READ, username=None, password=None)
    db = client[DATABASE_READ]
    collection = db[TABLE_EXACT_READ]
    pageSize=1000
    total = 0
    while(total < n):
        #db.getCollection('compound').find({"_id":{"$gt":0}},{id:1,cas:1,structImage:1}).sort({"_id": 1}).limit(5)
        dataSet = collection.find({"_id":{"$gt":startId}},{"id":1,"cas":1,"structImage":1}).sort([("_id",1)]).limit(pageSize)
        if(dataSet.count() == 0):
            break
        else:
            #save
            for compound in dataSet:
                if (total >= n):
                    break
                startId = compound["_id"]
                if(compound["structImage"].startswith("http://saasimg.molbase.net")):
                    #print(compound)
                    saveFilePath = IMAGE_PATH + str(compound["_id"])+".png"
                    result = downloadImage(compound["structImage"], IMAGE_PATH, str(compound["_id"])+".png")
                    a = result[0]
                    if(result[0] == saveFilePath):
                        total = total+1

        print(tool.Time() + "downloadImage current total:" + str(total))


    print(tool.Time() + "downloadImage end total:"+ str(total))



def who_is_it(encoding):
    margin = 0.5
    min_dist = 100
    distance = ""
    f_count = len(encoding)

    for f_index in range(f_count):
        # distance = K.sqrt(K.sum(K.pow(l - r, 2), 1, keepdims=True))
        l_f = ''.join(["POWER(", str(encoding[f_index]), "-", "a.f", str(f_index+1), ",2)"])
        if (f_index == f_count - 1):
            distance = ''.join([distance, l_f])
        else:
            distance = ''.join([distance, l_f, "+"])

    distance = ''.join(["SQRT","(", distance, ")"])
    sql = "select a.id, a.cas, ("+ distance +") as score\n" \
        "from exact_feature as a\n" \
        "order by score asc\n" \
        "limit 1"
    conn = pymysql.connect(DATABASE_HOST_SAVE, DATABASE_USER_SAVE, DATABASE_PWD_SAVE, DATABASE_SAVE, charset='utf8')
    cur = conn.cursor()
    cur.execute(sql);
    rows = cur.fetchall()
    print(rows)
    conn.commit()
    cur.close()
    conn.close()




if __name__ == "__main__":
    print("================db.py start ==================")
    saveData(5, 86114)
    #downloadImageStart(70614, n=4793)

    # model = ModelAndWeight()
    # encoding = predicte.img_to_encoding_2("image-test/15-web-cut-analysis.png", model)
    # who_is_it(encoding)

    print("")














