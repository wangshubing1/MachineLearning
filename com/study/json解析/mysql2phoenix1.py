#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Time    : 2018/4/20 10:45
# @Author  : 王树兵
# @Email   : duhanmin@foxmail.com
# @File    : mysql2phoenix.py
# @Software: PyCharm Community Edition


import pymysql

def get_connection(db):
    # 配置文件中配置
    # DB是为了链接而设置，建议默认"mysql"
    connection = pymysql.connect(host='192.168.9.111',
                             user='root',
                             password='123456',
                             db=db,
                             port=3306,
                             charset='utf8'
                             )
    return connection

def get_dbTableNameList(db):
    connection = get_connection(db)
    results = []
    try:
        with connection.cursor() as cursor:
            cursor
            sql = '''SHOW TABLES'''
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in range(len(result)):
                results.append(str(result[i][0]))
    finally:
        connection.close()
    return results

def get_dbNameList(db):
    connection = get_connection(db)
    results = []
    try:
        with connection.cursor() as cursor:
            sql = '''select SCHEMA_NAME from information_schema.schemata'''
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in range(len(result)):
                results.append(str(result[i][0]))
    finally:
        connection.close()
    return results

def get_table_info(tablename):
    db, table = tablename.split(".")
    '''
    table =  为表名，mysql,phoenix表名一致
    schema = 为库名
    '''
    cols = []
    create_head = '''create view if not exists "{0}.{1}"(
ROWKEY varchar primary key, 
'''.format(db, table)

    connection = get_connection(db)

    try:
        # 获取一个游标
        with connection.cursor(cursor=pymysql.cursors.DictCursor) as cursor:
            sql = 'SHOW FULL FIELDS FROM  {0}'.format(table)
            try:
                cout = cursor.execute(sql)  # 返回记录条数
            except:
                print("这个表没有权限！！！" + db +"****"+ sql)
            else:
                for row in cursor:  # cursor.fetchall()
                    # print(row)
                    cols.append("info"+"."+row['Field'])
                    row['Type'] = "varchar"
                    # 暂时不知道Phoenix comment怎么加，待定
                    # comment \'' + row['Comment'] + '\'
                    create_head += "\"" + "info" + "\"." + "\""+ row['Field'] +"\"" + ' ' + row['Type'] +',\n'

    finally:
        connection.close()
    create_str = create_head[:-2] + '\n' + ');'
    # return cols, create_str  # 返回字段列表与你建表语句
    return create_str  # 返回字段列表与你建表语句

def write_file(db):
    filename = 'mysql2phoenix/mysql2phoenix_' + db + '.sql'
    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        tableNameList = get_dbTableNameList(db)
        for tableName in tableNameList:
            create_str = get_table_info(db + "." + tableName)
            f.write(create_str + "\n")

if __name__ == '__main__':
    # 随便给一个库名字是为了和mysql建立连接
    for dbName in get_dbNameList("mysql"):
        write_file(dbName)
