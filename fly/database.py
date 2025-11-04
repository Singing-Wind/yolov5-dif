# pip install mysql-connector-python
import mysql.connector
from mysql.connector import errorcode, pooling
from mysql.connector import pooling
from datetime import datetime
import numpy as np
import cv2
import threading
import json
import time

# MySQL 配置
from pathlib import Path
def get_database_config():
    try:
        with open(Path(__file__).parent / 'database.json', 'r', encoding='utf-8') as fr:
            config = json.load(fp=fr)
    except Exception:
        with open(Path(__file__).parent / 'database.json', 'r', encoding='gbk') as fr:
            config = json.load(fp=fr)
    try:
        host, user, pawd = config['host'], config['user'], config['password']
    except:
        raise RuntimeError('无法获取"config/database.json"配置连接MYSQL数据库')
    return host, user, pawd

HOST, USER, PAWD = get_database_config()

class DATABASE(object):
    POOLSIZE = 5
    poollock = threading.Semaphore(POOLSIZE)
    config = {
        'pool_name': "mypool",
        'pool_size': POOLSIZE,
        'host': HOST,
        'user': USER,
        'password': PAWD,
        'database': 'fly_database',
        'charset':'utf8mb4'  # 确保连接时使用 utf8mb4 字符集
    }
    
    tables = {
        'results_table':
"""
CREATE TABLE IF NOT EXISTS results_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    side INT DEFAULT 0,
    fly_idx VARCHAR(50) DEFAULT 0,
    target_category VARCHAR(50),
    cx DOUBLE DEFAULT 0,
    cy DOUBLE DEFAULT 0,
    a DOUBLE DEFAULT 0,
    b DOUBLE DEFAULT 0,
    angle DECIMAL(5, 2),
    target_confidence DECIMAL(5, 4),
    bounding_box VARCHAR(255),
    image_name VARCHAR(255),
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
    }

    def __init__(self):
        self.connect = None
        self.conn = self.connect_to_mysql
        self.init_flag = False
        self.pool = None
        self.conn()
        assert self.check_and_create_table(), '连接数据库失败'

    def __del__(self):
        if self.pool:
            self.pool.pool_reset()

    # 连接到数据库
    def connect_to_mysql(self):
        if not self.init_flag:
            try:
                conn = mysql.connector.connect(**self.config)
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_BAD_DB_ERROR:
                    print("Database does not exist.")
                    # 创建数据库
                    self.create_database()
                else:
                    print(f"Error: {err}")
                conn = None
            if conn:
                conn.close()
            self.init_flag = True
            self.pool = pooling.MySQLConnectionPool(**self.config)
        if self.pool:
            self.poollock.acquire()
            return self.pool.get_connection()
        else:
            None
    
    def release_conn(self, conn):
        conn.close()
        self.poollock.release()

    # 创建数据库
    def create_database(self):
        conn = mysql.connector.connect(host=self.config['host'], user=self.config['user'], password=self.config['password'], charset=self.config['charset'])
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {self.config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cursor.close()
        conn.close()

    # 检查表是否存在的函数, 创建表
    def check_and_create_table(self):
        conn = self.conn()
        if conn:
            cursor = conn.cursor()
            for k, v in self.tables.items():
                cursor.execute(f"SHOW TABLES LIKE '{k}'")
                result = cursor.fetchone()
                
                if result:
                    print(f"表 '{k}' 已存在")
                else:
                    print(f"表 '{k}' 不存在，正在创建...")
                    cursor.execute(v)
                    conn.commit()
                    print(f"表 '{k}' 创建成功")
            
            cursor.close()
            self.release_conn(conn)
            return True
        else:
            self.release_conn(conn)
            return False

    # 插入或更新数据
    def insert_record(self, results, names, fly_idx, image_name):
        # {'time':<20}{'sd':<3}{'cls':<4}{'name':<20}{'conf':<4} {'cx':<7} {'cy':<7} {'a':<6} {'b':<6} {'angle':<8} {'uav':<3} {'map':<12}
        conn = self.conn()
        if conn:
            cursor = conn.cursor()
            times = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            for result in results:
                bounding_box = ', '.join([str(a) for a in result[:8]])
                target_category = names[int(result[9])]
                target_confidence = float(result[8])
                cx, cy, a, b, sin_t, cos_t = result[10:-1]
                angle = np.arctan2(sin_t, cos_t)*180/np.pi
                side = int(result[-1])

                
                # 插入目标表
                cursor.execute("""
                    INSERT INTO results_table (side, fly_idx, target_category, cx, cy, a, b, angle, target_confidence, bounding_box, image_name, time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, ROUND(%s, 2), ROUND(%s, 4), %s, %s, %s)
                """, (side, fly_idx, target_category, float(cx), float(cy), float(a), float(b), float(angle), float(target_confidence), bounding_box, image_name, times))
            
            conn.commit()
            cursor.close()
            self.release_conn(conn)
            return True
        else:
            self.release_conn(conn)
            return False
