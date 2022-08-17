# # databse 접근
import pymysql

db = pymysql.connect(host='54.180.134.240',
                     port=3306,
                     user='seokin',
                     password='ghdtjrdls777!',
                     db='WatchMe',
                     charset='utf8')
