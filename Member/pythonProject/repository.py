from pandas.io import sql

from flask_openCV import cursor


def findByNickName(nickName):
    sql = """SELECT member_id FROM member where nick_name = '%s' """ % (nickName)
    cursor.execute(sql)
    row = cursor.fetchone()

    # E : NO MEMBER
    if (row == None):
        return {"code": 504}
    return {"code": 200, "memberId" : row[0]}
