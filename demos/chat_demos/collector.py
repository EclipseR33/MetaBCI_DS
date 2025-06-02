from socket import *

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def collecting_all_info(user_name, user_info):
    udp_port_Online_emotion = 4023
    tcp_port_chatchat = 4024

    Online_emotion_server = socket(AF_INET, SOCK_DGRAM)
    # chatchat_server = socket(AF_INET, SOCK_DGRAM)

    # Online_emotion_server.bind(('192.168.31.10', udp_port_Online_emotion))
    Online_emotion_server.bind(('172.20.10.3', udp_port_Online_emotion))
    # chatchat_server.bind(('127.0.0.1', tcp_port_chatchat))
    # chatchat_server.listen(5)

    while True:
        emotion, clientAddress = Online_emotion_server.recvfrom(1024)
        Online_emotion_server.sendto("got it".encode(), clientAddress)
        if emotion:
            Online_emotion_server.close()
            break
    emotion = emotion.decode()
    print(emotion)
    if emotion == "happy":
        emotion_prop = ",现在心情不错~"
    elif emotion == "nervous":
        emotion_prop = ",现在有点紧张..."
    else:
        emotion_prop = ",现在心情还算平静."
    db = SQLAlchemy()
    class Current(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(80), unique=True, nullable=False)
        info = db.Column(db.String(200), unique=False, nullable=False)

    # DATABASE_URI = 'sqlite://///Users/meijiawei/Documents/metabci/MetaBCI/demos/chat_demos/users.db'
    DATABASE_URI = './users.db'

    # 创建数据库引擎
    engine = create_engine(DATABASE_URI)

    # 创建Session类
    Session = sessionmaker(bind=engine)

    # 创建Session实例
    session = Session()

    first_row = session.query(Current).first()

    if first_row:
        first_row.info = first_row.info + emotion_prop
        print(first_row.info)
        # 提交数据库会话
        session.commit()
    while True:
        print("into listening")
        # connectionSocket, addr = chatchat_server.accept()
        # message, addr = chatchat_server.recvfrom(1024)
        # message = chatchat_server.recv(1024)
        # if message.decode() == "user_name":
        #     chatchat_server.send(user_name.encode(), addr)
        # elif message.decode() == "user_info":
        #     chatchat_server.send(user_info.encode(), addr)

    print("finished")

if __name__ == '__main__':
    collecting_all_info("user_name", "user_info")