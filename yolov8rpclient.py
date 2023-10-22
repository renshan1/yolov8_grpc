import logging
import trailer_pb2
import trailer_pb2_grpc
import numpy as np
import grpc
import cv2
import base64

import time

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = trailer_pb2_grpc.TrailerStub(channel)

        #传入模型
        response1 = stub.Init(trailer_pb2.Config(kv = b'models/yolov8s.pt'))
        print(response1.code,response1.message)

        response2 = stub.Start(trailer_pb2.Request())
        print(response2.code,response2.message)

        response3 = stub.Status(trailer_pb2.Request())
        print(response3.status,response3.message)
        #传入图片
        image_path = "../yolov8/detect_data/bus.jpg"
        image = cv2.imread(image_path)
        image_base64 = base64.b64encode(cv2.imencode(".jpg", image)[1])
        response4 = stub.Service(trailer_pb2.ServiceRequest(cmd = b'begin', args = image_base64))
        im_base64 = response4.data.decode()
        decoded_image = np.frombuffer(base64.b64decode(im_base64), np.uint8)
        decoded_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
        # 显示图像
        cv2.imshow('Image', decoded_image)
        cv2.waitKey(0)  # 保持图像窗口打开，直到按下任意键
        cv2.destroyAllWindows()  # 关闭图像窗口

        response5 = stub.Query(trailer_pb2.DataRowsRequest(query = b'a'))
        print(response5.row)

        response6 = stub.Schema(trailer_pb2.SchemaRequest())
        print(response6.code,response6.message,response6.columns)

        response7 = stub.Stop(trailer_pb2.Response())
        print(response7.code,response7.message)

if __name__ == "__main__":
    logging.basicConfig()
    run()