import logging
import trailer_pb2
import trailer_pb2_grpc
import numpy as np
import grpc
import cv2
import base64
import time



def compress_image(image_path):
    image = cv2.imread(image_path)
    encoded_image = cv2.imencode(".jpg", image)[1]#利用这个编码,可以在后续使用base64编码时是正常大小
    image_base64 = base64.b64encode(encoded_image)#type：byte
    '''
    quilty = 80
    while len(image_base64) > 512000:
        if quilty < 40:
            print("压缩失败")
            break
        encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quilty])[1]
        image_base64 = base64.b64encode(encoded_image)
        quilty = quilty - 5'''

    return trailer_pb2.ServiceRequest(cmd = b'1', args = image_base64)

def transport_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('the video does not exist.')
        return False

    
    ret, frame = cap.read()
    while ret:
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoded_image = cv2.imencode(".jpg", frame)[1]
        image_base64 = base64.b64encode(encoded_image)
        yield trailer_pb2.StreamRequest(data=image_base64) 
        ret, frame = cap.read()
    cap.release()
    print('video read ended.')
    yield trailer_pb2.VideoRequest(f_data=None)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = trailer_pb2_grpc.TrailerStub(channel)

        #传入模型
        response1 = stub.Init(trailer_pb2.Config(kv = b'models/yolov8s_ship.pt'))
        print(response1.code,response1.message)

        response2 = stub.Start(trailer_pb2.Request())
        print(response2.code,response2.message)

        response3 = stub.Status(trailer_pb2.Request())
        print(response3.status,response3.message)

        #传入要检测的资源
        source_path = "../yolov5-master/detect/boattwo.mp4"
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('video', 1280, 480)
        if source_path[-3:] == "jpg":
            response4 = stub.Service(compress_image(source_path))
            im_base64 = response4.data.decode()
            decoded_image = np.frombuffer(base64.b64decode(im_base64), np.uint8)
            decoded_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
            # 显示图像
            cv2.imshow('video', decoded_image)
            cv2.waitKey(0)  # 保持图像窗口打开，直到按下任意键
            cv2.destroyAllWindows()  # 关闭图像窗口

        if source_path[-3:] == "mp4":
            response_video = stub.OnStream(transport_video(source_path))
            for response in response_video:
                im_base64 = response.data.decode()
                decoded_image = np.frombuffer(base64.b64decode(im_base64), np.uint8)
                decoded_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
                cv2.imshow('video',decoded_image)
                # 检查是否按下了键盘上的 'q' 键来退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
       
        response5 = stub.Query(trailer_pb2.DataRowsRequest(query = b'a'))
        print(response5.row)

        response6 = stub.Schema(trailer_pb2.SchemaRequest())
        print(response6.code,response6.message,response6.columns)

        response7 = stub.Stop(trailer_pb2.Response())
        print(response7.code,response7.message)

if __name__ == "__main__":
    logging.basicConfig()
    run()