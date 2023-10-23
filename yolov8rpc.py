import grpc
from concurrent import futures
import trailer_pb2
import trailer_pb2_grpc

import cv2
from ultralytics import YOLO
import numpy as np
import base64


# Load a pretrained YOLOv8n model



class Servicer(trailer_pb2_grpc.TrailerServicer):
    """Missing associated documentation comment in .proto file."""

    def Init(self, request, context):
        self.model = YOLO(request.kv.decode('utf-8'))        
        return trailer_pb2.Response(code = 0 ,message = 'Init')

    def Start(self, request, context):
        
        return trailer_pb2.Response(code = 1 ,message = 'Start')
    
    def Status(self, request, context):        
        return trailer_pb2.StatusResponse(status  = 2, message="Operation encountered an RUNNING.")
    
    def Service(self, request, context):
        image_base64 = request.args.decode()
        decoded_image = np.frombuffer(base64.b64decode(image_base64), np.uint8)
        decoded_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
        results = self.model(decoded_image)
        for result in results:
            im_array = result.plot()
            base64_string = base64.b64encode(cv2.imencode(".jpg", im_array)[1])
        return trailer_pb2.ServiceResponse(code = 2 ,data = base64_string )
    
    def OnStream(self, request_iterator, context):
        for frame in request_iterator:
            # 处理接收到的视频帧
            image_base64 = frame.data.decode()
            decoded_image = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            decoded_image = cv2.imdecode(decoded_image, cv2.IMREAD_COLOR)
            results = self.model(decoded_image)
            for result in results:
                im_array = result.plot()
                base64_string = base64.b64encode(cv2.imencode(".jpg", im_array)[1])
            # 发送处理后的帧给客户端
            yield trailer_pb2.StreamResponse(data = base64_string)


    def Query(self, request, context):        
        return trailer_pb2.DataRowsResponse(row = [{}] )
    
    def Schema(self, request, context):        
        return trailer_pb2.SchemaResponse(code = 2 ,message = 'Schema' ,columns = [{}])
    
    def Stop(self, request, context):
        return trailer_pb2.Response(code = 2 ,message = 'Stop')

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trailer_pb2_grpc.add_TrailerServicer_to_server(Servicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()