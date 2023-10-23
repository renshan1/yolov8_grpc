# yolov8_demo
这是一个利用GRPC实现：
在客户端负责选择检测模型与需要检测的图片，在服务器端实现目标的检测并传回检测结果的程序
# 编写接口
在.proto文件中，service类下写rpc方法。rpc方法有输入输出，在service类后可以定义输入输出的数据结构。
# 生成grpc代码
```sh
python -m grpc_tools.protocs -I./protos --python_out=. --grpc_python_out=. ./
```
# 服务端
import入生成的两个grpc代码，定义一个类Service，在里面重写方法（函数）。方法serve：启动一个 gRPC 服务器的代码。

# 客户端
定义一个run函数，负责连接服务器，然后调用trailer_pb2_grpc.TrailerStub类，即可以调用服务器中对应的重写方法，进行数据的传入与结果的传出。