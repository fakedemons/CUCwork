data_separation.py该文件用以处理fer2013中的数据，分离为emotion.csv和pixels.csv两份数据，识别可生成图片的部分并按照7:3的比例导入进face-images下的训练集（train-set）和测试集（verify-set），在两个集合里生成image-emotion.csv


model-CNN.py该文件用以训练模型，通过已生成好的训练集和测试集进行训练，由于设备限制和时间关系，在代码中调用了GPU进行模型训练以达到节省时间的目的。


launch.py为设计的面部情绪识别系统，通过调用已训练好的模型，对摄像头中检测到的人脸进行情绪识别，并能够根据识别到的情绪进行一定的反馈。为了更加直观地体现识别效果，使用条形图进行实时的情绪图像绘制。在feedback_images文件夹下放入了bravo.jpg、calm.jpg、surprise.jpeg、warm.jpeg、cute.jepg五张图片。当识别结束后，若检测到最多的表情是happy，则调用bravo.jpg输出，若为angry，则输出calm.jpg，若为fear和sad，则输出warm.jpeg，若为surprise，则输出surprise.jpeg，若为disgust，则输出cute.jpeg，若为neutral，则不输出，每五秒输出一次