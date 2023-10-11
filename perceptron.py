from random import randint
import numpy

class MyPerceptron:

    ## Hàm Khởi tạo
    def __init__(self, xTrain, yTrain) -> None:
        self.lenXY = len(xTrain)                    ## Lưu số lượng bộ dữ liệu đầu vào
        if self.lenXY != len(yTrain):               ## Nếu số lượng bộ dữ liệu đầu vào khác số lượng nhãn thì báo lỗi 
            raise ValueError('Parameters is not valid!')
        self.X = numpy.array(xTrain)                ## Lưu tập dữ liệu đầu vào
        self.Y = numpy.array([list(map(lambda x: 1 if x == 1 else -1, yTrain))]).T      ## Lưu tập nhãn dạng ma trận 1 cột
        self.w = []                                 ## Khởi tạo bộ trọng số w
        for i in range(len(self.X[0])):
            self.w.append(randint(int(-len(self.X[0]*2)), int(len(self.X[0])*2)))           ## Sinh ngẫu nhiên bộ trọng số w
        self.w = numpy.array([self.w])              ## Lưu lại bộ trọng số dưới dạng ma trận hàng
    
    ## Hàm kiểm tra xem nhãn được được đánh dấu đúng hay sai
    def check(self, x, y):
        xi = numpy.array(x)                         ## Lưu xi
        res = self.w@xi                             ## Tính nhãn dự đoán res = y^ = w*xi
        return True if res*y >= 0 else False        ## Nếu res (giá trị dự đoán) * y (giá trị thực tế) >=0 (cùng phía) thì nhãn được gán đúng

    ## Hàm kiểm tra xem toàn bộ nhãn đã được đánh dấu đúng chưa
    def stop(self):
        for i, v in enumerate(self.X):              ## Duyệt toàn bộ dữ liệu đầu vào
            if not self.check(v, self.Y[i]):        ## Kiểm tra nếu gặp một bộ bị gán nhãn sai thì trả về False và vị trí bộ dữ liệu bị gán sai
                return False, i         
        return True, -1                             ## Chạy được đến đây tức là toàn bộ các bộ dữ liệu đã được gán nhãn đúng

    ## Hàm train mô hình
    def fit(self, eta = 0.001, max_iter=2000):                     
        count = 0                                   ## Khởi tạo biến đếm số lần lặp
        while count < max_iter:                         ## Sử dụng thuật toán gradient descent để tính đạo hàm hàm mất mát = 0, số lần lặp là 1000
            check, index = self.stop()              ## Lấy về giá trị kiểm tra và chỉ số từ hàm stop()
            if check:                               ## Nếu check là True thì kết thúc
                return
            yixi = (numpy.array([self.Y[index]])*numpy.array([self.X[index]]))
            self.w = self.w + eta*yixi              ## Cập nhật lại hàm mất mát với hệ số học eta mặc định = 0.001
            count += 1                              ## Tăng biến đếm

    ## Hàm dự đoán
    def predict(self, xTest):
        test = numpy.array(xTest)                   ## Lưu tập dữ liệu test
        res = (test@self.w.T).T[0]                  ## Tính toán nhãn dự đoán của tập dữ liệu
        return list(map(lambda x: 0 if x < 0 else 1, res))                              ## Biến đổi tập nhãn dự đoán thành dạng số 0 1 rồi trả về