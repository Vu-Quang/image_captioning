'''
encode các ảnh tương ứng với tên được lưu trong caption dict
ý tưởng là sẽ đọc file ảnh rồi cho qua inception model để tạo encode
- đọc và nén ảnh, resize về kích cỡ đầu vào của inception là (299, 299) bằng hàm preprocess của keras
- xử lý đưa về dạng vector (batch, 299, 299)
- cho qua preprocess_input của inception model
- cho qua inception model để encode về vector (2048, )
- lưu lại dưới dạng dict image: encode

và do quá trình này tốn nhiều thời gian nên sẽ làm trước khi train, test rồi lưu dưới file pkl, sau này sử dụng sẽ chỉ việc load file rồi lấy encode theo image
'''

import numpy as np
from keras. preprocessing import image
from pickle import dump
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.models import Model
# from caption_encode import caption

# tạo model mới bằng inception  bỏ layer cuối
model = InceptionV3(weights='imagenet')
model = Model(model.input, model.layers[-2].output)

# hàm load ảnh sử dụng hàm load_image của keras
def load_img(img):
    img_fea = image.load_img('Images/' + img + '.jpg', target_size = (299, 299))
    # print(img_fea)
    input_array = image.img_to_array(img_fea)
    # print(input_array.shape)
    input_array = np.array([input_array]) #to batch size (batch, 299,299)
    # print(input_array.shape)
    result = preprocess_input(input_array)
    # print(result.shape)
    result = model.predict(result)
    # print(result.shape)
    return np.reshape(result, result.shape[1])

# lưu giá trị encode
img_encode = dict()

test = '1001773457_577c3a7d70'
img_encode[test] = load_img(test)
# print(img_encode[test].shape)

'''
đầu vào của ta là ảnh có kích cỡ bất kì, được mở bởi hàm load_img và trả về một đối tượng PIL, đấy là format đối tượng ảnh dưới dạng kỹ thuật số
là các giá trị 3 kênh RGB trên từng điểm ảnh, kích cỡ ảnh (số các điểm ảnh) là target_size của hàm load_img
để sử dụng được giá trị trên ta dùng hàm img_to_array để có ma trận (w, h, 3) với w*h là kích cỡ ảnh ở trên và 3 biểu thị tại mỗi điểm ảnh ta có 3 giá trị rgb, các gtri này từ 0-255
để train được nhiều cặp input-ouput phải thêm 1 chiều để có dạng (batch,...), sd np.array([]) hoặc expand_dims()
tiếp theo cho qua preprocess của inception model của model, tại đấy có thể thấy đầu vào được normalize và được thực hiện các kỹ thuật khác
sau khi predict để cho ra giá trị encode (mặc định shape đầu ra theo inception là (1, 2048)) ta chuyển về dạng vector cột để tiện cho việc input vào model dưới dạng (batch,..)

tới đây, ta sẽ chia ra encode training set và test set để lưu vào 2 dict khác nhau
'''

# training set
train_img_encode = dict()

# lấy thông tin các ảnh của tập train (tên ảnh)
def dataset_info(file):
    container = []
    fhand = open(file)
    for line in fhand:
        if line:
            line = line.split('.')
            container.append(line[0])
    fhand.close()
    return container

train_img = dataset_info('Flickr_8k.trainImages.txt')

for img in train_img:
    train_img_encode[img] = load_img(img)

if len(train_img_encode):
    print('encode thành công tập train gồm {} ảnh thành {} giá trị trong dictionary'.format(len(train_img), len(train_img_encode)))

# tiến hành lưu vào file pkl qua thư viện pickle
fhand = open('encoded_train_img.pkl', mode='wb')
dump(train_img_encode, fhand)
fhand.close()

# tương tự với tập test
test_img_encode = dict()
test_img = dataset_info('Flickr_8k.testImages.txt')

for img in test_img:
    test_img_encode[img] = load_img(img)

if len(test_img_encode):
    print('encode thành công tập test gồm {} ảnh thành {} giá trị trong dictionary'.format(len(test_img), len(test_img_encode)))

# tiến hành lưu vào file pkl qua thư viện pickle
fhand = open('encoded_test_img.pkl', mode='wb')
dump(test_img_encode, fhand)
fhand.close()