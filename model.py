from keras import Input, Model
from keras.layers import Dropout, Dense, Embedding, LSTM, add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from caption_encode import *
from image_encode import train_img, test_img
from pickle import load
from numpy import argmax, array
import matplotlib.pyplot as plt

# Tạo model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, word_dimension, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([word_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.load_weights('self_model.h5')
'''
tạo tập training ([X1,X2], y) với
X1: encoded img
X2: tokenized caption
y: next word of caption (category vector)

từ dict caption trong caption_encode ==> caption value xây X2, y còn key xây (link) ra X1 từ encoded_train_img dict trong file pkl
'''

train_img_encode = load(open('encoded_train_img.pkl', mode='rb'))
test_img_encode = load(open('encoded_test_img.pkl', mode='rb'))

def train_data(img_list, img_encoded):
    X1_list, X2_list, y_list = [], [], [] # vì không tính trước được độ lớn của toàn bộ tập train nên sẽ sd list rồi convert sang array
    n = 0
    for img in img_list:
        n +=1
        img_encode = img_encoded[img]
        caption_list = caption[img]
        for capt in caption_list:
            word_list = capt.split()
            word_list_token = [vocab_to_id[word] for word in word_list]
            for i in range(1, len(word_list_token)):
                in_seq = word_list_token[:i]
                out_seq = word_list_token[i]
                out_seq = to_categorical(out_seq, num_classes=vocab_size) # convert a int number to vector vocab_size dimension with only out_seq th = 1
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0] # padding behind 2d list to 2d array with same lenth at shape[1]=max_length
                y_list.append(out_seq)
                X2_list.append(in_seq)
                X1_list.append(img_encode)
        if n == 6:
            yield [array(X1_list), array(X2_list)], array(y_list)
            n = 0
            X1_list, X2_list, y_list = [], [], []

for i in range(5):
    data_generator = train_data(train_img, train_img_encode)
    history = model.fit(data_generator, epochs=1, steps_per_epoch=1000)

model.save_weights('self_model.h5')

'''
test trên tập test_img
cách thức là chọn 1 số trong 1--> 1000 (number of dataset) rồi lấy ảnh trong đó ==> in caption
lưu ý rằng: hàm Input() tự động thêm 1 chiều để có dang (batch,..) 
nên khi tạo data train ta đang tạo từng thành phần của batch ta phải xử lý đầu vào như trên để có dạng input mà chưa tính batch
==> khi train, model sẽ tự fit theo dạng ma trận theo batch_size mà ta chọn ==> sẽ đảm bảo đầu vào dạng (batch,..)
dó đó, với khi test sẽ phải cấp input theo dạng (1,..)
'''
print(len(test_img))
while i:= input('nhap vao so anh muon test: '):
    img = test_img[int(i)-1]
    print(img)
    img_encode = test_img_encode[img].reshape((1, 2048))
    cap_return = 'startseq'
    while len(cap_return.split()) < max_length:
        cap_return_id = [vocab_to_id[cap_return] for cap_return in cap_return.split()]
        input_cap = pad_sequences([cap_return_id], maxlen=max_length)
        result = model.predict([img_encode, input_cap])
        result = argmax(result)
        result = id_to_vocab[result]
        cap_return = cap_return + ' ' + result
        if result == 'endseq':
            break
    final_caption = cap_return.split()
    final_caption = ' '.join(final_caption[1:-1])
    img = plt.imread('Images/' + img + '.jpg')
    plt.imshow(img)
    plt.show()
    print(final_caption)
