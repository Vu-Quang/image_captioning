import numpy as np

def openfile(file):
    fhand = open(file, mode='r')
    result = fhand.read()
    fhand.close()
    return result

test = openfile(r'Flickr8k.token.txt')

test = test.split('\n')
print('amount of dataset (number of image+caption pair): ',len(test))

count = 0
# print('' in test) # check if empty line in file
for i in test:
    if i=='':
        count =+1
print('empty line: ', count)
# only one empty line in data file

# take caption from dataset and embedding, using image_name to link caption and image later

def make_cap(data): #input is list of str contains line
    caption  = dict()
    for line in data:
        if len(line):
            line = line.strip()
            line = line.split()
            key = line[0].split('.')
            dict_key = key[0] #take image_name, type str
            cap = line[1:]
            for text in cap:
                if len(text) <2:
                    while text in cap:
                        cap.remove(text) # preprocess some unneed words
            if dict_key in caption:
                dict_val = caption.get(dict_key)
                dict_val.append('startseq ' + ' '.join(cap) + ' endseq') #add caption with start and end of sequence, type str
            else:
                dict_val = ['startseq ' + ' '.join(cap) + ' endseq'] #because value will be replace if new item with same key be added so caption will be store in a list ==> list of str(caption)    
                caption[dict_key] = dict_val
        else: continue
    return caption

caption = make_cap(test)  

print('amount of dataset (number of image+cation)', len(caption)) #each image contain 5 caption ==> it will 1/5 amount of number of line of test data
# print(caption['1001773457_577c3a7d70'])
# 1 empty line in data ==> 40460/5 = 8092 

'''
mục tiêu là encode caption ví dụ gồm 50 từ (từ vector kích thước 50x1 tương ứng mỗi từ là 1 hàng)
sang ma trận mà mỗi từ sẽ ở dạng ma trận cột (kích thước 1x50 do sử dụng glove 50 chiều-dimension),
do đó ma trận encode câu (caption) này sẽ có kích thước 50x50
cách làm là sẽ 
1. tokenize các từ trong caption (đánh stt để mapping từ 1 đến hết toàn bộ các từ xuất hiện) ví dụ được N từ, mỗi từ sẽ là 1 ma trận cột 1xN
2. phải fix độ dài caption đầu vào để fix đc kích thước ma trận đầu ra ==> lấy kích thước chuỗi dài nhất trong số các caption, ví dụ là M (trên vừa ví dụ M=50)
với các caption ngắn hơn độ dài N, cần phải có 1 từ ảo để padding ==> số từ ở mục 1 phải là N+1 ==> mỗi từ 1x(N+1), ma trận đầu vào sẽ là M x (N+1)
3. kích thước đầu ra phải là độ dài caption x số chiều glove = M x 50
do đó, kích thước ma trận parameter sẽ là N+1 x 50
khi đó, mục tiêu là sẽ train (N+1 x 50) để encode N+1 từ kia được chính xác nhất có thể.
do ta sử dụng model train sẵn nên việc cần làm là lấy đầu ra kích thước M x 50 luôn từ caption tất nhiên với điều kiện là toàn bộ N+1 từ phải có trong bộ glove
kết luận 2 cách làm là:
- với mỗi caption ta lại đi tính M x 50 bằng cách lấy từng vector của từng từ trong cation trong glove. với từ không có trong đó, ví dụ như từ ảo padding ở trên
thì sẽ tính là np.zeros(1,50)*0 tức là toàn bộ sẽ là 0 rồi xếp lại với nhau
1 từ: 1x50 ==> M từ Mx50
ta sẽ sử dụng numpy zeros và đi set từng phần tử ở shape[0] là các vector cột của từng từ. (*)
ở đây thời tính toán sẽ là M * số caption * (thời gian mỗi lần đi truy xuất để tìm vector cột)
- tương tự như cách trên nhưng ta sẽ lợi dụng ma trận parameter kích thước N+1 x 50. ở đó, mỗi phần từ kích thước (1x50) chính là encode đầu ra của từ đó
==> tính luôn ma trận parameter, rồi kết hợp ma trận đầu vào ta được M x (N+1) x (N+1) x 50 = M x 50
cách này thực ra trong bước đi xây dựng ma trận parameter vẫn có bước (*) nhưng ta chỉ đi lấy 1 lần và sau đó, encode đầu ra sẽ được tính mà chỉ cần cấp đầu vào dạng M x (N+1)
thời gian tính toán sẽ là N+1 * (thời gian mỗi lần đi truy xuất để tìm vector cột) rồi sau đó việc nhân các vector là k đáng kể
trong đó thời gian đi truy xuất sẽ rất lâu, nên cách thứ 2 là tối ưu hơn. hơn nữa, nếu sử dụng bạn cũng sẽ build ra 1 model train word_embdedding rất có lợi
thêm vào đó, việc dựng ma trận đầu vào được hỗ trợ bởi layer Embedding trong keras qua việc chỉ cần khai báo vector M chiều với vị tí của từ trong caption trùng với
vị trí của nó trong vector và giá trị bằng tokenize của chính nó trong vocab mapping.

Tóm lại, ta cần:
1. tổng số từ xuất hiện trong tất cả caption: vocab_size N - để tạo vector M chiều kèm giá trị các từ trong khoảng [1,vocab]
2. độ dài lớn nhất của caption: M
'''

# 1. đi tạo dict vocab_to_id chứa từ:id mapping N từ ==> gửi N vào biến vocab_size
all_caption = []
max_length = 0 # do dai lon nhat M
for k,v in caption.items():
    for cap in v:
        max_length = max(max_length, len(cap.split()))
        all_caption.append(cap)

# print(all_caption[0])
print('tổng số caption: ', len(all_caption))
print('chiều dài caption lớn nhất: ', max_length)

vocab_to_id = {}
id = 1 #giá trị mapping của từ
for cap_list in all_caption:
    words = cap_list.split()
    for word in words:
        if word not in vocab_to_id:
            vocab_to_id[word] = id
            id += 1
print(vocab_to_id['mountain'])
vocab_size = len(vocab_to_id) + 1 # them 1 tu ao padding
print('số từ của vocab', vocab_size)

# xây dict id_to_vocab ngược lại dùng cho dự đoán từ sau này
id_to_vocab = dict()
for k,v in vocab_to_id.items():
    id_to_vocab[v] = k

print(id_to_vocab[538])

# set parameter matrix
word_dimension = 50
word_matrix = np.zeros((vocab_size, word_dimension))

# lấy các vector cột encode của các từ from bộ glove
f = open(r'glove.6B.50d.txt', encoding = 'utf-8')

glove_dict = dict()
for line in f:
    if line:
        item = line.split()
        k = item[0]
        v = np.asarray(item[1:], dtype = 'float32')
        glove_dict[k] = v

f.close()
print(glove_dict['the'])

# những từ k có trong glove sẽ set là vector 0
for k,v in vocab_to_id.items():
    if k in glove_dict:
        word_matrix[v] = glove_dict[k]

print(word_matrix.shape)

'''
với ma trận trên khi ta input đầu vào với cỡ (batch, 34-M từ) thì đầu ra sẽ ra được encode của cả batch dạng ma trận (batch, 34, 50)
ý tưởng là ta sẽ cho dự đoán từng từ một của câu xuất phát từ mặc định startseq và kết thúc bằng endseq hoặc chiều dài là 34
ví dụ đầu tiên sẽ là startseq + 33 padding ảo ==> dự đoán từ tiếp theo là A
start + A + 33 từ padding ảo ==>...
cứ như thế với mỗi cặp ảnh-caption sẽ tạo được tập dataset rất lớn.
tuy nhiên nhược điểm là những từ trong vocab không có trong glove đều được endcode thành vector zero nên model sẽ hiểu các từ startseq, endseq, padding ảo... là như nhau

với mô hình như thế, ta sẽ train với tập train được lưu dưới dạng các tên ảnh ==> sd caption dict để link ra các caption rồi tạo các caption encode như trên 
kết hợp với encode của ảnh tạo thành train_dataset
như vậy sẽ có những ảnh k có trong tập train ==> sẽ có những caption, từ không có trong vocab khi train sẽ thành thừa tài nguyên, tăng số lượng parameter và kích thước model
nhưng với các ảnh ngoài, ví dụ tập test có thể sẽ có những giá trị của ảnh tạo ra sẽ được hàm softmax tạo ra giá trị từ nằm ngoài train_vocab
do đó, việc để vocab bao gồm cả những từ đó là rất đáng để thử. nhưng trong real world, như thế có nghĩa là ta sẽ thêm càng nhiều từ vào vocab càng tốt, sẽ rất lớn...
'''