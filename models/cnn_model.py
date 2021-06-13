# coding:utf-8
# 语义匹配cnn孪生网络模型

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.optimizers import *
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

path = '/mnt/yardcephfs/mmyard/g_wxg_ob_dc/mlhustxiao/zixun_tag/tag_extraction/data/'
word_embedding_matrix = np.load(path + '/cache/word_embedding_matrix.npy')
keyword_embedding_matrix = np.load(path + '/cache/keyword_embedding_matrix.npy')
title_embedding_matrix = np.load(path + '/cache/title_embedding_matrix.npy')

print(word_embedding_matrix[:50])
print(keyword_embedding_matrix[:50])
print(title_embedding_matrix[:50])

train_q1_word_seq = np.load(path + '/cache/train_q1_word_seq.npy')
train_q2_word_seq = np.load(path + '/cache/train_q2_word_seq.npy')
train_q3_word_seq = np.load(path + '/cache/train_q3_word_seq.npy')
train_q4_word_seq = np.load(path + '/cache/train_q4_word_seq.npy')

val_q1_word_seq = np.load(path + '/cache/val_q1_word_seq.npy')
val_q2_word_seq = np.load(path + '/cache/val_q2_word_seq.npy')
val_q3_word_seq = np.load(path + '/cache/val_q3_word_seq.npy')
val_q4_word_seq = np.load(path + '/cache/val_q4_word_seq.npy')

train_y = np.load(path + '/cache/train_y.npy')
val_y = np.load(path + '/cache/val_y.npy')

print(train_q1_word_seq.shape)
print(train_q2_word_seq.shape)
print(train_q3_word_seq.shape)
print(train_q4_word_seq.shape)

print(val_q1_word_seq.shape)
print(val_q2_word_seq.shape)
print(val_q3_word_seq.shape)
print(val_q4_word_seq.shape)
# print(val_q5_word_seq.shape)

print(train_y.shape)
print(val_y.shape)

MAX_WORD_SEQUENCE_LENGTH = 800
MAX_TITLE_SEQUENCE_LENGTH = 50
MAX_KEYWORD_SEQUENCE_LENGTH = 1


def cnn_model():
    embedding_layer1 = Embedding(name="word_embedding",
                                 input_dim=word_embedding_matrix.shape[0],
                                 weights=[word_embedding_matrix],
                                 output_dim=word_embedding_matrix.shape[1],
                                 trainable=False)
    embedding_layer2 = Embedding(name="keyword_embedding",
                                 input_dim=keyword_embedding_matrix.shape[0],
                                 weights=[keyword_embedding_matrix],
                                 output_dim=keyword_embedding_matrix.shape[1],
                                 trainable=False)
    embedding_layer3 = Embedding(name="name_embedding",
                                 input_dim=title_embedding_matrix.shape[0],
                                 weights=[title_embedding_matrix],
                                 output_dim=title_embedding_matrix.shape[1],
                                 trainable=False)

    q1_word = Input(shape=(MAX_WORD_SEQUENCE_LENGTH,), dtype="int32")
    q1_word_embed = embedding_layer1(q1_word)

    q2_word = Input(shape=(MAX_KEYWORD_SEQUENCE_LENGTH,), dtype="int32")
    q2_word_embed = embedding_layer2(q2_word)
    q2_word_embed = Flatten()(q2_word_embed)

    q3_word = Input(shape=(MAX_TITLE_SEQUENCE_LENGTH,), dtype="int32")
    q3_word_embed = embedding_layer3(q3_word)

    q4_word = Input(shape=(13,), dtype="float32")
    q4_word_dense = Dense(100, activation="relu")(q4_word)

    conv1 = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')
    # conv3 = Conv1D(filters=100, kernel_size=1, padding='same', activation='relu')

    content = conv1(q1_word_embed)
    c = GlobalMaxPooling1D()(content)

    title = conv2(q3_word_embed)
    t = GlobalMaxPooling1D()(title)

    w = Dense(100, activation="relu")(q2_word_embed)

    diff1 = Lambda(lambda x: K.abs(x[0] - x[1]))([c, w])
    mul1 = Lambda(lambda x: x[0] * x[1])([c, w])

    diff2 = Lambda(lambda x: K.abs(x[0] - x[1]))([t, w])
    mul2 = Lambda(lambda x: x[0] * x[1])([t, w])

    merged = concatenate([t, diff1, mul1, diff2, mul2, q4_word_dense])
    # merged = Dropout(0.3)(merged)

    merged = BatchNormalization()(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[q1_word, q2_word, q3_word, q4_word], outputs=out)
    # model = multi_gpu_model(model, 4)
    model.compile(loss='binary_crossentropy', optimizer='nadam')
    return model


if __name__ == '__main__':
    model = cnn_model()
    print(model.summary())

    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    best_model_path = path + "/cache/zixun_cnn_model_weights" + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

    print('train cnn model...')
    hist = model.fit(
        [train_q1_word_seq, train_q2_word_seq, train_q3_word_seq, train_q4_word_seq],
        train_y,
        validation_data=(
            [val_q1_word_seq, val_q2_word_seq, val_q3_word_seq, val_q4_word_seq], val_y),
        epochs=4,
        batch_size=1024,
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1)
    model.load_weights(best_model_path)
    time_stamp = datetime.datetime.now()
    model.save(path + '/cache/cnn-model-{}.h5'.format(time_stamp.strftime('%m-%d-%H-%M')))
