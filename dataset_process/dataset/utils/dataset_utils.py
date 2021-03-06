import tensorflow as tf
def int64_feature(value):
    '''
    包装成example proto
    :param value:
    :return:
    '''
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def float_feature(value):
    if not isinstance(value,list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value=value))

def bytes_feature(value):       #生成字符串型的属性
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
