import os
import io
import xml.etree.ElementTree as ET
import tensorflow as tf
from dataset.dataset_config import DIRECTORY_ANNOTATIONS,DIRECTORY_IMAGES,NUM_IMAGES_TFRECORD
from dataset.utils.dataset_utils import bytes_feature,int64_feature,float_feature
def _convert_to_example(img,img_shape,labels,trunacteds,difficults,bndbox_size):
    '''将一张图片使用example，转换成protobuffer 格式
    :param img:
    :param img_shape:
    :param labels:
    :param trunacteds:
    :param difficults:
    :param bndbox_size:
    :return:
    '''
    # 为了转换需求，bbox由单个obj的四个位置值，
    # 转变成四个位置的单独列表
    # 即：[[12,120,330,333],[50,60,100,200]]————>[[12,50],[120,60],[330,100],[333,200]]
    ymin=[]
    xmin=[]
    ymax=[]
    xmax=[]
    print(bndbox_size)
    for b in bndbox_size:
        ymin.append(b[0])
        xmin.append(b[1])
        ymax.append(b[2])
        xmax.append(b[3])
    image_format=b'JPEG'
    print(type(img))
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    print('label',type(labels))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(img_shape[0]),
        'image/width': int64_feature(img_shape[1]),
        'image/depth': int64_feature(img_shape[2]),
        'image/shape': int64_feature(img_shape),
        # 'image/filename': bytes_feature(filename),
        # 'image/source_id': bytes_feature(filename),
        'image/data': bytes_feature(img),  # 二进制数据
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/trunacteds': int64_feature(trunacteds),
        'image/object/bbox/difficults': int64_feature(difficults),
        'image/object/class/text': bytes_feature(labels),  # label name
        # 'image/object/class/label': bytes_feature(classes),  # label num
    }))
    return example

def _process_image(dataset_dir,img_name):
    '''
    读取图像和xml文件
    :param dataset_dir:
    :param img_name:
    :return:
    '''
    #1.读取图像
    #图像路径
    img_path = os.path.join(dataset_dir,DIRECTORY_IMAGES,img_name+'.jpg')
    img = tf.gfile.FastGFile(img_path,'rb').read()#tensorflow读取图像
    #2.读取xml
    #xml路径
    xml_path =os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS,img_name+'.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()#'annotation'标签
    # 2.1获取图像尺寸信息
    size = root.find('size')
    img_shape=[
        int(size.find('height').text),
        int(size.find('width').text),
        int(size.find('depth').text)
    ]
    #2.2 获取bounding box 相关信息
    # bounding box可能有多个,用多个列表存储相关信息。
    labels = []
    trunacteds=[]
    difficults = []
    bndbox_sizes=[]
    bboxes = root.findall('object')
    for obj in bboxes:
        label = obj.find('name').text.encode(encoding='ascii')
        if obj.find('trunacted'):
            trunacted = int(obj.find('trunacted').text)
        else:
            trunacted = 0
        if obj.find('difficult'):
            difficult = int(obj.find('difficult').text)
        else:
            difficult = 0
        bndbox = obj.find('bndbox')
        bndbox_size=(
            float(bndbox.find('ymin').text) / img_shape[0],
            float(bndbox.find('xmin').text) / img_shape[1],
            float(bndbox.find('ymax').text) / img_shape[0],
            float(bndbox.find('xmax').text) / img_shape[1]
        )


        labels.append(label)
        trunacteds.append(trunacted)
        difficults.append(difficult)
        bndbox_sizes.append(bndbox_size)
    return img,img_shape,labels,trunacteds,difficults,bndbox_sizes


def _add_to_tfrecord(dataset_dir,img_name,tfrecord_writer):
    '''
    读取图片和xml文件，保存成一个Example
    :param dataset_dir:根目录
    :param img_name:图像名称
    :param tfrecord_writer:
    :return:
    '''
    #1.读取图片内容及相应的xml文件
    img, img_shape, labels, trunacteds, difficults, bndbox_size=_process_image(dataset_dir,img_name)

    #2.读取的内容封装成Example,
    example = _convert_to_example(img,img_shape,labels,trunacteds,difficults,bndbox_size)
    #3.Example序列化结果写入指定tfrecord文件
    tfrecord_writer.write(example.SerializeToString())
    return img,img_shape,labels,trunacteds,difficults,bndbox_size

def _get_output_tfrecord_name(output_dir,name,fdx):
    """

    :param output_dir:
    :param name:
    :param fdx:第几个tfrecord文件
    :return:
    """
    return os.path.join(output_dir,'%s_%06d'%(name,fdx)+'.tfrecord')

def run(dataset_dir,output_dir,name='train'):
    """
    运行转换代码逻辑。
    存入多个tfrecord文件，每个文件固定N个样本
    :param dataset_dir:数据集目录，包含annotations,jpeg文件夹
    :param output_dir:tfrecords存储目录
    :param name:数据集名字，指定名字以及train or test
    :return:
    """
    # 1. 判断数据集目录是否存在，创建一个目录
    if tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    # 输出路径需要已存在
    # if tf.gfile.Exists(output_dir):
    #     tf.gfile.MakeDirs(output_dir)
    # 2. 读取某个文件夹下的所有文件名字列表
    dir_path = os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS)
    files_path = sorted(os.listdir(dir_path))
    # 3. 循环名字列表，
    # 每200个图片及xml文件存储到一个tfrecord文件中
    num = len(files_path)
    i = 0
    fdx = 0
    while i < num:
        tf_record_name = _get_output_tfrecord_name(output_dir,name,fdx)
        with tf.python_io.TFRecordWriter(tf_record_name) as tf_record_writer:
            j = 0
            while i<num and j < NUM_IMAGES_TFRECORD:
                xml_path = files_path[i]
                img_name = xml_path.split('.')[0]
                #每个图像构建一个Example,保存到tf_record_name中
                img, img_shape, labels, trunacteds, difficults, bndbox_size=_add_to_tfrecord(dataset_dir,img_name,tf_record_writer)
                # print(img,img_shape,labels,trunacteds,difficults,bndbox_size)
                j += 1
                i += 1

        fdx += 1
    print('数据集%s转换成功'%(dataset_dir))


def read_and_decode(tfrecord_path):
    # 生成文件名列表
    filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle = False)
    #构建文件阅读器
    reader = tf.TFRecordReader()
    #读取文件名
    _, serialized_example = reader.read(filename_queue)
    ### features 的 key 必须和 写入时 一致，数据类型也必须一致，shape 可为 空
    features = tf.parse_single_example(serialized_example,features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/depth': tf.FixedLenFeature([], tf.int64),
        'image/shape': tf.FixedLenFeature([], tf.int64),
        # 'image/filename': bytes_feature(filename),
        # 'image/source_id': bytes_feature(filename),
        'image/data': tf.FixedLenFeature([], tf.string),  # 二进制数据
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
        'image/object/bbox/trunacteds': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/difficults': tf.FixedLenFeature([], tf.int64),
        'image/object/class/text': tf.FixedLenFeature([], tf.string),  # label name
        # 'image/object/class/label': bytes_feature(classes),  # label num
    })

    images = tf.decode_raw(features['image/data'], tf.uint8)
    # images = tf.reshape(images, [1200, 1600, 3])
    xmin = features['image/object/bbox/xmin']
    ymin = features['image/object/bbox/ymin']
    xmax = features['image/object/bbox/xmax']
    ymax = features['image/object/bbox/ymax']
    trunacteds = features['image/object/bbox/trunacteds']
    difficults = features['image/object/bbox/difficults']
#    img_posX = tf.cast(features['img_posX'], tf.int32)
#    img_posY = tf.cast(features['img_posY'], tf.int32)
#    img_posR = tf.cast(features['img_posR'], tf.int32)
#     if is_train == True:
#        img_raw, labelX, labelY, labelR = tf.train.shuffle_batch([images, img_posX, img_posY, img_posR],
#                                                  batch_size = 1,
#                                                  capacity = 3,
#                                                  min_after_dequeue = 3)
#     else:
#         img_raw, labelX, labelY, labelR = tf.train.batch([images, img_posX, img_posY, img_posR],
#                                                  batch_size = 1,
#                                                  capacity = 3)
    return images,xmin,ymin,xmax,ymax
