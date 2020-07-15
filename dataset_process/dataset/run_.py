from dataset import dataset_to_tfrecord
import tensorflow as tf
if __name__ == '__main__':
    dataset_dir=r'D:\MyData\zengxf\Downloads\tu\taobao\00 项目\01 目标检测\dataset_process\IMAGE\bird'
    output_dir=r'D:\MyData\zengxf\Downloads\tu\taobao\00 项目\01 目标检测\dataset_process\IMAGE\tfrecords\bird_tfrecord'
    # dataset_to_tfrecord.run(dataset_dir,output_dir,name='train')
    print('done!')
    # tfrecord_path =r'D:\MyData\zengxf\Downloads\tu\taobao\00 项目\01 目标检测\dataset\data'
    #'tfrecord'文件读取
    images, xmin, ymin, xmax, ymax = dataset_to_tfrecord.read_and_decode(output_dir)


    # sess = tf.Session()
    # # sess.run(tf.local_variables_initializer())
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # # tf.train.start_queue_runners(sess=sess)
    # image = sess.run(images)
    # print(images)
