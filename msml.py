# from https://github.com/michuanhaohao/keras_reid/blob/master/reid_tripletcls.py
def msml_loss(y_true, y_pred):
    global SN
    global PN
    feat_num = SN*PN # images num
    y_pred = K.l2_normalize(y_pred,axis=1)
    feat1 = K.tile(K.expand_dims(y_pred,axis = 0),[feat_num,1,1])
    feat2 = K.tile(K.expand_dims(y_pred,axis = 1),[1,feat_num,1])
    delta = feat1 - feat2
    dis_mat = K.sum(K.square(delta),axis = 2) + K.epsilon() # Avoid gradients becoming NAN
    dis_mat = K.sqrt(dis_mat)
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)
    positive = K.max(positive)
    negetive = K.min(negetive) 
    a1 = 0.6
    loss = K.mean(K.maximum(0.0,positive-negetive+a1))
    return loss 

# https://github.com/ZH-Lee/MSML_loss_facenet/blob/baf4158ec0ea98af26b7574258c8916650b59789/train_model.py
def msml_loss(anchor ,pos, neg1 , neg2, alpha):

    with tf.variable_scope('msml_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(neg1, neg2)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss,0.0),0)

    return loss
