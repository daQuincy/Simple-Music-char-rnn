# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:10:26 2019

@author: YQ
"""

import tensorflow as tf
from model import build_graph
from model import batch_generator
import config
import pickle
import time
import tqdm

tf_cfg = tf.ConfigProto(allow_soft_placement=True)
tf_cfg.gpu_options.allow_growth = True

tf.reset_default_graph()

data = pickle.load(open("yamaha_train.p", "rb"))
eval_data = pickle.load(open("yamaha_test.p", "rb"))

train_graph = tf.Graph()
eval_graph = tf.Graph()

with train_graph.as_default():
    pitch = tf.placeholder(tf.int32, (None, None))
    velocity = tf.placeholder(tf.int32, (None, None))
    dt = tf.placeholder(tf.int32, (None, None))
    duration = tf.placeholder(tf.int32, (None, None))
    
    noteseq = {"pitch": pitch, "velocity": velocity, "dt": dt, "duration": duration}
    
    rnn = build_graph(noteseq, config, True, False)
    
    global_step = tf.train.create_global_step()
    lr = tf.train.exponential_decay(config.learning_rate, global_step, 8000, 0.989)
    
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    grads, var = zip(*opt.compute_gradients(rnn["loss"]))
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    train_op = opt.apply_gradients(zip(grads, var), global_step=global_step)
    init = tf.global_variables_initializer()
    
    train_saver = tf.train.Saver()
    
    with tf.variable_scope("feature_loss"):
        summaries = [
            tf.summary.scalar("pitch_loss", rnn["losses"].pitch),
            tf.summary.scalar("velocity_loss", rnn["losses"].velocity),
            tf.summary.scalar("dt_loss", rnn["losses"].dt),
            tf.summary.scalar("duration_loss", rnn["losses"].duration)
        ]
    with tf.variable_scope("misc"):
        summaries2 = [
            tf.summary.scalar("loss", rnn["loss"]),   
            tf.summary.scalar("learning_rate", lr)
        ]
        
    train_sum_op = tf.summary.merge([summaries, summaries2])

with eval_graph.as_default():
    epitch = tf.placeholder(tf.int32, (None, None))
    evelocity = tf.placeholder(tf.int32, (None, None))
    edt = tf.placeholder(tf.int32, (None, None))
    eduration = tf.placeholder(tf.int32, (None, None))
    
    eglobal_step = tf.train.create_global_step()
    enoteseq = {"pitch": epitch, "velocity": evelocity, "dt": edt, "duration": eduration}
    eval_rnn = build_graph(enoteseq, config, True, True)
    eval_saver = tf.train.Saver()

    with tf.variable_scope("misc"):
        summary = [tf.summary.scalar("eval_loss", eval_rnn["loss"])]
        
    eval_sum_op = tf.summary.merge(summary)
    

train_sess = tf.Session(graph=train_graph, config=tf_cfg)
eval_sess = tf.Session(graph=eval_graph, config=tf_cfg)

train_writer = tf.summary.FileWriter(f"experiments/{config.save_path}/train", train_sess.graph)
eval_writer = tf.summary.FileWriter(f"experiments/{config.save_path}/eval", eval_sess.graph)
train_sess.run(init)

generator = batch_generator(data, config)
eval_generator = batch_generator(eval_data, config, transpose=False)

start = time.time()
train_sess.run(init)
eval_loss = 0
pbar = tqdm.tqdm(range(config.max_iterations))
for iterations in pbar:
    p, v, d, r = next(generator)
    feed = {pitch: p, velocity: v, dt: d, duration: r}
    _, loss, summ, step = train_sess.run([train_op, rnn["loss"], train_sum_op, global_step], feed_dict=feed)
    train_writer.add_summary(summ, step)
    
    if iterations%1000 == 0:
        train_saver.save(train_sess, f"experiments/{config.save_path}/{str(iterations).zfill(5)}")
        eval_saver.restore(eval_sess, f"experiments/{config.save_path}/{str(iterations).zfill(5)}")
        p, v, d, r = next(eval_generator)
        feed = {epitch: p, evelocity:v, edt: d, eduration: r}
        eval_loss, summ, step = eval_sess.run([eval_rnn["loss"], eval_sum_op, eglobal_step], feed_dict=feed)
        eval_writer.add_summary(summ, step)
    
    pbar.set_description(f"Train Loss: {loss:.4f} Eval Loss: {eval_loss:.4f}")

train_saver.save(train_sess, f"experiments/{config.save_path}/yamaha")

print("Total training time: {}".format(time.time()-start))
