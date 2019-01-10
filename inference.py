#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:55:02 2019

@author: yq
"""

import tensorflow as tf
from model import build_graph
from model import convert_midi
import config
#import pickle
import argparse
import midi2vec

tf.reset_default_graph()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prime_midi", required=True)
ap.add_argument("-o", "--output_file", required=True)
ap.add_argument("-c", "--checkpoint", required=True)
ap.add_argument("-t", "--temperature", default=1.0)
ap.add_argument("-l", "--length", default=100)
ap.add_argument("-s", "--scope_name", default="yamaha")
args = vars(ap.parse_args())

data = midi2vec.convert(args["prime_midi"])

if data == None:
    raise ValueError("Terminating process, corrupted MIDI..")

pitch = tf.placeholder(tf.int32, (None, None))
velocity = tf.placeholder(tf.int32, (None, None))
dt = tf.placeholder(tf.int32, (None, None))
duration = tf.placeholder(tf.int32, (None, None))

noteseq = {}
noteseq["pitch"] = pitch
noteseq["velocity"] = velocity
noteseq["dt"] = dt
noteseq["duration"] = duration

#with tf.variable_scope(args["scope_name"]):
rnn = build_graph(noteseq, config, False, False, temperature=float(args["temperature"]),
                  length=int(args["length"]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, args["checkpoint"])
    p, v, d, r = data["pitch"], data["velocity"], data["dt"], data["duration"]
    
    print("[INFO] Generating song...")
    song = sess.run(rnn, feed_dict={pitch: p, velocity: v, dt: d, duration: r})

song = song[0]
convert_midi(song[:, 0], song[:, 1], song[:, 2], song[:, 3], config, out=args["output_file"])



#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ckpt
#ckpt("ckpt/yamaha", "", "")

