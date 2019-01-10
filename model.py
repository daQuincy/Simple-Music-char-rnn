#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:57:01 2019

@author: yq
"""

import tensorflow as tf
from collections import namedtuple
import numpy as np
import random
import pretty_midi
import matplotlib.pyplot as plt

features = namedtuple("features", ["dict", "feat", "infer"])
rnn_stuff = namedtuple("rnn_stuff", ["init", "out", "state", "logit"])
loss_stuff = namedtuple("loss_stuff", ["pitch", "velocity", "dt", "duration"])

dwrap = tf.contrib.rnn.DropoutWrapper
he_init = tf.contrib.layers.variance_scaling_initializer()

def build_inputs(noteseqs, config):
    pitchs = noteseqs["pitch"]
    dts = noteseqs["dt"]
    velocitys = noteseqs["velocity"]
    durations = noteseqs["duration"]

    # one hot encode the features
    pitch = tf.one_hot(pitchs, depth=config.pitch_depth)
    dt = tf.one_hot(dts, depth=config.max_discrete_times)
    velocity = tf.one_hot(velocitys, depth=config.max_discrete_velocities)
    duration = tf.one_hot(durations, depth=config.max_discrete_durations)

    feat = tf.concat([pitch, velocity, dt, duration], axis=2)

    feat_dict = {"pitch": pitch, "dt": dt, "velocity": velocity,
                 "duration": duration}

    feat_index = tf.stack([pitchs, velocitys, dts, durations], axis=-1)

    f = features(feat_dict, feat, feat_index)

    return f


def build_training_rnn(features, dropout, config):
    x_shape = config.pitch_depth + config.max_discrete_times + config.max_discrete_velocities + config.max_discrete_durations

    #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    if config.rnn == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell
    elif config.rnn == "gru":
        cell = tf.nn.rnn_cell.GRUCell
    elif config.rnn == "rnn":
        cell = tf.nn.rnn_cell.RNNCell
    else:
        raise ValueError("Cell not specified")
        
    cell = tf.nn.rnn_cell.MultiRNNCell(
            [dwrap(cell(num_units=config.rnn_dim, name="LAYER_{}".format(i)), output_keep_prob=dropout) for i in range(config.rnn_layers)])

    layers = tf.keras.layers.Dense(x_shape, kernel_initializer=he_init)

    initial_state = cell.zero_state(config.batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, features, initial_state=initial_state)

    logits = layers(outputs)

    r = rnn_stuff(initial_state, outputs, states, logits)

    return r

def get_logits(logits, config):
    a = config.pitch_depth
    b = config.max_discrete_velocities
    c = config.max_discrete_times
    d = config.max_discrete_durations

    pitch_logit = logits[:, :, :a]
    velocity_logit = logits[:, :, a:a+b]
    dt_logit = logits[:, :, a+b:a+b+c]
    duration_logit = logits[:, :, -d:]

    return pitch_logit, velocity_logit, dt_logit, duration_logit


def build_loss(logits, labels, config):
    pitch_logit, velocity_logit, dt_logit, duration_logit = get_logits(logits, config)

    xp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels["pitch"][:, 1:, :], logits=pitch_logit), axis=[1, 0])
    xv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels["velocity"][:, 1:, :], logits=velocity_logit), axis=[1, 0])
    xd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels["dt"][:, 1:, :], logits=dt_logit), axis=[1, 0])
    xr = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels["duration"][:, 1:, :], logits=duration_logit), axis=[1, 0])

    l = loss_stuff(xp, xv, xd, xr)

    return xp+xv+xd+xr, l

def build_inference_rnn(primer, prime_index, config, temperature, length):
    x_shape = config.pitch_depth + config.max_discrete_times + config.max_discrete_velocities + config.max_discrete_durations


    #cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    cell = tf.nn.rnn_cell.LSTMCell
    cell = tf.nn.rnn_cell.MultiRNNCell(
            [dwrap(cell(num_units=config.rnn_dim, name="LAYER_{}".format(i))) for i in range(config.rnn_layers)])

    layers = tf.keras.layers.Dense(x_shape)

    initial_state = cell.zero_state(tf.shape(primer)[0], dtype=tf.float32)

    mid_out, mid_states = tf.nn.dynamic_rnn(cell, primer, initial_state=initial_state)
    mid_out = mid_out[:, -1, :]
    mid_logit = layers(mid_out)


    def multi_sample(logits):
        a = config.pitch_depth
        b = config.max_discrete_velocities
        c = config.max_discrete_times
        d = config.max_discrete_durations

        pitch_logit = logits[:, :a] / temperature
        velocity_logit = logits[:, a:a+b] / temperature
        dt_logit = logits[:, a+b:a+b+c] / temperature
        duration_logit = logits[:, -d:] / temperature

        pitch = tf.multinomial(pitch_logit, 1, output_dtype=tf.int32)[:, 0]
        dt = tf.multinomial(dt_logit, 1, output_dtype=tf.int32)[:, 0]
        velocity = tf.multinomial(velocity_logit, 1, output_dtype=tf.int32)[:, 0]
        duration = tf.multinomial(duration_logit, 1, output_dtype=tf.int32)[:, 0]

        return pitch, dt, velocity, duration


    def get_next(logits):
        pitch, dt, velocity, duration = multi_sample(logits)

        xpitch = tf.one_hot(pitch, depth=config.pitch_depth)
        xdt = tf.one_hot(dt, depth=config.max_discrete_times)
        xvelocity = tf.one_hot(velocity, depth=config.max_discrete_velocities)
        xduration = tf.one_hot(duration, depth=config.max_discrete_durations)

        feat = tf.concat([xpitch, xvelocity, xdt, xduration], axis=-1)

        return feat, [pitch, velocity, dt, duration]

    #with tf.variable_scope("rnn", reuse=True):
    song = []
    next_input, prediction = get_next(mid_logit)
    states = mid_states
    for i in range(length):
        outputs, states = cell(inputs=next_input, state=states)
        logits = layers(outputs)
        next_input, prediction = get_next(logits)
        song.append(tf.stack(prediction, axis=1))

    song = tf.stack(song, axis=1)
    song = tf.concat([prime_index, song], axis=1)

    return song



def build_graph(noteseqs, config, is_train, is_eval, temperature=None, length=None):
    features = build_inputs(noteseqs, config)

    if is_train:
        if is_eval:
            dropout = 1.0
        else:
            dropout = config.dropout
        rnn_out = build_training_rnn(features.feat[:, :-1, :], dropout, config)
        total_loss, losses = build_loss(rnn_out.logit, features.dict, config)

        graph = {}
        graph["rnn"] = rnn_out
        graph["loss"] = total_loss
        graph["losses"] = losses

        return graph

    else:
        rnn_out = build_inference_rnn(features.feat, features.infer, config, temperature, length)

        return rnn_out

def batch_generator(dataset, config, transpose=True, ss=-1):
    pitch = dataset["pitch"]
    velocity = dataset["velocity"]
    dt = dataset["dt"]
    duration = dataset["duration"]

    if ss > 0:
        seq_len = ss
    else:
        seq_len = np.random.choice(np.arange(config.min_len, config.max_len))

    pp = []
    vv = []
    dd = []
    rr = []

    while True:
        z = list(zip(pitch, velocity, dt, duration))
        random.shuffle(z)

        pitch, velocity, dt, duration = zip(*z)

        for i in range(len(pitch)):
            s = np.random.choice(len(pitch[i])-seq_len)

            pp.append(pitch[i][s:s+seq_len])
            vv.append(velocity[i][s:s+seq_len])
            dd.append(dt[i][s:s+seq_len])
            rr.append(duration[i][s:s+seq_len])

            if len(pp) == config.batch_size:
                if transpose:
                    t = np.random.choice(np.arange(-5, 5), (config.batch_size, 1))
                    t = 21 + t
                else:
                    t = 21
                yield(np.stack(pp) - t,
                      ((np.stack(vv)*config.max_discrete_velocities) / 128).astype("int"),
                      (np.minimum(np.stack(dd)*31.25, config.max_discrete_times-1)).astype("int"),
                      (np.minimum(np.stack(rr)*31.25, config.max_discrete_durations-1)).astype("int")
                      )

                pp = []
                vv = []
                dd = []
                rr = []
                if ss > 0:
                    seq_len = ss
                else:
                    seq_len = np.random.choice(np.arange(config.min_len, config.max_len))
                    
def test_generator(dataset, config, transpose=True, ss=-1, min_len=32, max_len=128):
    pitch = dataset["pitch"]
    velocity = dataset["velocity"]
    dt = dataset["dt"]
    duration = dataset["duration"]

    if ss > 0:
        seq_len = ss
    else:
        seq_len = np.random.choice(np.arange(min_len, max_len))

    pp = []
    vv = []
    dd = []
    rr = []

    for i in range(len(pitch)):
        s = np.random.choice(len(pitch[i])-seq_len)

        pp.append(pitch[i][s:s+seq_len])
        vv.append(velocity[i][s:s+seq_len])
        dd.append(dt[i][s:s+seq_len])
        rr.append(duration[i][s:s+seq_len])

        if len(pp) == config.batch_size:
            if transpose:
                t = np.random.choice(np.arange(-5, 5), (config.batch_size, 1))
                t = 21 + t
            else:
                t = 21
            yield(np.stack(pp) - t,
                  ((np.stack(vv)*config.max_discrete_velocities) / 128).astype("int"),
                  (np.minimum(np.stack(dd)*31.25, config.max_discrete_times-1)).astype("int"),
                  (np.minimum(np.stack(rr)*31.25, config.max_discrete_durations-1)).astype("int")
                  )

            pp = []
            vv = []
            dd = []
            rr = []
            if ss > 0:
                seq_len = ss
            else:
                seq_len = np.random.choice(np.arange(min_len, max_len))                    
                
    if len(pp) > 0:
        if transpose:
            t = np.random.choice(np.arange(-5, 5), (len(pp), 1))
            t = 21 + t
        else:
            t = 21
            
        pp = np.stack(pp) - t
        assert np.max(pp) < 88
        assert np.min(pp) >= 0
        yield(pp,
              ((np.stack(vv)*config.max_discrete_velocities) / 128).astype("int"),
              (np.minimum(np.stack(dd)*31.25, config.max_discrete_times-1)).astype("int"),
              (np.minimum(np.stack(rr)*31.25, config.max_discrete_durations-1)).astype("int")
              )

def convert_midi(pitches, velocities, dts, durations, config, out="test.mid"):
    mid = pretty_midi.PrettyMIDI(resolution=config.resolution )
    piano = pretty_midi.Instrument(program=config.program)

    start = 0
    for i in range(len(pitches)):
        if i == 0:
            start = 0
        else:
            start += (dts[i] /31.25)

        pitch = pitches[i] + 21
        velocity = min((velocities[i]+1) * (128//config.max_discrete_velocities), 127)
        duration = durations[i] / 31.25
        end = start + duration

        note = pretty_midi.Note(pitch=pitch, velocity=velocity, start=start, end=end)
        piano.notes.append(note)

    mid.instruments.append(piano)

    print("[INFO] Writing generated music to {}".format(out))
    mid.write(out)
    
    #piano_roll = mid.get_piano_roll()
    
    #plt.figure(figsize=(20, 8))
    #plt.title("Piano Roll")
    #plt.imshow(np.flipud(piano_roll)/128)
    #plt.show()


if __name__ == "__main__":
    import config
    import pickle
    tf.reset_default_graph()

#    pitch = tf.placeholder(tf.int32, (None, None))
#    velocity = tf.placeholder(tf.int32, (None, None))
#    dt = tf.placeholder(tf.int32, (None, None))
#    duration = tf.placeholder(tf.int32, (None, None))
#
#    noteseq = {}
#    noteseq["pitch"] = pitch
#    noteseq["velocity"] = velocity
#    noteseq["dt"] = dt
#    noteseq["duration"] = duration
#
#    test = build_graph(noteseq, config, True)

    data = pickle.load(open("yamaha_test.p", "rb"))
    gen = batch_generator(data, config)
    
    for i in range(1000):
        a = next(gen)
