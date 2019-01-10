#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:05:08 2019

@author: yq
"""

import pretty_midi
import glob
import numpy as np
import pickle

def get_data(files):
    dataset = {}

    notes = []
    velocities = []
    dts = []
    durations = []

    min_len = 9999999
    max_len = 0

    for file in files:
        print("[INFO] Processing {}".format(file))

        note = []
        velocity = []
        dt = []
        duration = []
        midi_note = []

        try:
            mid = pretty_midi.PrettyMIDI(file)
        except:
            print("[Error] Corrupted file")
            continue
        
        for instrument in mid.instruments:
            if not instrument.is_drum:
                
                for n in instrument.notes:
                    midi_note.append((n.pitch, n.start, n.end, n.velocity))

        midi_note = sorted(midi_note, key=lambda x: x[1])

        prev_start = 0
        for m in midi_note:
            t = m[1] - prev_start

            note.append(m[0])
            velocity.append(m[-1])
            dt.append(np.float32(t))
            duration.append(np.float32(m[2] - m[1]))

            prev_start = m[1]

        if len(note) < min_len:
            min_len = len(note)
        if len(note) > max_len:
            max_len = len(note)

        notes.append(note)
        velocities.append(velocity)
        dts.append(dt)
        durations.append(duration)


    dataset["pitch"] = notes
    dataset["velocity"] = velocities
    dataset["dt"] = dts
    dataset["duration"] = durations

    print(" ")
    print("[INFO] Max len: {}  Min len: {}".format(max_len, min_len))
    print(" ")
    return dataset






train_file = []
for i in [2002, 2004, 2009, 2011, 2013, 2014, 2015, 2018]:
    t = glob.glob("yamaha/{}/*".format(i))
    train_file += sorted(t)

test_file = glob.glob("yamaha/2017/*")

training_set = get_data(train_file)
pickle.dump(training_set, open("yamaha_train.p", "wb"))

testing_set = get_data(test_file)
pickle.dump(testing_set, open("yamaha_test.p", "wb"))


