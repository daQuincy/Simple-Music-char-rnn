# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 21:02:16 2019

@author: YQ
"""

import pretty_midi
import numpy as np
import config
import pickle
import argparse

def convert(file):
    note = []
    velocity = []
    dt = []
    duration = []
    midi_note = []
    
    try:
        print("[INFO] Reading primer MIDI file...")
        mid = pretty_midi.PrettyMIDI(file)
    except:
        print("[Error] Corrupted file")
        return None
    
    for instrument in mid.instruments:
        if not instrument.is_drum:
            for n in instrument.notes:
                midi_note.append((n.pitch, n.start, n.end, n.velocity))            
                
    midi_note = sorted(midi_note, key=lambda x: x[1])
        
    prev_start = 0
    for m in midi_note:
        t = m[1] - prev_start
        
        note.append(m[0]-21)
        velocity.append((m[3] * config.max_discrete_velocities // 128))
        dt.append(np.minimum(t*31.25, config.max_discrete_times))
        duration.append(np.minimum((m[2]-m[1])*31.25, config.max_discrete_durations))
        
        prev_start = m[1]
    
    note = np.expand_dims(np.array(note, dtype="int"), axis=0)   
    velocity = np.expand_dims(np.array(velocity, dtype="int"), axis=0)
    dt = np.expand_dims(np.array(dt, dtype="int"), axis=0)
    duration = np.expand_dims(np.array(duration, dtype="int"), axis=0)
        
    data = {"pitch": note, "velocity": velocity, "dt": dt, "duration": duration}
    
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--midi", required=True)
    ap.add_argument("-p", "--pickle", required=True)
    args = vars(ap.parse_args())
    
    data = convert(args["midi"])
    pickle.dump(data, open(args["pickle"], "wb"))
