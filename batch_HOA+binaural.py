# Example 1: 
# - Renders scenes to 5th order HOA
# - from HOA to Binaural with 4 different (static) head rotations (5 channel flac file)
# - storage of FOA as flac
# ------------------------------------

from ambiscaper import *
import numpy as np
import os
import soundfile as sf
import scipy.signal
import random
from random import choice
#from pedalboard import Pedalboard, Reverb, load_plugin

# AmbiScaper settings
soundscape_duration = 15.0 #35.0
num_scenes = 20 # number of audio scenes
num_events = 10 # number of audio events per scene

ambisonics_order = 5
numAmbiCoef = (ambisonics_order+1)*(ambisonics_order+1)
hrtf_folder = os.path.join(os. getcwd(),'./HRTF/')
hrtf_data, hrtf_sample_rate = sf.read(hrtf_folder + "/sh_hrir_order_" + str(ambisonics_order) + ".wav")
symVec = [1, 
         -1, 1, 1, 
         -1,-1, 1, 1, 1, 
         -1,-1,-1, 1, 1, 1, 1,
         -1,-1,-1,-1, 1, 1, 1, 1, 1,
         -1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1]   

# get ht_data from ht-captures folder
ht_path = os.path.join(os. getcwd(), '../resources/ht-captures/')
ht_data_name = []
for item in os.listdir(ht_path):
    if os.path.splitext(item)[1] == '.flac':
        ht_data_name.append(item)
frame_length = 256

# We want to use the full samples folder as potential events
samples_folder = './samples/Acoustics_Book'

outfolder = os.path.join(os. getcwd(), './time-varying yaw rotation-horizontalOnly/')
outfolderFOA = os.path.join(outfolder, 'FOA/')
outfolderBin = os.path.join(outfolder, 'Binaural/')

if not os.path.exists(outfolder): 
    os.mkdir(outfolder)
if not os.path.exists(outfolderFOA): 
    os.mkdir(outfolderFOA)
if not os.path.exists(outfolderBin): 
    os.mkdir(outfolderBin)

for scene_idx in range(num_scenes):
    folder = "scene"+str(scene_idx)
    destination_path = os.path.join(outfolder, folder)
    ### Create an ambiscaper instance
    ambiscaper = AmbiScaper(duration=soundscape_duration,
                            ambisonics_order=ambisonics_order,
                            fg_path=samples_folder,
                            bg_path=samples_folder)

    # Configure reference noise floor level
    ambiscaper.ref_db = -30

    ### Add a background event

    # Background events, by definition, have maximum spread
    # That means that they will only contain energy in the W channel (the first one)

    #ambiscaper.add_background(source_file=('const', 'tr-1788d-piece4-sl.wav'), source_time=('const', 0))

    numChans = 5 # 2 audio channels + 3 head pose channels 
    ### Add an event
    
    for event_idx in range(num_events):
        ambiscaper.add_event(source_file=('choose', []),
                             source_time=('uniform', 0, soundscape_duration),
                             event_time=('uniform', 0, soundscape_duration),
                             event_duration=('const', soundscape_duration),
                             event_azimuth=('uniform', 0, 2 * np.pi),
                             #event_elevation=('uniform', -np.pi / 2, np.pi / 2),
                             #event_azimuth=('const', 0),
                             event_elevation=('const', 0),
                             event_spread=('const', 0),
                             snr=('uniform', 0, 10),
                             pitch_shift=('const', 1),
                             time_stretch=('const', 1))



    ### Generate the audio and the annotation
    ambiscaper.generate(destination_path=destination_path,
                        generate_txt=True,
                        allow_repeated_source=True,
                        disable_sox_warnings=True,
                        disable_instantiation_warnings=True)

    
    ambi_data, ambi_sample_rate = sf.read(destination_path+"/"+folder+".wav")


    #if max(abs(ambi_data[:,0]))>1.0:
    #    print('ambi_data too hot')


    # Test if the nornalization works inside the file core.py. if not, it will print 'normalizing!'
    maxVal_ambi = np.max(abs(ambi_data[:, 0]))
    if maxVal_ambi > 0.99:
        # normalize entire ambi_data
        print('normalization needed!')


    sf.write(outfolderFOA+"/"+folder+"_horizontalOnly_AMBIX.flac", ambi_data[:,0:4], ambi_sample_rate)  #, subtype='FLOAT')


## Binauralization with Google Omnitone
    assert(ambi_sample_rate==hrtf_sample_rate)
    print(ambi_data.shape)
    output_file_duration_samples = ambi_data.shape[0]
    #print(output_file_duration_samples)
    output_signal = np.zeros((output_file_duration_samples+hrtf_data.shape[0]-1, numChans))
    for k in range(numAmbiCoef):        
        temp = scipy.signal.fftconvolve(ambi_data[:, k], hrtf_data[:, k])        
        output_signal[:,0] += temp
        output_signal[:,1] += temp*symVec[k]
    maxVal_output = np.amax(abs(output_signal[:, :2]))
    if maxVal_output > 0.99:
        print("output_signal_binaural_rot_0 too hot. maximum value is", maxVal_output)
        output_signal[:, :2] /= (maxVal_output * 1.02)
    sf.write(outfolderBin+"/"+folder+"_horizontalOnly_rot_0_binaural.flac", output_signal, ambi_sample_rate)#, subtype='FLOAT')

    selected_ht_filename = choice(ht_data_name)
    print('selected headtracker file is: ', selected_ht_filename)
    ht_data, ht_samplerate = sf.read(os.path.join(ht_path, selected_ht_filename))
    ht_duration = len(ht_data) / ht_samplerate  # get the time duration of ht_data

    # generate two binary flags in order to control the time reverse and phase flipping
    binary_flag = random.randint(0, 1)
    binary_flag2 = random.randint(0, 1)
    ht_data_selected_time = max(0, int((ht_duration-soundscape_duration) * ht_samplerate))
    start_frame = random.randint(0, ht_data_selected_time)
    stop_frame = min(len(ht_data), start_frame + int(ht_samplerate * soundscape_duration))  # stop_frame has a fixed length, i.e. time of start_frame + time of soundscape
    # step_size = int(np.where(start_frame - stop_frame < 0, 1, -1))
    ht_data_trunc = ht_data[start_frame: stop_frame, :]
    if binary_flag == 1:  # time reverse
        print('time reverse ht data')
        ht_data_trunc = np.flipud(ht_data_trunc)
    #else:
    #    print('time revers does not occur')
    if binary_flag2 == 1:  # phase flipping
        print('inverting direction of ht data')
        ht_data_trunc = -ht_data_trunc
    #else:
    #    print('phase flipping does not occur')

    mtx = np.identity(numAmbiCoef)
    # solve the dimensions' difference between ambi_data and ht_data
    ambi_data_rot = np.zeros(ambi_data.shape)
    #output_signal = np.zeros((ambi_data_rot.shape[0]+hrtf_data.shape[0]-1,numChans))
    output_signal = np.zeros((output_file_duration_samples + hrtf_data.shape[0] - 1, numChans))
    min_length = min(len(ambi_data), len(ht_data_trunc))
    for i in np.arange(0, min_length-1, frame_length).tolist():
        yaw = (ht_data_trunc[i, 0] * np.pi)
        # print(yaw_deg)
        # yaw = yaw_deg*(np.pi/180.0)
        if ambisonics_order>0:
            cosYaw = np.cos(yaw)
            sinYaw = np.sin(yaw)
            mtx[1,1] = cosYaw
            mtx[3,3] = cosYaw
            mtx[1,3] = sinYaw
            mtx[3,1] = -sinYaw
        if ambisonics_order>1:
            cos2Yaw = np.cos(2*yaw)
            sin2Yaw = np.sin(2*yaw)
            mtx[4,4] = cos2Yaw
            mtx[8,8] = cos2Yaw
            mtx[4,8] = sin2Yaw
            mtx[8,4] = -sin2Yaw
            mtx[5:8,5:8] = mtx[1:4,1:4]
        if ambisonics_order>2:
            cos3Yaw = np.cos(3*yaw)
            sin3Yaw = np.sin(3*yaw)
            mtx[9,9] = cos3Yaw
            mtx[15,15] = cos3Yaw
            mtx[ 9,15] = sin3Yaw
            mtx[15, 9] = -sin3Yaw
            mtx[10:15,10:15] = mtx[4:9,4:9]
        if ambisonics_order>3:
            cos4Yaw = np.cos(4*yaw)
            sin4Yaw = np.sin(4*yaw)
            mtx[16,16] = cos4Yaw
            mtx[24,24] = cos4Yaw
            mtx[16,24] = sin4Yaw
            mtx[24,16] = -sin4Yaw
            mtx[17:24,17:24] = mtx[9:16,9:16]
        if ambisonics_order>4:
            mtx[25,25] =  np.cos(5*yaw)
            mtx[35,35] =  mtx[25,25]
            mtx[25,35] =  np.sin(5*yaw)
            mtx[35,25] =  -mtx[25,35]
            mtx[26:35,26:35] = mtx[16:25,16:25]

        ambi_data_rot[i:i+frame_length,:] = np.matmul(ambi_data[i:i+frame_length,:], mtx)
        output_signal[i:i+frame_length,2] = yaw/np.pi
    ambi_data_rot[i + frame_length+1:len(ambi_data)-1, :] = np.matmul(ambi_data[i + frame_length+1:len(ambi_data)-1, :], mtx)
    output_signal[i + frame_length+1:len(output_signal)-1, 2] = yaw / np.pi

    maxVal_W_rot = max(abs(ambi_data_rot[:, 0]))
    if maxVal_W_rot > 0.99:
        print('ambi_data_rot too hot, max Val is', maxVal_W_rot)

    ## Binauralization with Google Omnitone
    assert(ambi_sample_rate==hrtf_sample_rate)
    assert(output_file_duration_samples==ambi_data_rot.shape[0])

    for k in range(numAmbiCoef):
        temp = scipy.signal.fftconvolve(ambi_data_rot[:, k], hrtf_data[:, k])
        output_signal[:, 0] += temp
        output_signal[:, 1] += temp*symVec[k]
    maxVal_output = np.amax(abs(output_signal[:, :2]))
    if maxVal_output > 0.99:
        print("output_signal_binaural too hot. maximum value is", maxVal_output)
        output_signal[:, :2] /= (maxVal_output * 1.02)


    # sf.write(outfolderBin+"/"+folder+"_horizontalOnly_rot_"+str(yaw_deg)+"_binaural_TEST.flac", output_signal, ambi_sample_rate)#, subtype='FLOAT')
    # output_signal[:, 0] /= (maxVal*1.02)
    # output_signal[:, 1] /= (maxVal*1.02)
    sf.write(outfolderBin+"/"+folder+"_horizontalOnly_rot_binaural.flac", output_signal, ambi_sample_rate)#, subtype='FLOAT')
    if np.max(abs(output_signal[:, 0])) > 1.0:
        print("left channel clips")
    if np.max(abs(output_signal[:, 1])) > 1.0:
        print("right channel clips")

    #elif (np.max(abs(output_signal[:,0])) and np.max(abs(output_signal[:,1]))) > 1.0:
    #    print("both exist hopping")
    #else:
    #    print("no hopping")

    if 0:
        yaw = -yaw_deg*(np.pi/180.0)

        if ambisonics_order>0:
            cosYaw = np.cos(yaw)
            sinYaw = np.sin(yaw)
            mtx[1,1] = cosYaw
            mtx[3,3] = cosYaw
            mtx[1,3] = sinYaw
            mtx[3,1] = -sinYaw
        if ambisonics_order>1:
            cos2Yaw = np.cos(2*yaw)
            sin2Yaw = np.sin(2*yaw)
            mtx[4,4] = cos2Yaw
            mtx[8,8] = cos2Yaw
            mtx[4,8] = sin2Yaw
            mtx[8,4] = -sin2Yaw
            mtx[5:8,5:8] = mtx[1:4,1:4]
        if ambisonics_order>2:
            cos3Yaw = np.cos(3*yaw)
            sin3Yaw = np.sin(3*yaw)
            mtx[9,9] = cos3Yaw
            mtx[15,15] = cos3Yaw
            mtx[ 9,15] = sin3Yaw
            mtx[15, 9] = -sin3Yaw
            mtx[10:15,10:15] = mtx[4:9,4:9]
        if ambisonics_order>3:
            cos4Yaw = np.cos(4*yaw)
            sin4Yaw = np.sin(4*yaw)
            mtx[16,16] = cos4Yaw
            mtx[24,24] = cos4Yaw
            mtx[16,24] = sin4Yaw
            mtx[24,16] = -sin4Yaw
            mtx[17:24,17:24] = mtx[9:16,9:16]
        if ambisonics_order>4:
            mtx[25,25] =  np.cos(5*yaw)
            mtx[35,35] =  mtx[25,25]
            mtx[25,35] =  np.sin(5*yaw)
            mtx[35,25] =  -mtx[25,35]
            mtx[26:35,26:35] = mtx[16:25,16:25]

        hrtf_data_rot = np.matmul(hrtf_data,mtx)
#        hrtf_data_rot2 = np.matmul(np.matmul(hrtf_data,np.diag(symVec[0:numAmbiCoef])),mtx)
        hrtf_data_rot2 = np.matmul(hrtf_data,np.matmul(np.diag(symVec[0:numAmbiCoef]),mtx))
        #hrtf_data_rot2 = np.matmul(np.diag(symVec[0:numAmbiCoef]),np.matmul(hrtf_data,mtx).T).T # doesnt work
        print(hrtf_data.shape)
        print(hrtf_data_rot.shape)
        #test when rotating the hrtfs
        output_signal2 = np.zeros((output_file_duration_samples+hrtf_data.shape[0]-1,numChans))
        for k in range(numAmbiCoef):
            temp = scipy.signal.fftconvolve(ambi_data[:, k], hrtf_data_rot[:, k])
            temp2 = scipy.signal.fftconvolve(ambi_data[:, k], hrtf_data_rot2[:, k])
            output_signal2[:,0] += temp
            output_signal2[:,1] += temp2 #*symVec[k]
        output_signal[:,2] = yaw_deg/180
        sf.write(destination_path+"/../"+folder+"_horizontalOnly_rot_"+str(yaw_deg)+"_bin_test.flac", output_signal2, ambi_sample_rate)#, subtype='FLOAT')
    #remove HOA file to save some HD space:
os.remove(destination_path+"/"+folder+".wav")

