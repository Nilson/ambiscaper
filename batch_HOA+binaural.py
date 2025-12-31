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
import time
import spaudiopy as spa
import generateBackgroundExamples
random.seed(42)
np.random.seed(42)


##############################################
# AmbiScaper settings
soundscape_duration = 15.0 #35.0
num_scenes = 5 #10 #20 # number of audio scenes
num_events = 10 # number of audio events per scene

ambisonics_order = 5
yawRotationOnly = True  # if True, only yaw rotation is applied, pitch and roll are fixed to zero
keepHoaScenes = False  # if True, the resulting HOA scenes are kept on disk, otherwise only the FOA version is kept as .flac audio file.
samples_folder = './samples/Acoustics_Book'
bg_folder = './samples/Backgrounds'
outfolder = os.path.join(os.getcwd(), './Testing/')

##############################################

if bg_folder == './samples/Backgrounds' and not os.path.exists(bg_folder):    
    os.makedirs(bg_folder)
    generateBackgroundExamples.main(order=ambisonics_order, duration=soundscape_duration, fs=48000,
                                    filename=os.path.join(bg_folder, 'background1.wav'),
                                    noise_type='pink')
    generateBackgroundExamples.main(order=ambisonics_order, duration=soundscape_duration, fs=48000,
                                    filename=os.path.join(bg_folder, 'background2.wav'),
                                    noise_type='white')
    generateBackgroundExamples.main(order=ambisonics_order, duration=soundscape_duration, fs=48000,
                                    filename=os.path.join(bg_folder, 'background3.wav'),
                                    noise_type='brown')
    
    


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
# for head-tracking based soundfield rotation 
frame_length = 512
hopsize = int(frame_length/2)



outfolderFOA = os.path.join(outfolder, 'FOA/')
outfolderBin = os.path.join(outfolder, 'Binaural/')

if not os.path.exists(outfolder): 
    os.mkdir(outfolder)
if not os.path.exists(outfolderFOA): 
    os.mkdir(outfolderFOA)
if not os.path.exists(outfolderBin): 
    os.mkdir(outfolderBin)

start_overall_time = time.time()
for scene_idx in range(num_scenes):
    folder = "scene"+str(scene_idx)
    destination_path = os.path.join(outfolder, folder)
    ### Create an ambiscaper instance
    start_time = time.time()
    ambiscaper = AmbiScaper(duration=soundscape_duration,
                            ambisonics_order=ambisonics_order,
                            fg_path=samples_folder,
                            bg_path=bg_folder)

    # Configure reference level ref_db, 
    # i.e. the loudnes of the background, measured in LUFS. 
    # Later when we add foreground events, we’ll have to specify an snr (signal-to-noise ratio) value, 
    # i.e. by how many decibels (dB) should the foreground event be louder (or softer) with respect 
    # to the background level specified by ref_db.

    ambiscaper.ref_db = -30    
    ambiscaper.sr = hrtf_sample_rate
    ### Add a background event

    # Background events, by definition, have maximum spread
    # That means that they will only contain energy in the W channel (the first one)

    #ambiscaper.add_background(source_file=('const', 'test.wav'), source_time=('const', 0))
    
    ambiscaper.add_background(source_file=('choose', []), 
                              source_time=('const', 0),
                              event_azimuth=('uniform', 0, 2 * np.pi))
    

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
                             pitch_shift=('const', 0),
                             time_stretch=('const', 1))



    ### Generate the audio and the annotation
    ambiscaper.generate(destination_path=destination_path,
                        fix_clipping=True,
                        peak_normalization=False,
                        generate_txt=True,
                        allow_repeated_source=True,
                        save_isolated_events=False,
                        disable_sox_warnings=True,
                        disable_instantiation_warnings=True)

    print("-1- %s seconds - Ambiscaper ---" % (time.time() - start_time))
    
    ambi_data, ambi_sample_rate = sf.read(destination_path+"/"+folder+".wav")

    sf.write(outfolderFOA+"/"+folder+"_AMBIX.flac", ambi_data[:,0:4], ambi_sample_rate, subtype="PCM_24")  #, subtype='FLOAT')

    start_time = time.time()
    ## Binauralization with Google Omnitone
    assert(ambi_sample_rate==hrtf_sample_rate)
    #print(ambi_data.shape)
    output_file_duration_samples = ambi_data.shape[0]
    #print(output_file_duration_samples)
    output_signal = np.zeros((output_file_duration_samples+hrtf_data.shape[0]-1, numChans))
    for k in range(numAmbiCoef):        
        temp = scipy.signal.fftconvolve(ambi_data[:, k], hrtf_data[:, k])        
        output_signal[:,0] += temp
        output_signal[:,1] += temp*symVec[k]
    maxVal_output = np.amax(abs(output_signal[:, :2]))
    if maxVal_output > 0.99: 
        print("output_signal_binaural_rot_0 too hot. maximum value is", 20.0*np.log10(maxVal_output), "dBFS")
        output_signal[:, :2] /= (maxVal_output * 1.02)
    sf.write(outfolderBin+"/"+folder+"_horizontalOnly_rot_0_binaural.flac", output_signal, ambi_sample_rate)#, subtype="PCM_24"subtype='FLOAT')
    
    selected_ht_filename = choice(ht_data_name)
    # print('selected headtracker file is: ', selected_ht_filename)
    ht_filename = os.path.join(ht_path, selected_ht_filename)
    ht_info = sf.info(ht_filename)
    ht_data_selected_time = max(0, int((ht_info.duration-soundscape_duration) * ht_info.samplerate))
    start_frame = int(random.randint(0, ht_data_selected_time))
    stop_frame = int(start_frame + ht_info.samplerate * soundscape_duration)  # stop_frame has a fixed length, i.e. time of start_frame + time of soundscape    
    ht_data_trunc, ht_samplerate = sf.read(ht_filename, start=start_frame, stop=stop_frame)

    # generate two binary flags in order to control the time reverse and phase flipping
    binary_flag = random.randint(0, 1)
    binary_flag2 = random.randint(0, 1)
    if binary_flag == 1:  # time reverse
        #print('time reverse ht data')
        ht_data_trunc = np.flipud(ht_data_trunc)    
    if binary_flag2 == 1:  # phase flipping
        #print('inverting direction of ht data')
        ht_data_trunc = -ht_data_trunc
    #else:
    #    print('phase flipping does not occur')

    mtx = np.identity(numAmbiCoef)
    # solve the dimensions' difference between ambi_data and ht_data
    ambi_data_rot = np.zeros(ambi_data.shape)
    #output_signal = np.zeros((ambi_data_rot.shape[0]+hrtf_data.shape[0]-1,numChans))
    output_signal = np.zeros((output_file_duration_samples + hrtf_data.shape[0] - 1, numChans))
    min_length = min(len(ambi_data), len(ht_data_trunc))
    
    yaw, pitch, roll = 0, 0, 0    
    w = np.hanning(frame_length)
    w = w[:, np.newaxis] 
    if yawRotationOnly == False:
        for i in np.arange(0, min_length-frame_length, hopsize).tolist():        
            yaw = ht_data_trunc[i, 0] * np.pi
            pitch = ht_data_trunc[i, 1] * np.pi
            roll = ht_data_trunc[i, 2] * np.pi 
            output_signal[i:i+frame_length,2] = ht_data_trunc[i, 0]                               
            output_signal[i:i+frame_length,3] = ht_data_trunc[i, 1]            
            output_signal[i:i+frame_length,4] = ht_data_trunc[i, 2]
            mtx = spa.sph.sh_rotation_matrix(ambisonics_order, yaw, pitch, roll, sh_type='real').T  
            ambi_data_rot[i:i+frame_length,:] += (ambi_data[i:i+frame_length,:]*w) @ mtx       
            # TODO: fade with overlap-add to avoid occasional zipper noise artifacts at frame boundaries      
    else: #yaw rotation only, much faster to compute mtx   
        for i in np.arange(0, min_length-frame_length, hopsize).tolist():   
            yaw = ht_data_trunc[i, 0] * np.pi  
            output_signal[i:i+frame_length,2] = ht_data_trunc[i, 0]                                        
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
            ambi_data_rot[i:i+frame_length,:] += (ambi_data[i:i+frame_length,:]*w) @ mtx                            
        
    # process the remaining samples       
    tail = ambi_data[i + hopsize : len(ambi_data), :]
    tail[0:hopsize,:] *= w[0:hopsize,:]
    ambi_data_rot[i + hopsize : len(ambi_data), :] += tail @ mtx
    output_signal[i + hopsize : len(output_signal), 2] = yaw / np.pi
    output_signal[i + hopsize : len(output_signal), 3] = pitch / np.pi
    output_signal[i + hopsize : len(output_signal), 4] = roll / np.pi

    maxVal_W_rot = max(abs(ambi_data_rot[:, 0]))
    if maxVal_W_rot > 0.99:
        print('ambi_data_rot too hot, max Val is ', maxVal_W_rot)

    ## Binauralization with Google Omnitone
    assert(ambi_sample_rate==hrtf_sample_rate)
    assert(output_file_duration_samples==ambi_data_rot.shape[0])

    for k in range(numAmbiCoef):
        temp = scipy.signal.fftconvolve(ambi_data_rot[:, k], hrtf_data[:, k])
        output_signal[:, 0] += temp
        output_signal[:, 1] += temp*symVec[k]
    maxVal_output = np.amax(abs(output_signal[:, :2]))
    if maxVal_output > 0.99:
        print("output_signal_binaural too hot. maximum value is ", maxVal_output)
        output_signal[:, :2] /= (maxVal_output * 1.02)

    sf.write(outfolderBin+"/"+folder+"_horizontalOnly_rot_binaural.flac", output_signal, ambi_sample_rate)#, subtype='FLOAT')
    if np.max(abs(output_signal[:, 0])) > 1.0:
        print("left channel clips")
    if np.max(abs(output_signal[:, 1])) > 1.0:
        print("right channel clips")
    print("-2- %s seconds - Binauralization and Rotation ---" % (time.time() - start_time))

    if keepHoaScenes == False:
        os.remove(destination_path+"/"+folder+".wav")
print("## %s seconds - Overall ---" % (time.time() - start_overall_time))
