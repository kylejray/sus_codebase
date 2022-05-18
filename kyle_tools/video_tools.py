import ffmpeg
import os
import numpy as np



def video_tile(vid_list, row_length, save_name='out.mp4'):
    '''
    takes in a list of same sized videos and tiles them into a single video at 'save_name', important: will overwrite anything that has the file name of the output file depends on python ffmpeg
    
    Parameters
    ----------
    vid_list: list of video files
        the tiled videos are made from this list, will probably act weird if their sizes are not the same
    row_length: int
        number of videos in each row
    save_name: str
        where the video will be saved and what format
    
    Returns
    -------
    None: there are no returns, it just saves the video to 'save_name'

    '''
    N = len(vid_list)
    column_length = int(np.ceil(N/row_length))

    videoprobe = ffmpeg.probe(vid_list[0]) #get info on video
    vwidth, vheight = videoprobe["streams"][0]["width"], videoprobe["streams"][0]["height"] # get width of main video
    #scale = (1080 / videoheight) #set some scales
    videowidth = vwidth*row_length
    videoheight = vheight*column_length
        
    video_list = [ffmpeg.input(item) for item in vid_list]
    #videostr = [ffmpeg.filter(item, "scale", "%dx%d" %(videowidth, videoheight)) for item in videostr]
    videostr = ffmpeg.filter(video_list[0], "pad", videowidth, videoheight)
    
    for i,item in enumerate(video_list):
        if i>0:
            j = i//row_length
            k = i - row_length*j
            videostr = ffmpeg.overlay(videostr, item, x=k*vwidth, y=j*vheight) #overlay image
    

    try: os.remove(save_name)
    except: pass
    
    videostr = ffmpeg.output(videostr, save_name)
    ffmpeg.run(videostr)

def video_concat(vids, save_name='concat.mp4'):
    
    vids = [ffmpeg.input(item) for item in vids]
    try: os.remove(save_name)
    except: pass
    ffmpeg.concat(*vids, v=1).output(save_name).run()

    # SAVED FOR LATER
    # ffmpeg -i out.mp4 -vf mpdecimate,setpts=N/FRAME_RATE/TB out_out2.mp4
    # this command removes still frames, super useful. 