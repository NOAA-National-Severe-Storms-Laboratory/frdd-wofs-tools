import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import yaml
from optparse import OptionParser

# plt.rcParams['animation.html'] = 'jshtml'
# plt.rcParams['animation.embed_limit'] = 4000  # for 4 panel 73 frame, we need more than 3 GB
# plt.rcParams['animation.ffmpeg_path'] = "/Users/Louis.Wicker/miniconda3/bin/ffmpeg"

_reduce_image = False

#-----------------------------------------------------------------------------------------
# handy timer function

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(self.datetime.now() - self.tic))
        
#-----------------------------------------------------------------------------------------
# function to read images, and if needed, rescale

"""
  Read_png reads an image from a run directory - can be adapted for other tasks
"""

def read_wofs_png(rundir, fhour, plottype, time, www_dir = None):
 
# create image filename

    plotname = ("%s_f%3.3i.png" % (plottype, time))
    
    image_path = os.path.join(rundir, fhour, plotname)
    
    if www_dir != None:
        image_path = os.path.join(www_dir, image_path)
    
# Read Image 

    try:
        img = mpimg.imread(image_path)
    except:
        print("\n ==> STITCH ERROR:  cannot read image file: %s  EXITING!!! \n" % (image_path))
    
    if _reduce_image:
        
        (width, height) = (img.width // 2, img.height // 2)
        return img.resize((width, height),PIL.Image.LANCZOS)

    else:
        return img


#-----------------------------------------------------------------------------------------
def plot_animation(fig, fhour, times, frames):
    
    artists = []

    npanel = 0
    
    # gotta do some list mangling
    nrows  = len(frames)
    ncols  = np.int(sum( [ len(item) for item in frames]) / nrows)
    
    print("Frames will be %i rows and %i columns" % (nrows, ncols))
    
    for frow in frames: 
        
        for fitem in frow:
        
            print('Now processing Run: %s  for plottype:  %s' % (fitem[0], fitem[1].upper()))

            npanel += 1   # this is the location of each image over time in the plot

            ax = fig.add_subplot(nrows, ncols, npanel)

            plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, hspace = 0.1, wspace = 0.1)

            # now plot every image for the panel slot in ax.subplots()  Then we append all of those 
            # plots in time together in list ax_artists, and then append all 4 sequences into list artists[]

            ax_artists = []

            for t in times: 
                try:
                    image = read_wofs_png(fitem[0], fhour, fitem[1], t)
                    img = ax.imshow(image, animated=True)
                    txt = ax.set_title('%s   %s  f%3.3i min' %(os.path.split(fitem[0])[-1], fitem[1].upper(), t))
                    ax_artists.append([txt,img])
                except:
                    continue

            # Append each temporal sequence of frames for one subplot regions over time into artists[]

            artists.append(ax_artists)
            
    print('Created all the artist images, now reorganizing...')

    # Take the per-subplot panel lists of artists and join them together to get
    # one list, where each new list is the list of all articles to draw in a single frame
    # Example, if there are 4 panels, there are 4 members in the artists lists, each with a sequence
    # of images from the temporal sequence.  By using *zip, all 4 lists items are created in
    # 'panel_artists' as a 4 member list.  Then break these into single panels, and then attach
    # them together into a single new list called frame which now has a membership of all four panels
    # for each time.  So artists[] had a dimension of [nrows*ncols, len(time)], and now new_artists 
    # has a dimension of [len(time), nrows*ncols]

    
    new_artists = []
    for panel_artists in zip(*artists):
        frame = []
        for panel in panel_artists:
            frame.extend(panel)
        new_artists.append(frame)
        
    print('Completed creation of multi-panel sequence, returning...')
        
    # return the new list
    
    return new_artists

#-----------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-f", "--file", dest="file",    type="string", default="None",  \
                                    help = "Input config file to parse")

    parser.add_option(      "--doc", dest="doc",      type="string", default="DEFAULT",  \
                                    help = "Specific plot configuration to use in YAML multi-document config file")

    parser.add_option(      "--all", dest="all",  default=False, action="store_true", \
                                 help = "Boolean flag process all the documents in a file")

    (options, args) = parser.parse_args()
    
    if options.file == None:
        parser.print_help()
        print("\n ==> STITCH ERROR:  no input configuration filename!  EXITING!!!")
        sys.exit()
    else:
        with open(options.file) as file:
            all_params = yaml.load(file, Loader=yaml.FullLoader)


    # local function to create individual movies from multi-doc (plot) YAML files

    def create_movie(input_params):

        nrows    = input_params['nrows']
        ncols    = input_params['ncols']
        ftimes   = input_params['ftimes']
        fhours   = input_params['fhours']
        frames   = input_params['frames']
        out_dir  = input_params['output_movie'][0]['dir']
        out_name = input_params['output_movie'][1]['label']

        # These are needed for animation plots

        plt.rcParams['animation.html']        = input_params['rcParams'][0]['animation.html']  
        plt.rcParams['animation.embed_limit'] = input_params['rcParams'][1]['animation.embed_limit']  
        plt.rcParams['animation.ffmpeg_path'] = input_params['rcParams'][2]['animation.ffmpeg_path']  
    
        print("ffmpeg:  ", plt.rcParams['animation.ffmpeg_path'])  

        times  = ftimes[0] + ftimes[2]*(np.arange(1 + (ftimes[1]-ftimes[0])/ftimes[2]))

        for hour in fhours:
             
            fname_movie = ('%s/%s_%sUTC_Comparison.mp4' % (out_dir, out_name, hour[0:2]))
            print("Making:  %s" % fname_movie)

            fig    = plt.figure(figsize=(8.5*ncols, 9*nrows))

            with timeit():
                movie = plot_animation(fig, hour, times, frames)

            with timeit():
                my_anim = manimation.ArtistAnimation(fig, movie, interval=100, blit=True)

            with timeit():
                WriterClass = manimation.writers['ffmpeg']
                writer = WriterClass(fps=10, bitrate=1800)
                my_anim.save(fname_movie, writer=writer, dpi=72)

    # end create_movie function

    # if "--all" is False, just create a single set of movies for the forecast hours submitted. 

    if options.all == False:

        create_movie(all_params[options.doc])

    else:

    # if "--all" is True, create multiple movies for the forecast hours submitted. 
    # "key" is the header for the YAML document section ("W_MAX_90", "UH_2to5_90", etc).

        for key in all_params.keys():

            create_movie(all_params[key])
