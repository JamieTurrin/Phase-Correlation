# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 10:57:22 2015

@author: James Turrin
"""
#######################################################################################
#                               Phase correlation program

# INPUTS: 2 satellite images of same size, path/filename for output, window size, step size
# OUTPUT: COSI-Corr-style output file with 3 bands: E/W disp., N/S disp., and SNR.

########################################################################################
################################ Input parameters  #####################################

path1 = 'c:/users/james/desktop/testfiles/logan/logan_2001_subimage1.tif'  #first image
path2 = 'c:/users/james/desktop/testfiles/logan/logan_2001_subimage2.tif'  #2nd image
outfile = 'c:/users/james/desktop/testfiles/logan/logan_ws32_step16.tif'  #output file
txt_path = 'c:/users/james/desktop/testfiles/logan/logan_ws32_step16.txt'

ws = 32     #window size
step = 16    #step size (grid spacing)
spatial_res = 30.0  #spatial resolution of images
###########################################################################################
###########################################################################################

from read_tiff_file import read_tiff_file
from write_multiband_tiff_file2 import write_multiband_tiff_file2
import numpy as np
import math
from skimage import exposure

#read images, get columns, rows, bands
image1,cols,rows,bands = read_tiff_file(path1,dims=1)
image2 = read_tiff_file(path2)

#create text file to recieve results in list form
out_txt = open(txt_path,"a") # "a" means append
out_txt.write('X_disp   Y_disp   dx    dy    SNR\n') #\n creates carriage return

#calc. beginning and ending points of correlation process
begin_x = np.int(math.floor(ws/2)-1)
begin_y = np.int(math.floor(ws/2)-1)
end_x = np.int(cols-begin_x)
end_y = np.int(rows-begin_y)

#create output array
out_cols = np.int(math.floor((cols-ws)/step)+1)  #calc. # of columns for output file
out_rows = np.int(math.floor((rows-ws)/step)+1)  #calc. # of rows for output file
output = np.zeros((out_rows,out_cols,3))  #create array to hold displacement data
output[0:out_rows,0:out_cols,0:2] = np.NaN  #fill output with NaNs
m = 0  #counter for output rows
n = 0  #counter for output columns

print ''
print 'BEGINNING PHASE CORRELATION'
# Begin correlation process:
for i in range(begin_y,end_y,step):             #iterate over rows
   for j in range(begin_x,end_x,step):          #iterate over columns
   
       box1 = image1[i-begin_y:i+begin_y+1,j-begin_x:j+begin_x+1]   #extract ref. window
       box2 = image2[i-begin_y:i+begin_y+1,j-begin_x:j+begin_x+1]   #extract search window
       
#       box1 = exposure.equalize_hist(box1)  #equalize histograms.
#       box2 = exposure.equalize_hist(box2)  #this step may enhance results.       
       
       box1fft = np.fft.fft2(box1)       #convert to Fourier domain, yields complex array
       box2fft = np.fft.fft2(box2)       #convert to Fourier domain, yields complex array
       
       box1conj = np.conjugate(box1fft)          #convert box1 to its complex conjugate       
       norm_power_spect = (box1conj*box2fft)/(np.abs(box1fft)*np.abs(box2fft))
       
       box3 = np.fft.ifft2(norm_power_spect)    #inverse fft yields complex array
       box3_real = np.real(box3)  #get real portion of box3 (correlation surface)
       
       peak = np.nanmax(box3_real)   #get max value of correlation surface
       peak_index = np.argmax(box3_real)  #get 1-D index of max value from flattened peak array
       
       if peak_index == 0:
           dx = 0.0
           dy = 0.0
           peak_x = 0.0
           peak_y = 0.0
           snr = 1.0
       else:
           peak_col = np.size(box3_real,1)    #get # columns in correlation surface
           peak_row = np.size(box3_real,0)    #get # rows in correlation surface
           peak_y = np.float(math.floor(peak_index/peak_col))  #get row of max value in correlation surface
           peak_x = np.float(peak_index-(peak_col*peak_y))  #get column of max value in correlation surface
           
           snr = peak/np.nanmean(np.abs(box3_real))    #calc. SNR
           
           # subpixel estimation using Gaussian fitting function explained by
           # Argyriou and Vlachos, J. of Electronic Imaging, 16(3), 2007.
           # 'On the estimation of subpixel motion using phase correlation'
           if peak_x == 0 or peak_x >= peak_col-1:
               dx = 0.0
           else:
               dx_num = np.log(box3_real[peak_y,peak_x+1]) - np.log(box3_real[peak_y,peak_x-1])
               dx_denom = 2*(2*np.log(box3_real[peak_y,peak_x])-np.log(box3_real[peak_y,peak_x+1])-np.log(box3_real[peak_y,peak_x-1]))
               dx = dx_num/dx_denom #subpixel motion in x direction (East/West)
               if math.isnan(dx) == True:
                   dx = 0.0
           
           if peak_y == 0.0 or peak_y >= peak_row-1:
               dy = 0.0
           else:
               dy_num = np.log(box3_real[peak_y+1,peak_x]) - np.log(box3_real[peak_y-1,peak_x])
               dy_denom = 2*(2*np.log(box3_real[peak_y,peak_x])-np.log(box3_real[peak_y+1,peak_x])-np.log(box3_real[peak_y-1,peak_x]))
               dy = dy_num/dy_denom #subpixel motion in y direction (North/South)
               if math.isnan(dy) == True:
                   dy = 0.0

       if np.abs(peak_x) > ws/2:
           peak_x = peak_x - ws +1  #convert x offsets > ws/2 to negative offsets
       if np.abs(peak_y) > ws/2:
           peak_y = peak_y - ws +1  #convert y offsets > ws/2 to negative offsets

       output[m,n,0] = (peak_x + dx)*spatial_res  #E/W displacement
       output[m,n,1] = (peak_y + dy)*spatial_res  #N/S displacement
       output[m,n,2] = snr                #SNR 
       n = n+1              #iterate counter for columns
       
       if np.abs(peak_x) > 0.0 and np.abs(peak_y) > 0.0:
           out_txt.write(" %f %f %f %f %f\n" %(peak_x,peak_y,dx,dy,snr)) #\n creates carriage return
       
   if i%100 == 0:
       percent_complete = math.floor(100*float(i)/float(rows-1))
       print 'Phase correlation is',percent_complete, '% complete'
           
   m = m+1 #iterate counter for rows
   n = 0  #reset counter for columns to zero     

out_txt.close()  #close text file
       
# create output tiff file
write_multiband_tiff_file2(outfile,output,out_cols,out_rows,3)

print ''
print 'PHASE CORRELATION IS COMPLETE'





















