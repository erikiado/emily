#! /usr/bin/python
import Image
 
# 'Python Imaging Library' template.  
# -a good reference site is http://effbot.org/imagingbook/
# -first line of code is for linux.  Change for windows or other OS
# -must make template.py executable .. give it permission to run as program.
# -i'll comment on the lines that are PIL specific.
# -PIL starts coordinate [0,0] at top left pixel.
# -place "test.jpg" image in same folder as code.
 
 
 
 
 
#_______________________________________________________load image/create 'canvas'
source = Image.open("test.jpg") #open an image for glitching (can be .png or other)
img = source.load()             #open an image for glitching (can be .png or other)
 
print source.format     #display source info for shits and giggles
print source.size       #display source info for shits and giggles
print source.mode       #display source info for shits and giggles
 
x = source.size[0]      #width of source image .. good for ending loops
y = source.size[1]      #height of source image .. good for ending loops
 
canvas = Image.new("RGB",(x,y),(255,255,255))   #create a second image to write to
img0 = canvas.load()                            #create a second image to write to
#creating a second image/canvas to write to means you don't have to overwrite the pixels on the source image.  That lets you reference the original values of pixels you have already worked on.
 
 
 
 
 
 
#_______________________________________________________run
i=10
j=10
 
pixel_value = img[i,j]  #read the value of pixel [10,10]. Gives output like (10,24,255)
r = img[i,j][0]         #read just red component value
g = img[i,j][1]         #read just green component value
b = img[i,j][2]         #read just blue component value
 
print "pixel_value =",pixel_value              
print "r =",r
print "g =",g
print "b =",b
 
img0[i,j] = (r,g,b)     #store RGB value of original image to same pixel on blank canvas
 
 
img0[(i+1),j] = (r,g,b) #assign pixel [10,10] value from above to [11,10]
img[(i+1),j] = (r,g,b)  #same as above but re-writes source image pixel (don't recommend doing)
 
 
 
 
 
 
 
#_______________________________________________________save
source.save("template.png")     #save img values as a .png file
canvas.save("template0.png")    #save img0 values as a .png file
 
# I often write only to img0 so I only use "canvas.save()".  If you write to img then do "source.save()"