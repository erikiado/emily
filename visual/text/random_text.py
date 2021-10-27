import random

import cv2 as cv


# def vortex(screenpos,i,nletters):
#     d = lambda t : 1.0/(0.3+t**8) #damping
#     a = i*np.pi/ nletters # angle of the movement
#     v = rotMatrix(a).dot([-1,0])
#     if i%2 : v[1] = -v[1]
#     return lambda t: screenpos+400*d(t)*rotMatrix(0.5*d(t)*a).dot(v)
    
# def cascade(screenpos,i,nletters):
#     v = np.array([0,-1])
#     d = lambda t : 1 if t<0 else abs(np.sinc(t)/(1+t**4))
#     return lambda t: screenpos+v*400*d(t-0.15*i)

# def arrive(screenpos,i,nletters):
#     v = np.array([-1,0])
#     d = lambda t : max(0, 3-3*t)
#     return lambda t: screenpos-400*v*d(t-0.2*i)
    
# def vortexout(screenpos,i,nletters):
#     d = lambda t : max(0,t) #damping
#     a = i*np.pi/ nletters # angle of the movement
#     v = rotMatrix(a).dot([-1,0])
#     if i%2 : v[1] = -v[1]
#     return lambda t: screenpos+400*d(t-0.1*i)*rotMatrix(-0.2*d(t)*a).dot(v)

class RandomText:
  stroke = None
  size = None
  x = None
  y = None
  r = None
  g = None
  b = None
  text = None
  master_height = 0
  master_width = 0
  test_run = False
  words = ['erikiado', 'erikiano', '@@@@@@', '######', '%%%%%%%', 'sube','baja','detente','sigue','escuchame','callate','sigueme','vete','explora']
  output_name = 'output.mp4'


  def __init__(self, master_width, master_height, test_run=False):
    self.master_height = master_height
    self.master_width = master_width
    self.update()
    self.text = ''
    self.test_run = test_run
  
  def update(self, text=None, stroke=None, size=None):
    self.x = random.randint(0,self.master_width - int(self.master_width/4))
    self.y = random.randint(0,self.master_height - int(self.master_height/5))
    self.r = random.randint(0,255)
    self.g = random.randint(0,255)
    self.b = random.randint(0,255)
    if size:
      self.size = size
    else:
      self.size = random.randint(1,4)
    if stroke:
      self.stroke = stroke
    else:
      self.stroke = random.randint(1,5)
    if text:
      self.text = text
    else:
      next_index = random.randint(0,len(self.words)-1)
      self.text = self.words[next_index]
  
  def put(self,frame):
    cv.putText(frame, self.text, (self.x, self.y), cv.FONT_HERSHEY_SIMPLEX, self.size , (self.r,self.g,self.b), self.stroke, cv.LINE_AA)
