
# detectar bordes al color

class Palette():
    def __init__(self, palette_img_src, brightness_func):
        self.palette = {}
        with Image.open(palette_img_src) as palette_img:
            palette_img_height = palette_img.size[1]
            list(tqdm(self._build_palette(palette_img, brightness_func), total=palette_img_height))
        self.sorted_keys = sorted(self.palette.keys())
    def _build_palette(self, palette_img, brightness_func):
        width, height = palette_img.size
        palette_pixels = palette_img.load()  # getting PixelAccess
        for j in range(height):
            for i in range(width):
                brightness = brightness_func(palette_pixels[i, j])
                self.palette[brightness] = palette_pixels[i, j] # nothing smart (ex: avg): we only keep the last brightness value processed
            yield 'ROW_COMPLETE' # progress tracking
    def __getitem__(self, key):
        i = bisect_left(self.sorted_keys, key)  # O(logN)
        return self.palette[self.sorted_keys[i]]


  def noisy(self,noise_typ,image):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      # mean = 0
      # var = 0.2
      # sigma = var**0.5
      mean = 0.5
      var = 0.7
      sigma = var**0.8
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy


  def steal_colors(img, palette_img_src):
      print('# Building replacement palette')
      palette = Palette(palette_img_src, brightness_func=luminosity)

      print('# Subsituting colors')
      img_height = img.size[1]
      list(tqdm(subst_img_colors(img, palette, brightness_func=luminosity), total=img_height))


  def subst_img_colors(img, luminosity2color_palette, brightness_func):
      width, height = img.size
      img = img.load()  # getting PixelAccess
      for j in range(height):
          for i in range(width):
              img[i, j] = luminosity2color_palette[brightness_func(img[i, j])]
          yield 'ROW_COMPLETE' # progress tracking


  def luminosity(pixel):
      r, g, b = pixel[:3]
      return 0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)

  
  def tearY(self, frame):
    size = frame.shape
    sizex = size[1]
    block = sizex/10
    blockx = random.random()*10*block
    frame[:,blockx:blockx+block,:] = 0
    return frame

    

