from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


# https://www.cairographics.org/samples/
# https://www.cairographics.org/cookbook/
class FontManager(object):
	"""docstring for FontManager"""
	def __init__(self, arg):
		super(FontManager, self).__init__()
		self.arg = arg
	
	# https://stackoverflow.com/questions/4458696/finding-out-what-characters-a-given-font-supports
	def check_font_characters(self):
		ttf = TTFont(sys.argv[1], 0, verbose=0, allowVID=0,
                ignoreDecompileErrors=True,
                fontNumber=-1)
		chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
		print(list(chars))
		# Use this for just checking if the font contains the codepoint given as
		# second argument:
		#char = int(sys.argv[2], 0)
		#print(Unicode[char])
		#print(char in (x[0] for x in chars))
		ttf.close()