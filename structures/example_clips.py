example_clips = [
	{ 
	  'start': 0,
	  'options': {
	    'peak_duration': 10,
	    'video_source_rate': 60,
	    'image_source_rate': 20,
	    'peaks_enabled': False
	  }
	},
	{ 
	  'start': lambda total_frames: (4*int(total_frames/6)) - 80,
	  'options': {
	    'peak_duration': 50,
	    'video_source_rate': 20,
	    'image_source_rate': 10,
	    'peaks_enabled': True
	  }
	},
]

example_text_clips = [
	{ 
	  'start': 0,
	  'options': {
	  	'text_sprite': 'hola',
	    'peak_duration': 20,
	    'image_source_rate': 60,
	    'lyric_text_rate': 45,
	    'peaks_enabled': False,
	    'categories': ['happy'],
	  }
	},
	{ 
	  'start': lambda total_frames: (1*int(total_frames/2)) - 10,
	  'options': {
	  	'text_sprite': 'adios',
	    'peak_duration': 20,
	    'image_source_rate': 3,
	    'lyric_text_rate': 35,
	    'categories': ['happy'],
	    'peak_categories': ['ecstasy'],
	  }
	}
]