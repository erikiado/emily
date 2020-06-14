from visual.video.video_editor import VideoEditor
from structures.example_clips import example_clips, example_text_clips

# <<<>>> should be updated with your paths
# categories are a list of strings that help filter images and videos that contain that string

if __name__ == "__main__":

	editor_configuration = dict(
	  file_paths=[
	    # '<<</absolute/path/to/resources_directory/>>>',
	  ],
	  output_path='<<</absolute/path/to/results>>>',
	  fonts_path='<<</absolute/path/to/fonts/>>>',
	  q='test', # 'test', '720', '1080'  
	  # image_source_rate=3,
	  # video_source_rate=30*8,
	  fps=30,
	  debug=True,
	  verbose=True
	)

	video_editor = VideoEditor(**editor_configuration)

	palette = [(255,255,255),(255,0,255),(0,255,255),(0,255,0)]
	background = (0,0,0)

	# Live playback of random videos and images from assets
	# video_editor.play_live_video()
	# video_editor.play_live_video(clips=example_clips)


	# Test a single filter
	# video_editor.test_filter('invert')
	# video_editor.test_filter('change_colors', ['change-colors'])


	# Generate a video based on audio length reacting to peaks
	# 	render lyrics based on timestamps
	# video_editor.generate_lyric_video('<<</absolute/path/to/projects/example_project/song.wav>>>',
	# 																	main_categories=['happy'], 
	#                                   peak_categories=['ecstasy'],
	# 																	lyrics='<<</absolute/path/to/projects/example_project/lyrics.txt>>>',
	# 																	colors=[
	# 																	  (0,0,0),
	# 																	  (255,255,255),
	# 																	  (255,0,0),
	# 																	],
	# 																	font_colors=[
	# 																    (0,0,0),
	# 																	  (255,255,255),
	# 																	  (255,255,255),
	# 																	  (255,255,255),
	# 																	  (255,255,255),
	# 																	  (255,255,255),
	# 																	  (255,0,0),
	# 																	  (255,0,0),
	# 																	  (255,0,0),
	# 																	  (255,0,0),
	# 																	],
	# 																	clips=example_clips)
	# video_editor.generate_lyric_video('<<</absolute/path/to/projects/example_project/song.wav>>>', clips=example_clips)
	# video_editor.generate_lyric_video('<<</absolute/path/to/projects/example_project/song.wav>>>')


	# Generate a text video with a solid background and a randomized font
	# video_editor.generate_text_sprite('erikiado', colors=palette, background=background)


	# Generate a randomized video of selected duration
	# video_editor.generate_b_roll(seconds=10, source_rate=5, image_source_rate=3)


	# Generate randomized video for audio duration
	# video_editor.generate_audio_roll('<<</absolute/path/to/projects/example_project/audio_clip.wav>>>')
	# video_editor.generate_audio_roll(
	# 	'<<</absolute/path/to/projects/example_project/song.wav>>>',
	# 	output_name='debuggeando',
	# 	clips=debuggeando_clips)


	# Generate short video configured by clips to be shared to instagram
	# video_editor.generate_instagram_video('erikiado64', example_text_clips, colors=palette, background=background)

	






























