
def timestamp_to_video_frame(fps, ts):
  minute, s_ms = ts.split(':')
  seconds, ms = s_ms.split('.')
  minute, seconds, ms = int(minute), int(seconds), int(ms)
  time_frame = ((minute * 60) + seconds) * fps
  time_frame += round(ms*fps/1000)
  return time_frame