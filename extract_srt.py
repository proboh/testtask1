import ffmpeg

# Extract subtitles from the video
ffmpeg.input('video1.mp4').output('subtitles.srt', map='0:s:0').run()
