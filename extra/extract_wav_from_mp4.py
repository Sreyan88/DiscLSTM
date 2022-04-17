import subprocess
import glob
from tqdm import tqdm
for filename in tqdm(glob.glob('output_repeated_splits_test/*')):
    vid_name = ''.join(filename.split('/')[-1].split('.')[:-1])
    process = subprocess.run(['ffmpeg','-i',f'{filename}','-vn',f'test_splits_audio/{vid_name}.wav'])
    # print(process.stdout)
