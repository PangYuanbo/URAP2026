from pathlib import Path
clips={2,9,10,17,21,22,24,35}
root=Path(r'D:\URAP_datasets\TransVisDrone\NPS\AllFrames\train')
out=Path(r'C:\Users\aaron\Desktop\URAP\data_templates\nps_hard_train_v104.txt')
files=sorted(path for path in root.glob('*.png') if int(path.stem.split('_')[1]) in clips)
out.write_text('\n'.join(str(path) for path in files)+'\n',encoding='utf8')
print({'clips':sorted(clips),'images':len(files),'output':str(out)})
