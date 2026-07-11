from pathlib import Path
import json, os, random, shutil
root=Path(r'D:\URAP_datasets\TransVisDrone\NPS\AllFrames\train')
original=Path(r'D:\URAP_datasets\TransVisDrone\NPS\NPSvisdroneStyle\train\labels')
corrected=Path(r'C:\Users\aaron\Desktop\URAP\data_templates\nps_hard_annotations_v104\train\labels')
out_list=Path(r'C:\Users\aaron\Desktop\URAP\data_templates\nps_hard_replay_v165.txt')
out_root=Path(r'D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165\annotations')
out_labels=out_root/'train'/'labels';out_labels.mkdir(parents=True,exist_ok=True)
hard={2,9,10,17,21,22,24,35}
by_clip={}
for path in root.glob('*.png'):
    clip=int(path.stem.split('_')[1]);by_clip.setdefault(clip,[]).append(path)
selected=[]
for clip,paths in sorted(by_clip.items()):
    paths=sorted(paths)
    if clip in hard:selected.extend(paths)
nonhard=[clip for clip in sorted(by_clip) if clip not in hard]
target=12000;base=target//len(nonhard);extra=target%len(nonhard)
for position,clip in enumerate(nonhard):
    paths=sorted(by_clip[clip]);count=min(len(paths),base+(position<extra));indices=[round(i*(len(paths)-1)/max(1,count-1)) for i in range(count)] if count else []
    selected.extend(paths[index] for index in sorted(set(indices)))
selected=sorted(set(selected))
for image in selected:
    clip=int(image.stem.split('_')[1]);destination=out_labels/(image.stem+'.txt')
    if destination.exists():continue
    source=(corrected if clip in hard else original)/(image.stem+'.txt')
    if source.exists():
        shutil.copy2(source,destination)
out_list.write_text('\n'.join(str(path) for path in selected)+'\n',encoding='utf-8')
summary={'hard_clips':sorted(hard),'hard_images':sum(len(by_clip[c]) for c in hard),'replay_target':target,'selected_images':len(selected),'label_files':sum(1 for _ in out_labels.glob('*.txt')),'list':str(out_list),'annotation_root':str(out_root)}
Path(r'C:\Users\aaron\Desktop\URAP\data_templates\nps_hard_replay_v165_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


