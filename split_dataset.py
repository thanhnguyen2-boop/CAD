import os
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

def split_dataset(src_dir: str, dst_dir_pmi: str, dst_dir_other: str):
    src_dir = Path(src_dir)
    dst_dir_pmi = Path(dst_dir_pmi)
    dst_dir_other = Path(dst_dir_other)
    
    # Create directories
    for dst in [dst_dir_pmi, dst_dir_other]:
        for split in ['train', 'valid', 'test']:
            os.makedirs(dst / split / 'images', exist_ok=True)
            os.makedirs(dst / split / 'labels', exist_ok=True)
            
    splits = ['train', 'valid', 'test']
    
    # Original classes: 0: Isometric, 1: PMI, 2: Table, 3: Text, 4: View
    # PMI mapping: 1 -> 0
    # Other mapping: 0->0, 2->1, 3->2, 4->3
    other_mapping = {0: 0, 2: 1, 3: 2, 4: 3}
    
    for split in splits:
        print(f"Processing {split}...")
        src_labels_dir = src_dir / split / 'labels'
        src_images_dir = src_dir / split / 'images'
        
        if not src_labels_dir.exists():
            continue
            
        for label_file in tqdm(list(src_labels_dir.glob('*.txt'))):
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            pmi_lines = []
            other_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                rest = " ".join(parts[1:])
                
                if cls_id == 1:
                    pmi_lines.append(f"0 {rest}\n")
                elif cls_id in other_mapping:
                    other_lines.append(f"{other_mapping[cls_id]} {rest}\n")
                    
            # Write to PMI dataset
            with open(dst_dir_pmi / split / 'labels' / label_file.name, 'w', encoding='utf-8') as f:
                f.writelines(pmi_lines)
                
            # Write to Other dataset
            with open(dst_dir_other / split / 'labels' / label_file.name, 'w', encoding='utf-8') as f:
                f.writelines(other_lines)
                
            # Copy images
            img_name = label_file.stem + '.jpg'
            src_img = src_images_dir / img_name
            if not src_img.exists():
                img_name = label_file.stem + '.png'
                src_img = src_images_dir / img_name
                
            if src_img.exists():
                shutil.copy2(src_img, dst_dir_pmi / split / 'images' / img_name)
                shutil.copy2(src_img, dst_dir_other / split / 'images' / img_name)
                
    # Write yaml files
    yaml_pmi = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 1,
        'names': ['PMI']
    }
    with open(dst_dir_pmi / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_pmi, f, sort_keys=False)
        
    yaml_other = {
        'train': '../train/images',
        'val': '../valid/images',
        'test': '../test/images',
        'nc': 4,
        'names': ['Isometric', 'Table', 'Text', 'View']
    }
    with open(dst_dir_other / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_other, f, sort_keys=False)

if __name__ == '__main__':
    src = r'c:\Users\thanh.nguyen2\Documents\project-python\cad\data\cad-drawing-v5'
    dst_pmi = r'c:\Users\thanh.nguyen2\Documents\project-python\cad\data\cad-drawing-pmi'
    dst_other = r'c:\Users\thanh.nguyen2\Documents\project-python\cad\data\cad-drawing-other'
    split_dataset(src, dst_pmi, dst_other)
    print("Done!")
