import os
import shutil

def clean_pycache(directory):
    """递归清除所有__pycache__目录和.pyc文件"""
    count = 0
    
    for root, dirs, files in os.walk(directory):
        # 删除__pycache__目录
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
            count += 1
            
        # 删除.pyc文件
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                print(f"Removing: {pyc_path}")
                os.remove(pyc_path)
                count += 1
    
    return count

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned = clean_pycache(project_dir)
    print(f"\nCleanup complete! Removed {cleaned} cache files/directories.")
