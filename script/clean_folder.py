import os
import shutil

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size

if __name__ == "__main__":
    summary_path = "/mnt/data1/tyluan/workspace/code/DeepHandMesh/output/summary"
    for dir in os.listdir(summary_path):
        test_case = os.path.join(summary_path, dir)
        print(getdirsize(test_case))
        # if getdirsize(test_case) < 1000:
        #     shutil.rmtree(test_case)
