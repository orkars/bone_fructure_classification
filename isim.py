import os
images_path = r"/home/orka/Desktop/croped_bone_project/croped/Fract"
images_path2 = r"/home/orka/Desktop/croped_bone_project/croped/Fracture"
image_list = os.listdir(images_path)
i=0
for image in image_list:
    ext = os.path.splitext(image)[1]
    ext2 = os.path.splitext(image)[0]
    if ext == '.JPG' or '.jpeg' or '.JPEG':
        src = images_path + '/' + image
        dst = images_path2 + '/' + str(i) + ext
        os.rename(src, dst)
    i+=1
