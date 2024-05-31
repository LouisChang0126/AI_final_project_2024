import os
import google.generativeai as genai
import PIL.Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

genai.configure(api_key='AIzaSyDgzCCY2P-wWOMEr6M2OyMNIOpuWEpToW8')
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

mypath = '/media/user/新增磁碟區/user/Downloads (except program)/AI_final_store/AI_final_project_2024/AI_final_project_2024-linux'
extend_path = '/data/Arch_Dataset/'
file_path = os.listdir(mypath + extend_path)
#print(len(file_path))
#print(file_path[1924])
#temp = plt.imread(mypath + extend_path + '129.jpg')
#plt.imshow(temp)
#plt.show()
#temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
#temp = cv2.imread(mypath + extend_path + '129.jpg')
#print(temp)
#cv2.imshow('ing', temp)
#cv2.waitKey()

the_promt = "Tell me when was this architecture constructed and other imformation about it. Please answer in the following format: \n"
the_promt += "Year of the acrchitecture constructed : {The year of this architecture constructed. If you don't have an exact answer, guess it! Answer with an integer. If it is in BC year, use negative integer.}\n"
the_promt += "Longitude of the architecture : {The longitude of this architecture. If you don't have an exact answer, guess it! Answer with a floating number with ° E or ° W}\n"
the_promt += "Latitude of the architecture : {The latitude of this architecture. If you don't have an exact answer, guess it! Answer with a floating number with ° N or ° S}\n"
the_promt += "Name of architecture : {Name of this arcitecture. If you don't know the exact name, just give a short discription which no more than 8 words.}\n"
#the_promt += "Other discription of this architecture : {Any discription you want to add to this architecture.}"
#print(the_promt)

#data = [['picture', 'year', 'longitude', 'latitude', 'building', 'source']]
#data = list()

csv_file_path = 'data.csv'
save_image_path = '/data/Arch_dataset_done/'

for i in range(0, 10000):
    time.sleep(3)
    img = PIL.Image.open(mypath + extend_path + file_path[i])
    #plt.imshow(img)
    #cv2.imwrite(mypath + save_image_path + '{}.jpg'.format(i), cv2.cvtColor(plt.imread(mypath + extend_path + file_path[i]), cv2.COLOR_RGB2BGR))
    img = img.convert("RGB")
    img.save(mypath + save_image_path + '{}.jpg'.format(i))

    response = model.generate_content([the_promt, img], stream=True)
    response.resolve()

    print(response.text)
    print()

    architecture_string = response.text
    #print(architecture_string.split('\n'))

    architecture_list = []
    for line in architecture_string.split('\n'):
        if ': ' in line:
            architecture_list.append(line.split(': ')[1])
        else:
            pass
    #print(architecture_list)

    data = ['{}.jpg'.format(i)] + architecture_list + ['Gemini']

    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

print("已成功寫入到", csv_file_path)